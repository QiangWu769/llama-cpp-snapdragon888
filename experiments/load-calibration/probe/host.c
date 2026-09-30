#include <inttypes.h>
#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "hmx_load_calibration.h"
#include "calibration.h"

_Static_assert(sizeof(struct calibration_report)<CALIBRATION_REPORT_BYTES,"report exceeds mapped buffer");
static uint64_t monotonic_ns(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000u+t.tv_nsec;}
static uint64_t cntvct(void) {uint64_t value;asm volatile("isb\nmrs %0, cntvct_el0":"=r"(value));return value;}
static uint64_t cntfrq(void) {uint64_t value;asm volatile("mrs %0, cntfrq_el0":"=r"(value));return value;}
static uint32_t number(const char *s) {
    char *end;unsigned long value=strtoul(s,&end,10);
    if(!*s || *end || value>UINT32_MAX){fprintf(stderr,"invalid integer\n");exit(2);}return (uint32_t)value;
}
static uint64_t digest(const void *buffer,size_t bytes) {
    const unsigned char *p=buffer;uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
int main(int argc,char **argv) {
    setbuf(stdout,NULL);setbuf(stderr,NULL);
    unsigned mode=CALIBRATION_HMX,seconds_requested=20,period_us=1000000,duty=100,chunk=0,workers=1;
    for(int i=1;i<argc;++i) {
        if(i+1>=argc){fprintf(stderr,"missing argument\n");return 2;}
        const char *key=argv[i++],*value=argv[i];
        if(!strcmp(key,"--mode")) {
            if(!strcmp(value,"scalar"))mode=CALIBRATION_SCALAR;
            else if(!strcmp(value,"hvx"))mode=CALIBRATION_HVX;
            else if(!strcmp(value,"hmx"))mode=CALIBRATION_HMX;
            else return 2;
        } else if(!strcmp(key,"--seconds"))seconds_requested=number(value);
        else if(!strcmp(key,"--period-us"))period_us=number(value);
        else if(!strcmp(key,"--duty"))duty=number(value);
        else if(!strcmp(key,"--chunk"))chunk=number(value);
        else if(!strcmp(key,"--workers"))workers=number(value);
        else {fprintf(stderr,"unknown option %s\n",key);return 2;}
    }
    if(!chunk)chunk=mode==CALIBRATION_HMX?32:mode==CALIBRATION_HVX?4096:16384;
    uint64_t duration=(uint64_t)seconds_requested*1000000;
    if(duration<1000000 || duration>60000000 || period_us<250000 || period_us>5000000 ||
       duration%period_us || duration/period_us>CALIBRATION_MAX_PERIODS ||
       (duty!=0 && duty!=25 && duty!=50 && duty!=100) || chunk>65536 || workers!=1) {
        fprintf(stderr,"usage: %s --mode scalar|hvx|hmx --seconds 1..60 --duty 0|25|50|100\n"
                       " --period-us 250000..5000000 (exact duration divisor; <=128 periods)\n"
                       " --chunk 1..65536 --workers 1\n",argv[0]);return 2;
    }
    alarm(seconds_requested+20);
    struct remote_rpc_control_unsigned_module ctrl={.domain=3,.enable=1};
    int setup=remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&ctrl,sizeof(ctrl));
    if(setup){fprintf(stderr,"unsigned PD status=%d\n",setup);return 2;}
    char uri[512];snprintf(uri,sizeof(uri),"%s&_dom=cdsp",hmx_load_calibration_URI);
    remote_handle64 h;
    int opened=hmx_load_calibration_open(uri,&h);
    if(opened){fprintf(stderr,"open status=%d\n",opened);return 2;}
    struct calibration_report *p=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,CALIBRATION_REPORT_BYTES);
    if(!p)return 2;
    memset(p,0,CALIBRATION_REPORT_BYTES);
    p->magic=CALIBRATION_MAGIC;p->version=CALIBRATION_VERSION;p->report_size=sizeof(*p);
    p->mode=mode;p->duration_us=duration;p->period_us=period_us;p->duty_percent=duty;p->chunk_iterations=chunk;p->workers=workers;
    int fd=rpcmem_to_fd(p),mapped=fd<0?-1:fastrpc_mmap(3,fd,p,0,CALIBRATION_REPORT_BYTES,FASTRPC_MAP_FD);
    if(mapped){fprintf(stderr,"map status=%d\n",mapped);return 2;}
    __sync_synchronize();uint64_t before_ns=monotonic_ns(),before_counter=cntvct();
    /* Exactly one compute RPC, regardless of duration or duty cycle. */
    int rpc=hmx_load_calibration_run(h,fd,CALIBRATION_REPORT_BYTES);
    uint64_t after_counter=cntvct(),after_ns=monotonic_ns();__sync_synchronize();
    if(rpc || p->result || p->stage!=5) {
        fprintf(stderr,"failed rpc=%d result=%d stage=%d; mappings retained until process teardown\n",rpc,p->result,p->stage);
        _Exit(3);
    }
    unsigned bad=0;
    uint64_t completed=p->active_iterations+p->warmup_iterations;
    if(mode==CALIBRATION_SCALAR)for(unsigned i=0;i<4;++i)bad+=p->scalar_values[i]!=calibration_expected(i*32,completed);
    if(mode==CALIBRATION_HVX)for(unsigned i=0;i<128;++i)bad+=p->hvx_values[i]!=calibration_expected(i,completed);
    if(mode==CALIBRATION_HMX)for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c) {
        float expected=calibration_hmx_expected(r,c);
        bad+=!isfinite(p->hmx_values[r*32+c]) || p->hmx_values[r*32+c]!=expected;
    }
    uint64_t elapsed=p->measurement_end_qtimer-p->measurement_start_qtimer;
    if(p->period_count!=duration/period_us || elapsed<duration*192/10 || !elapsed ||
       (duty && !p->active_iterations) || (!duty && p->active_iterations))++bad;
    uint64_t sum_active=0,sum_iterations=0;
    for(unsigned i=0;i<p->period_count;++i){sum_active+=p->periods[i].active_qtimer;sum_iterations+=p->periods[i].iterations;}
    if(sum_active!=p->active_qtimer || sum_iterations!=p->active_iterations || p->active_qtimer>elapsed)++bad;
    const char *name=mode==CALIBRATION_SCALAR?"scalar":mode==CALIBRATION_HVX?"hvx":"hmx";
    const void *output=mode==CALIBRATION_SCALAR?(void*)p->scalar_values:mode==CALIBRATION_HVX?(void*)p->hvx_values:(void*)p->hmx_values;
    size_t output_bytes=mode==CALIBRATION_SCALAR?sizeof(p->scalar_values):mode==CALIBRATION_HVX?sizeof(p->hvx_values):sizeof(p->hmx_values);
    printf("{\"schema_version\":1,\"build_uuid\":\""CALIBRATION_BUILD_UUID"\",\"mode\":\"%s\",\"workers\":1,\"compute_rpcs\":1,\"passed\":%s,\"mismatches\":%u,"
           "\"requested_duration_us\":%u,\"period_us\":%u,\"requested_duty_percent\":%u,\"chunk_iterations\":%u,"
           "\"host_rpc_start_monotonic_ns\":%"PRIu64",\"host_rpc_end_monotonic_ns\":%"PRIu64","
           "\"host_rpc_start_cntvct\":%"PRIu64",\"host_rpc_end_cntvct\":%"PRIu64",\"host_cntfrq_hz\":%"PRIu64","
           "\"rpc_wall_seconds\":%.9f,\"qtimer_hz\":19200000,\"qtimer_start\":%"PRIu64",\"qtimer_end\":%"PRIu64","
           "\"elapsed_qtimer\":%"PRIu64",\"active_qtimer\":%"PRIu64",\"measured_active_region_percent\":%.9f,"
           "\"elapsed_pcycles\":%"PRIu64",\"active_pcycles\":%"PRIu64",\"pcycles_available\":%s,"
           "\"active_iterations\":%"PRIu64",\"warmup_iterations\":%"PRIu64",\"output_fnv1a64\":\"%016"PRIx64"\","
           "\"vtcm_bytes\":%u,\"total_vtcm\":%u,\"available_vtcm\":%u,"
           "\"power_vote_status\":%d,\"power_up_status\":%d,\"lock_status\":%d,\"unlock_status\":%d,"
           "\"release_status\":%d,\"power_down_status\":%d,\"dcvs_reset_status\":%d,\"sleep_status\":%d,\"periods\":[",
           name,bad?"false":"true",bad,p->duration_us,p->period_us,p->duty_percent,p->chunk_iterations,
           before_ns,after_ns,before_counter,after_counter,cntfrq(),(after_ns-before_ns)*1e-9,
           p->measurement_start_qtimer,p->measurement_end_qtimer,elapsed,p->active_qtimer,
           100.0*p->active_qtimer/elapsed,p->measurement_end_pcycles-p->measurement_start_pcycles,p->active_pcycles,
           p->measurement_end_pcycles>p->measurement_start_pcycles?"true":"false",p->active_iterations,
           p->warmup_iterations,digest(output,output_bytes),p->vtcm_bytes,p->total_vtcm,p->available_vtcm,
           p->power_vote_status,p->power_up_status,p->lock_status,p->unlock_status,p->release_status,
           p->power_down_status,p->dcvs_reset_status,p->sleep_status);
    for(unsigned i=0;i<p->period_count;++i) {
        const struct calibration_period *r=p->periods+i;
        printf("%s{\"index\":%u,\"start_qtimer\":%"PRIu64",\"end_qtimer\":%"PRIu64",\"active_qtimer\":%"PRIu64","
               "\"elapsed_pcycles\":%"PRIu64",\"active_pcycles\":%"PRIu64",\"iterations\":%"PRIu64"}",
               i?",":"",i,r->start_qtimer,r->end_qtimer,r->active_qtimer,r->elapsed_pcycles,r->active_pcycles,r->iterations);
    }
    printf("]}\n");
    int unmapped=fastrpc_munmap(3,fd,p,CALIBRATION_REPORT_BYTES);
    if(unmapped){fprintf(stderr,"host unmap status=%d; buffer retained\n",unmapped);_Exit(3);}
    rpcmem_free(p);int closed=hmx_load_calibration_close(h);
    if(closed){fprintf(stderr,"close status=%d\n",closed);return 3;}
    return bad?1:0;
}
