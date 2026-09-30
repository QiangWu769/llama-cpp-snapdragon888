#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <string.h>
#include "hmx_load_calibration.h"
#include "calibration.h"

static int power_client;
static int running;
void calibration_hvx_chunk(uint32_t *, unsigned);
static uint32_t hvx_values[128] __attribute__((aligned(128)));

__attribute__((noinline)) static void scalar_chunk(uint32_t *values, unsigned iterations) {
    uint32_t a=values[0],b=values[1],c=values[2],d=values[3];
    for(unsigned i=0;i<iterations;++i)
        asm volatile("%0 = add(%0, %4)\n%1 = add(%1, %5)\n%2 = add(%2, %6)\n%3 = add(%3, %7)"
                     : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
                     : "r"(calibration_increment(0)), "r"(calibration_increment(1)),
                       "r"(calibration_increment(2)), "r"(calibration_increment(3)));
    values[0]=a;values[1]=b;values[2]=c;values[3]=d;
}
__attribute__((noinline)) static void hmx_chunk(void *a, void *b, void *c, unsigned iterations) {
    for(unsigned i=0;i<iterations;++i) {
        asm volatile("mxclracc.hf\n"
                     "{ activation.hf = mxmem(%0, %3):deep\nweight.hf = mxmem(%1, %3) }\n"
                     "mxmem(%2, %4):after.hf = acc"
                     :: "r"(a), "r"(b), "r"(c), "r"(65535), "r"(2047) : "memory");
    }
    asm volatile("barrier" ::: "memory");
}
static void chunk(struct calibration_report *p, void *a, void *b, void *c, unsigned count) {
    if(p->mode==CALIBRATION_SCALAR) scalar_chunk(p->scalar_values,count);
    else if(p->mode==CALIBRATION_HVX) calibration_hvx_chunk(hvx_values,count);
    else hmx_chunk(a,b,c,count);
}
static int valid(const struct calibration_report *p) {
    return p->magic==CALIBRATION_MAGIC && p->version==CALIBRATION_VERSION && p->report_size==sizeof(*p) &&
      p->mode<=CALIBRATION_HMX && p->workers==1 && p->duration_us>=1000000u && p->duration_us<=60000000u &&
      p->period_us>=250000u && p->period_us<=5000000u && p->duration_us%p->period_us==0 &&
      p->duration_us/p->period_us<=CALIBRATION_MAX_PERIODS &&
      (p->duty_percent==0 || p->duty_percent==25 || p->duty_percent==50 || p->duty_percent==100) &&
      p->chunk_iterations>0 && p->chunk_iterations<=65536u;
}
AEEResult hmx_load_calibration_open(const char *uri, remote_handle64 *h) { (void)uri;*h=1;return 0; }
AEEResult hmx_load_calibration_close(remote_handle64 h) { (void)h;return 0; }
AEEResult hmx_load_calibration_run(remote_handle64 h,int32 fd,int32 bytes) {
    (void)h;
    if(bytes<(int)sizeof(struct calibration_report))return AEE_EBADPARM;
    struct calibration_report *p=NULL;
    int err=HAP_mmap_get(fd,(void**)&p,NULL);
    if(err)return err;
    qurt_mem_cache_clean((qurt_addr_t)p,sizeof(*p),QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
    if(!valid(p)){HAP_mmap_put(fd);return AEE_EBADPARM;}
    if(!__sync_bool_compare_and_swap(&running,0,1)){HAP_mmap_put(fd);return AEE_EFAILED;}
    unsigned resource=0;
    int powered=0,locked=0;
    void *a=NULL,*b=NULL,*c=NULL;
    HAP_power_request_t power;
    p->stage=1;p->result=0;
    p->power_up_status=p->lock_status=p->unlock_status=p->release_status=p->power_down_status=CALIBRATION_NOT_ATTEMPTED;
    p->sleep_status=0;
    memset(&power,0,sizeof(power));power.type=HAP_power_set_DCVS_v3;
    power.dcvs_v3.dcvs_enable=TRUE;power.dcvs_v3.dcvs_option=HAP_DCVS_V2_PERFORMANCE_MODE;
    power.dcvs_v3.set_latency=TRUE;power.dcvs_v3.latency=100;
    power.dcvs_v3.set_core_params=TRUE;
    power.dcvs_v3.core_params.min_corner=HAP_DCVS_VCORNER_NOM;
    power.dcvs_v3.core_params.max_corner=HAP_DCVS_VCORNER_TURBO_L3;
    power.dcvs_v3.core_params.target_corner=HAP_DCVS_VCORNER_TURBO_L3;
    err=p->power_vote_status=HAP_power_set(&power_client,&power);if(err)goto done;
    for(unsigned i=0;i<4;++i)p->scalar_values[i]=calibration_initial(i*32u);
    for(unsigned i=0;i<128;++i)hvx_values[i]=calibration_initial(i);
    if(p->mode==CALIBRATION_HVX) {
        memset(&power,0,sizeof(power));power.type=HAP_power_set_HVX;power.hvx.power_up=TRUE;
        err=p->power_up_status=HAP_power_set(&power_client,&power);if(err)goto done;powered=1;
        err=p->lock_status=qurt_hvx_lock(QURT_HVX_MODE_128B);if(err)goto done;locked=1;
    } else if(p->mode==CALIBRATION_HMX) {
        memset(&power,0,sizeof(power));power.type=HAP_power_set_HMX;power.hmx.power_up=TRUE;
        err=p->power_up_status=HAP_power_set(&power_client,&power);if(err)goto done;powered=1;
        compute_res_attr_t attr;
        compute_res_vtcm_page_t total_pages,available_pages;
        unsigned int total_size,available_size;
        err=HAP_compute_res_query_VTCM(0,&total_size,&total_pages,&available_size,&available_pages);if(err)goto done;
        p->total_vtcm=total_size;p->available_vtcm=available_size;
        if(p->available_vtcm<CALIBRATION_VTCM_BYTES){err=AEE_ENOMEMORY;goto done;}
        err=HAP_compute_res_attr_init(&attr);if(err)goto done;
        err=HAP_compute_res_attr_set_hmx_param(&attr,1);if(err)goto done;
        err=HAP_compute_res_attr_set_vtcm_param(&attr,CALIBRATION_VTCM_BYTES,1);if(err)goto done;
        resource=HAP_compute_res_acquire(&attr,100000);if(!resource){err=AEE_EFAILED;goto done;}
        unsigned char *vtcm=HAP_compute_res_attr_get_vtcm_ptr(&attr);
        if(!vtcm || ((uintptr_t)vtcm&255)){err=AEE_EFAILED;goto done;}
        p->vtcm_bytes=CALIBRATION_VTCM_BYTES;
        a=vtcm;b=vtcm+65536;c=vtcm+131072;
        unsigned char *bias=vtcm+133120;
        memset(vtcm,0,CALIBRATION_VTCM_BYTES);
        for(unsigned tile=0;tile<32;++tile) for(unsigned row=0;row<32;++row) for(unsigned col=0;col<32;++col) {
            ((__fp16 *)a)[tile*1024+(row/2)*64+col*2+row%2]=(__fp16)((row+1)/1024.0f);
            ((__fp16 *)b)[tile*1024+(row/2)*64+col*2+row%2]=(__fp16)(((int)(col%7)-3)/4.0f);
        }
        for(unsigned i=0;i<1024;++i)((uint16_t*)c)[i]=0x7e00;
        qurt_mem_cache_clean((qurt_addr_t)vtcm,CALIBRATION_VTCM_BYTES,QURT_MEM_CACHE_FLUSH_INVALIDATE,QURT_MEM_DCACHE);
        err=p->lock_status=HAP_compute_res_hmx_lock(resource);if(err)goto done;locked=1;
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    }
    p->stage=2;
    chunk(p,a,b,c,1);p->warmup_iterations=1;
    /* Resources, packing and the sanity operation are outside this window. */
    p->measurement_start_qtimer=HAP_perf_get_qtimer_count();
    p->measurement_start_pcycles=HAP_perf_get_pcycles();
    if(!p->measurement_start_qtimer){err=AEE_EFAILED;goto done;}
    p->stage=3;
    const uint64_t period_ticks=(uint64_t)p->period_us*192/10;
    const unsigned count=p->duration_us/p->period_us;
    for(unsigned i=0;i<count;++i) {
        struct calibration_period *r=p->periods+i;
        const uint64_t period_end=p->measurement_start_qtimer+(i+1)*period_ticks;
        const uint64_t active_end=p->measurement_start_qtimer+i*period_ticks+period_ticks*p->duty_percent/100;
        r->start_qtimer=HAP_perf_get_qtimer_count();
        const uint64_t start_cycles=HAP_perf_get_pcycles();
        uint64_t active_start=r->start_qtimer,active_cycles=start_cycles;
        while(HAP_perf_get_qtimer_count()<active_end) {
            chunk(p,a,b,c,p->chunk_iterations);
            r->iterations+=p->chunk_iterations;
        }
        uint64_t after_active=HAP_perf_get_qtimer_count();
        if(r->iterations) {
            r->active_qtimer=after_active-active_start;
            r->active_pcycles=HAP_perf_get_pcycles()-active_cycles;
        }
        while(after_active<period_end) {
            uint64_t sleep_us=(period_end-after_active)*10/192;
            if(!sleep_us)sleep_us=1;
            err=p->sleep_status=HAP_timer_sleep(sleep_us);if(err)goto done;
            after_active=HAP_perf_get_qtimer_count();
        }
        r->end_qtimer=HAP_perf_get_qtimer_count();
        r->elapsed_pcycles=HAP_perf_get_pcycles()-start_cycles;
        p->active_qtimer+=r->active_qtimer;p->active_pcycles+=r->active_pcycles;
        p->active_iterations+=r->iterations;p->period_count=i+1;
    }
    p->measurement_end_pcycles=HAP_perf_get_pcycles();
    p->measurement_end_qtimer=HAP_perf_get_qtimer_count();
    p->stage=4;
    if(p->mode==CALIBRATION_HVX)memcpy(p->hvx_values,hvx_values,sizeof(hvx_values));
    if(p->mode==CALIBRATION_HMX) {
        qurt_mem_cache_clean((qurt_addr_t)c,2048,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
        for(unsigned r=0;r<32;++r)for(unsigned n=0;n<32;++n)
            p->hmx_values[r*32+n]=(float)((__fp16 *)c)[(r/2)*64+n*2+r%2];
    }
done:
    p->result=err;
    if(locked)p->unlock_status=p->mode==CALIBRATION_HMX?HAP_compute_res_hmx_unlock(resource):qurt_hvx_unlock();
    if(resource)p->release_status=HAP_compute_res_release(resource);
    if(powered) {
        memset(&power,0,sizeof(power));power.type=p->mode==CALIBRATION_HMX?HAP_power_set_HMX:HAP_power_set_HVX;
        if(p->mode==CALIBRATION_HMX)power.hmx.power_up=FALSE;else power.hvx.power_up=FALSE;
        p->power_down_status=HAP_power_set(&power_client,&power);
    }
    HAP_power_set_dcvs_v3_init(&power);p->dcvs_reset_status=HAP_power_set(&power_client,&power);
    if(!err && locked && p->unlock_status)err=p->unlock_status;
    if(!err && resource && p->release_status)err=p->release_status;
    if(!err && powered && p->power_down_status)err=p->power_down_status;
    if(!err && p->dcvs_reset_status)err=p->dcvs_reset_status;
    p->result=err;
    if(!err)p->stage=5;
    asm volatile("barrier" ::: "memory");
    qurt_mem_cache_clean((qurt_addr_t)p,sizeof(*p),QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
    int put_status=HAP_mmap_put(fd);
    __sync_lock_release(&running);
    return err?err:put_status;
}
