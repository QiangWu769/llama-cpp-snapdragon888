#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hmx_probe.h"
#include "probe.h"

static float half_round(double value) { return (float)(__fp16)value; }
static float half_bits(uint16_t raw) { __fp16 value;memcpy(&value,&raw,2);return (float)value; }
static double reference(const struct probe_report *p,unsigned o,unsigned r,unsigned n) {
    if(p->feature==EXPLICIT_CLEAR)return 0;
    if(p->feature==CLEAR_BOTH&&o>=2)return 0;
    if(p->feature==EXPLICIT_SWAP&&o==0)return 0;
    unsigned group=p->feature==WEIGHT_DEEP?o:0;
    unsigned ar=r,at=0;
    if(p->feature==SINGLE) {
        // Tests use mask 28 (YYYXX), so adding the encoded spatial offset
        // moves complete X rows and wraps into the second spatial crouton.
        unsigned shift=p->offset;
        ar=(r+shift)%32;at=(r+shift)/32;
        if(p->variant==1)at=1-at;
    }
    double ref=0;
    for(unsigned k=p->start;k<=(p->tiles-1)*32+p->stop;++k) {
        unsigned t=p->feature==SINGLE?at:k/32;
        ref+=(double)activation_value(t,ar,k%32,p->pattern)*
             weight_value(k/32,k%32,n,group,p->pattern);
    }
    if(p->feature==WEIGHT_NEGATE)ref=-ref;
    if(p->feature==ACCUMULATE||(p->feature==RETAIN&&o==1))ref*=2;
    if(p->feature==CLEAR_BOTH&&o==1)ref*=2;
    if(p->feature==POSITIVE&&ref<0)ref=0;
    if(p->feature==ADD_BIAS) {
        uint16_t raw=(uint16_t)(p->bias_words[n]>>16);
        __fp16 value;memcpy(&value,&raw,2);ref+=(float)value;
        if(p->variant&&ref<0)ref=0;
    }
    if(p->feature==LEGACY_CLIP) {
        if(ref>1.9990234375)ref=1.9990234375;
        if(ref< -1.9990234375)ref=-1.9990234375;
    }
    if(p->feature==MODERN_BIAS_COMPAT) {
        unsigned lo=p->bias_words[n],hi=p->bias_words[32+n];
        ref+=half_bits((uint16_t)(hi>>16));
        switch((hi>>8)&7) {
            case 1:if(ref>0)ref=0;break;
            case 2:if(ref<0)ref=0;break;
            case 3:ref=fabs(ref);break;
            case 4:ref=-ref;break;
            case 5:ref=ref<0?-ref:0;break;
            case 6:ref=ref>0?-ref:0;break;
            case 7:ref=-fabs(ref);break;
        }
        ref=ref*half_bits((uint16_t)lo)+half_bits((uint16_t)(lo>>16));
    }
    return half_round(ref);
}
int main(int argc,char **argv) {
    setbuf(stdout,NULL);setbuf(stderr,NULL);alarm(35);
    if(argc!=12){fprintf(stderr,"Expected feature tiles start stop pattern act store offset mask variant bias-preset\n");return 2;}
    uint32_t v[11];for(int i=0;i<11;++i)v[i]=strtoul(argv[i+1],NULL,0);
    struct probe_report cfg={0};cfg.magic=PROBE_MAGIC;
    cfg.feature=v[0];cfg.tiles=v[1];cfg.start=v[2];cfg.stop=v[3];cfg.pattern=v[4];
    cfg.act_control=v[5];cfg.store_control=v[6];cfg.offset=v[7];cfg.spatial_mask=v[8];cfg.variant=v[9];
    for(unsigned i=0;i<256;++i) {
        if(v[10]==1)cfg.bias_words[i]=0x10203000+i;
        if(v[10]==2)cfg.bias_words[i]=i<32?0x3c00:0;
        if(v[10]==3)cfg.bias_words[i]=i<32?0x4000:0;
        if(v[10]==4)cfg.bias_words[i]=i<32?0x3c000000:0;
        if(v[10]==5)cfg.bias_words[i]=i<32?0x3c00+(i%4)*0x400:0;
        if(v[10]==6)cfg.bias_words[i]=i<32?0x2800:0;
        if(v[10]==7&&i<32) {
            __fp16 value=(__fp16)(((int)(i%7)-3)/4.0f);uint16_t raw;memcpy(&raw,&value,2);
            cfg.bias_words[i]=(uint32_t)raw<<16;
        }
        if(v[10]>=8&&v[10]<=12&&i<64) {
            if(i<32) {
                cfg.bias_words[i]=v[10]==10?0x4000:(v[10]==12?0xb4003800:(v[10]==11?0x3c003c00:0x3c00));
            } else {
                cfg.bias_words[i]=cfg.variant<<8;
                if(v[10]==9)cfg.bias_words[i]|=0x3c000000;
                if(v[10]==12)cfg.bias_words[i]|=0x38000000;
            }
        }
    }
    for(unsigned o=0;o<MAX_OUTPUTS;++o)for(unsigned i=0;i<1024;++i)cfg.output[o][i]=NAN;
    printf("CONFIG feature=%u tiles=%u start=%u stop=%u pattern=%u act=0x%x store=0x%x offset=%u mask=%u variant=%u bias=%u\n",
        cfg.feature,cfg.tiles,cfg.start,cfg.stop,cfg.pattern,cfg.act_control,cfg.store_control,cfg.offset,cfg.spatial_mask,cfg.variant,v[10]);
    int err;
    struct remote_rpc_control_unsigned_module ctrl={.domain=3,.enable=1};
    err=remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&ctrl,sizeof(ctrl));
    printf("unsigned_PD_status=0x%x\n",err);if(err)return 2;
    char uri[512];snprintf(uri,sizeof(uri),"%s&_dom=cdsp",hmx_probe_URI);
    remote_handle64 handle;
    err=hmx_probe_open(uri,&handle);printf("open_status=0x%x\n",err);if(err)return 2;
    struct probe_report *p=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,PROBE_BYTES);
    if(!p)return 2;
    int fd=rpcmem_to_fd(p);
    err=fastrpc_mmap(3,fd,p,0,PROBE_BYTES,FASTRPC_MAP_FD);
    printf("mapping_status=0x%x\n",err);if(err)return 2;
    memcpy(p,&cfg,sizeof(cfg));__sync_synchronize();
    err=hmx_probe_run(handle,fd,PROBE_BYTES,0);__sync_synchronize();
    printf("RPC status=0x%x stage=%d result=%d lock=%d total_vtcm=%u outputs=%u guard_bad=%u\n",
        err,p->stage,p->result,p->lock_status,p->total_vtcm,p->outputs,p->guard_bad);
    printf("CLEANUP unlock=%d release=%d power_down=%d\n",p->unlock_status,p->release_status,p->power_down_status);
    int failures=err||p->stage!=130||p->result||p->unlock_status||p->release_status||p->power_down_status||p->guard_bad;
    unsigned expected_outputs=1;
    if(cfg.feature==BIAS_ROUNDTRIP||cfg.feature==BIAS2_ROUNDTRIP||cfg.feature==BIAS_BANKS)expected_outputs=0;
    if(cfg.feature==WEIGHT_DEEP||cfg.feature==RETAIN||cfg.feature==CONVERT_CLEAR||
       cfg.feature==REPEAT_RETAIN||cfg.feature==EXPLICIT_SWAP)expected_outputs=2;
    if(cfg.feature==CLEAR_BOTH||cfg.feature==BEFORE_SEQUENCE)expected_outputs=4;
    failures+=p->outputs!=expected_outputs||p->outputs>MAX_OUTPUTS;
    // Copy back to cached memory before the numerical reference loop.
    struct probe_report *result=malloc(sizeof(*result));memcpy(result,p,sizeof(*result));
    if(!failures&&p->outputs)for(unsigned o=0;o<result->outputs;++o) {
        unsigned bad=0,nonfinite=0;float max_error=0,minimum=INFINITY,maximum=-INFINITY;
        for(unsigned r=0;r<32;++r)for(unsigned n=0;n<32;++n) {
            float actual=result->output[o][r*32+n];
            float ref=reference(result,o,r,n),e=fabsf(actual-ref);
            if(!isfinite(actual)){++nonfinite;}
            if(actual<minimum)minimum=actual;if(actual>maximum)maximum=actual;
            if(e>max_error)max_error=e;
            if((!isfinite(actual)&&!(isinf(actual)&&actual==ref))||e>0.00001f) {
                if(bad<4)printf("MISMATCH output=%u row=%u col=%u actual=%.9g expected=%.9g\n",o,r,n,actual,ref);
                ++bad;
            }
        }
        printf("NUMERIC output=%u checked=1024 bad=%u nonfinite=%u max_abs_error=%.9g min=%.9g max=%.9g samples=",o,bad,nonfinite,max_error,minimum,maximum);
        for(unsigned i=0;i<16;++i)printf("%.9g,",result->output[o][i]);printf("\n");
        printf("RAW output=%u sentinel_equal=%u data=",o,result->untouched[o]);
        for(unsigned i=0;i<16;++i)printf("%04x,",result->raw[o][i]);printf("\n");
        failures+=bad!=0;
    }
    if(result->feature==BIAS_ROUNDTRIP||result->feature==BIAS2_ROUNDTRIP||result->feature==BIAS_BANKS) {
        unsigned same=0,touched=0;for(unsigned i=0;i<256;++i){same+=result->bias_readback[i]==result->bias_words[i];touched+=result->bias_readback[i]!=0xa5a5a5a5;}
        printf("BIAS_ROUNDTRIP same_words=%u touched_words=%u data=",same,touched);
        for(unsigned i=0;i<256;++i)printf("%08x,",result->bias_readback[i]);printf("\n");
        if(result->variant==0) {
            unsigned expected=result->feature==BIAS_ROUNDTRIP?32:(result->feature==BIAS_BANKS?256:64);
            failures+=same!=expected||touched!=expected;
            for(unsigned i=0;i<256;++i)failures+=result->bias_readback[i]!=
                (i<expected?result->bias_words[i]:0xa5a5a5a5);
        } else if(result->feature==BIAS_ROUNDTRIP) {
            for(unsigned i=0;i<256;++i)failures+=result->bias_readback[i]!=
                (i<32?result->bias_words[i]:(i<64?0:0xa5a5a5a5));
        } else {
            for(unsigned i=0;i<256;++i)failures+=result->bias_readback[i]!=
                (i<32?result->bias_words[i]:0xa5a5a5a5);
        }
    }
    fastrpc_munmap(3,fd,p,PROBE_BYTES);rpcmem_free(p);free(result);hmx_probe_close(handle);
    printf("FEATURE_TEST %s failures=%d\n",failures?"FAIL":"PASS",failures);return failures?1:0;
}
