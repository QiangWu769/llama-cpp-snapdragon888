#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_power.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <string.h>
#include "hmx_probe.h"
#include "probe.h"

static int power_client;
static void mark(struct probe_report *p, int stage) {
    p->stage=stage;
    __asm__ volatile("barrier" ::: "memory");
    qurt_mem_cache_clean((qurt_addr_t)p,sizeof(*p),QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
    FARF(ALWAYS,"HMX_PROBE stage=%d result=%d",stage,p->result);
}
AEEResult hmx_probe_open(const char *uri, remote_handle64 *handle) {
    (void)uri; *handle=1;return AEE_SUCCESS;
}
AEEResult hmx_probe_close(remote_handle64 handle) {
    (void)handle;return AEE_SUCCESS;
}
static void multiply(unsigned a,unsigned ac,unsigned w,unsigned wc,int kind) {
    if(kind==BASIC||kind==PARTIAL||kind==WEIGHT_NEGATE)
        __asm__ volatile("{ activation.hf = mxmem(%0, %1)\nweight.hf = mxmem(%2, %3) }"
                         ::"r"(a),"r"(ac),"r"(w),"r"(wc):"memory");
    else if(kind==WEIGHT_DEEP)
        __asm__ volatile("{ activation.hf = mxmem(%0, %1)\nweight.hf = mxmem(%2, %3):deep }"
                         ::"r"(a),"r"(ac),"r"(w),"r"(wc):"memory");
    else if(kind==SINGLE)
        __asm__ volatile("{ activation.hf = mxmem(%0, %1):single\nweight.hf = mxmem(%2, %3) }"
                         ::"r"(a),"r"(ac),"r"(w),"r"(wc):"memory");
    else
        __asm__ volatile("{ activation.hf = mxmem(%0, %1):deep\nweight.hf = mxmem(%2, %3) }"
                         ::"r"(a),"r"(ac),"r"(w),"r"(wc):"memory");
}
static void store(__fp16 *out,unsigned control,int retain,int pos,int before) {
    if(before)
        __asm__ volatile("mxmem(%0, %1):before.hf = acc"::"r"(out),"r"(control):"memory");
    else if(pos)
        __asm__ volatile("mxmem(%0, %1):after:pos.hf = acc"::"r"(out),"r"(control):"memory");
    else if(retain)
        __asm__ volatile("mxmem(%0, %1):after:retain.hf = acc"::"r"(out),"r"(control):"memory");
    else
        __asm__ volatile("mxmem(%0, %1):after.hf = acc"::"r"(out),"r"(control):"memory");
}
AEEResult hmx_probe_run(remote_handle64 handle,int32 fd,int32 bytes,int32 mode) {
    (void)handle;(void)mode;
    if(bytes<PROBE_BYTES)return AEE_EBADPARM;
    struct probe_report *p=NULL;
    int err=HAP_mmap_get(fd,(void**)&p,NULL);
    if(err)return err;
    qurt_mem_cache_clean((qurt_addr_t)p,sizeof(*p),QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
    if(p->magic!=PROBE_MAGIC||p->tiles<1||p->tiles>32||p->start>31||p->stop>31||p->feature>BIAS_BANKS||
       p->start%8||(p->stop+1)%8||(p->tiles-1)*32+p->stop<p->start||
       (p->feature==WEIGHT_DEEP&&p->tiles!=1)||
       (p->feature==SINGLE&&(p->tiles!=1||p->spatial_mask!=28||p->offset>28||p->offset%4||p->variant>2))){
        HAP_mmap_put(fd);return AEE_EBADPARM;
    }
    int powered=0,locked=0;
    unsigned int resource=0;
    HAP_power_request_t power;
    compute_res_attr_t attr;
    compute_res_vtcm_page_t total_pages,available_pages;
    p->result=0;
    p->has_lock1=compute_resource_hmx_lock!=NULL;
    p->has_lock2=compute_resource_hmx_lock2!=NULL;
    mark(p,1);
    memset(&power,0,sizeof(power));power.type=HAP_power_set_HMX;power.hmx.power_up=TRUE;
    err=p->power_status=HAP_power_set(&power_client,&power);if(err)goto end;
    powered=1;mark(p,11);
    unsigned int total_vtcm=0,available_vtcm=0;
    err=p->query_status=HAP_compute_res_query_VTCM(0,&total_vtcm,&total_pages,&available_vtcm,&available_pages);
    p->total_vtcm=total_vtcm;p->available_vtcm=available_vtcm;if(err)goto end;
    err=p->attr_status=HAP_compute_res_attr_init(&attr);if(err)goto end;
    err=p->hmx_attr_status=HAP_compute_res_attr_set_hmx_param(&attr,1);if(err)goto end;
    err=p->vtcm_attr_status=HAP_compute_res_attr_set_vtcm_param(&attr,262144,1);if(err)goto end;
    mark(p,30);resource=p->resource_id=HAP_compute_res_acquire(&attr,100000);
    if(!resource){err=AEE_EFAILED;goto end;}
    unsigned char *vtcm=HAP_compute_res_attr_get_vtcm_ptr(&attr);
    p->vtcm_address=(uint32_t)(uintptr_t)vtcm;
    if(!vtcm||((uintptr_t)vtcm&2047)){err=AEE_EFAILED;goto end;}
    err=p->lock_status=HAP_compute_res_hmx_lock(resource);if(err)goto end;
    locked=1;mark(p,50);
    memset(vtcm,0,262144);
    __fp16 *a=(__fp16*)vtcm;
    __fp16 *w=(__fp16*)(vtcm+65536);
    __fp16 *c=(__fp16*)(vtcm+196608);
    uint32_t *bias=(uint32_t*)(vtcm+214016);
    uint32_t *bias_out=(uint32_t*)(vtcm+216064);
    unsigned nt=p->feature==SINGLE?2:p->tiles;
    if(p->feature==SINGLE&&p->variant==2) {
        // Poison the gap so an accidental contiguous read cannot pass.
        for(unsigned i=0;i<3072;++i)((uint16_t*)a)[i]=0x7e00;
    }
    for(unsigned t=0;t<nt;++t)for(unsigned r=0;r<32;++r)for(unsigned k=0;k<32;++k)
        a[(p->feature==SINGLE&&p->variant==2?t*2048:t*1024)+packed_index(r,k)]=(__fp16)activation_value(t,r,k,p->pattern);
    unsigned first=p->start,last=(p->tiles-1)*32+p->stop;
    unsigned channels=last-first+1;
    unsigned groups=p->feature==WEIGHT_DEEP?2:1;
    // Partial K uses a compact weight stream; activation croutons remain full.
    for(unsigned g=0;g<groups;++g)for(unsigned q=0;q<channels;++q)for(unsigned n=0;n<32;++n) {
        unsigned k=first+q;
        unsigned index=g*channels*32+packed_index(q,n);
        w[index]=(__fp16)weight_value(k/32,k%32,n,g,p->pattern);
    }
    unsigned sentinel=output_sentinel(p);
    for(unsigned i=0;i<MAX_OUTPUTS*1024;++i)((uint16_t*)c)[i]=sentinel;
    for(unsigned i=0;i<256;++i)bias[i]=p->bias_words[i];
    memset(bias_out,0xa5,1024);
    qurt_mem_cache_clean((qurt_addr_t)vtcm,262144,QURT_MEM_CACHE_FLUSH_INVALIDATE,QURT_MEM_DCACHE);
    __asm__ volatile("barrier":::"memory");mark(p,60);
    __asm__ volatile("mxclracc.hf":::"memory");mark(p,72);
    if(p->feature==BIAS_BANKS) {
        for(unsigned bank=0;bank<4;++bank) {
            unsigned address=(unsigned)(uintptr_t)(bias+64*bank)|bank;
            __asm__ volatile("bias = mxmem2(%0)"::"r"(address):"memory");
        }
        for(unsigned bank=0;bank<4;++bank) {
            unsigned address=(unsigned)(uintptr_t)(bias_out+64*bank)|bank;
            __asm__ volatile("mxmem2(%0) = bias"::"r"(address):"memory");
        }
        p->outputs=0;goto readback;
    }
    if(p->feature==BIAS2_ROUNDTRIP) {
        __asm__ volatile("bias = mxmem2(%0)"::"r"(bias):"memory");
        if(p->variant)__asm__ volatile("mxmem(%0) = bias"::"r"(bias_out):"memory");
        else __asm__ volatile("mxmem2(%0) = bias"::"r"(bias_out):"memory");
        p->outputs=0;goto readback;
    }
    if(p->feature==MODERN_BIAS_COMPAT)
        __asm__ volatile("bias = mxmem2(%0)"::"r"(bias):"memory");
    else __asm__ volatile("bias = mxmem(%0)"::"r"(bias):"memory");mark(p,82);
    if(p->feature==BIAS_ROUNDTRIP) {
        if(p->variant)__asm__ volatile("mxmem2(%0) = bias"::"r"(bias_out):"memory");
        else __asm__ volatile("mxmem(%0) = bias"::"r"(bias_out):"memory");
        p->outputs=0;goto readback;
    }
    unsigned ar=(unsigned)(uintptr_t)a|(p->start<<2);
    unsigned ac=p->act_control;
    unsigned wr=(unsigned)(uintptr_t)w|(p->feature==WEIGHT_NEGATE?32:0);
    unsigned wc=groups*channels*64-1;
    if(p->feature==SINGLE) {
        ar|=spatial_bits(p->offset);
        if(p->variant==1){ar+=2048;ac=(unsigned)(-2048)|spatial_bits(p->spatial_mask)|(p->stop<<2);}
        if(p->variant==2)ac=4096|spatial_bits(p->spatial_mask)|(p->stop<<2);
    }
    mark(p,91);multiply(ar,ac,wr,wc,p->feature);mark(p,92);
    p->outputs=1;
    if(p->feature==ACCUMULATE) {
        multiply(ar,ac,wr,wc,BASELINE);store(c,p->store_control,0,0,0);
    } else if(p->feature==RETAIN||p->feature==CONVERT_CLEAR) {
        store(c,p->store_control,p->feature==RETAIN,0,0);
        multiply(ar,ac,wr,wc,BASELINE);
        store(c+1024,p->store_control,0,0,0);p->outputs=2;
    } else if(p->feature==EXPLICIT_CLEAR) {
        __asm__ volatile("mxclracc.hf":::"memory");
        store(c,p->store_control,0,0,0);
    } else if(p->feature==REPEAT_RETAIN) {
        store(c,p->store_control,1,0,0);store(c+1024,p->store_control,1,0,0);p->outputs=2;
    } else if(p->feature==WEIGHT_DEEP) {
        store(c,p->store_control,0,0,0);
        if(p->variant)__asm__ volatile("mxswapacc.hf":::"memory");
        store(c+1024,p->store_control,0,0,0);p->outputs=2;
    } else if(p->feature==EXPLICIT_SWAP) {
        __asm__ volatile("mxswapacc.hf":::"memory");
        store(c,p->store_control,1,0,0);
        __asm__ volatile("mxswapacc.hf":::"memory");
        store(c+1024,p->store_control,1,0,0);p->outputs=2;
    } else if(p->feature==CLEAR_BOTH) {
        // Seed both banks with distinguishable nonzero matrix products.
        store(c,p->store_control,1,0,0);
        __asm__ volatile("mxswapacc.hf":::"memory");
        multiply(ar,ac,wr,wc,BASELINE);
        multiply(ar,ac,wr,wc,BASELINE);
        store(c+1024,p->store_control,1,0,0);
        __asm__ volatile("mxclracc.hf":::"memory");
        store(c+2048,p->store_control,1,0,0);
        __asm__ volatile("mxswapacc.hf":::"memory");
        store(c+3072,p->store_control,1,0,0);p->outputs=4;
    } else if(p->feature==BEFORE_SEQUENCE) {
        __asm__ volatile("mxmem(%0,%1):before:retain.hf = acc"::"r"(c),"r"(p->store_control):"memory");
        store(c+1024,p->store_control,1,0,0);
        __asm__ volatile("mxmem(%0,%1):before:retain.hf = acc"::"r"(c+2048),"r"(p->store_control):"memory");
        store(c+3072,p->store_control,1,0,0);p->outputs=4;
    } else store(c,p->store_control,0,p->feature==POSITIVE||(p->feature==ADD_BIAS&&p->variant),p->feature==BEFORE);
readback:
    __asm__ volatile("barrier":::"memory");
    qurt_mem_cache_clean((qurt_addr_t)(vtcm+196608),22528,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
    mark(p,112);
    for(unsigned i=0;i<256;++i)p->bias_readback[i]=bias_out[i];
    for(unsigned o=0;o<p->outputs;++o) {
        p->untouched[o]=0;
        for(unsigned r=0;r<32;++r)for(unsigned n=0;n<32;++n) {
            unsigned i=o*1024+packed_index(r,n);
            p->output[o][r*32+n]=(float)c[i];
            p->untouched[o]+=((uint16_t*)c)[i]==sentinel;
            if(r==0&&n<16)p->raw[o][n]=((uint16_t*)c)[i];
        }
    }
    p->guard_bad=0;
    for(unsigned i=p->outputs*1024;i<MAX_OUTPUTS*1024;++i)
        p->guard_bad+=((uint16_t*)c)[i]!=sentinel;
    for(unsigned i=212992;i<214016;++i)p->guard_bad+=vtcm[i]!=0;
    mark(p,120);
end:
    p->result=err;
    if(locked)p->unlock_status=HAP_compute_res_hmx_unlock(resource);
    if(resource)p->release_status=HAP_compute_res_release(resource);
    if(powered){memset(&power,0,sizeof(power));power.type=HAP_power_set_HMX;power.hmx.power_up=FALSE;
        p->power_down_status=HAP_power_set(&power_client,&power);}
    mark(p,err?p->stage:130);HAP_mmap_put(fd);return err;
}
