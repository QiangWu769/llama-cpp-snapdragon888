// Real FastRPC tests: results must be produced by the DSP, never a host fallback.
#include <math.h>
#include <inttypes.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "host/session.h"
#include "htp_ops.h"
#include "message.h"
#include "op_reg.h"
#include "../../tests/v68/matmul_reference.h"

#define ARENA_SIZE (16 * 1024 * 1024)
#define OUTPUT_GUARD 128
static unsigned char *arena;
static int arena_fd;
static size_t cursor;
static int hmx_reference;
static int hmx_k_chunk;
static int vtcm_layout_suite;
static int tests_run;
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int alloc_offset(size_t bytes) {
    int off = (int)((cursor + 127) & ~(size_t)127);
    cursor = off + bytes;
    if (cursor > ARENA_SIZE) abort();
    return off;
}
static struct RpcmemBufAddr addr(int offset) {
    struct RpcmemBufAddr a = {arena_fd, offset};
    return a;
}
static int run_op_with_expected_status(unsigned int op, const void *params, size_t size, int expected_status) {
    memset(arena, 0, 4096);
    struct MessageHeader *msg = (void *)arena;
    msg->n_reqs = 1;
    msg->req_offsets[0] = sizeof(*msg) + 2 * sizeof(int32_t);
    msg->req_offsets[1] = msg->req_offsets[0] + sizeof(struct RequestHeader) + sizeof(struct OpComputeRequest) + size;
    struct RequestHeader *req = message_header_get_request_ptr(msg, 0);
    req->state = -999;
    req->type = REQUEST_TYPE_OP_COMPUTE;
    struct OpComputeRequest *compute = (void *)req->data;
    compute->op = op;
    memcpy(compute->payload, params, size);
    __sync_synchronize();
    msg->state.v[0] = 1;
    double start = now_seconds();
    while (!msg->state.v[1]) {
        if (now_seconds() - start > 20.0) {
            fprintf(stderr, "FAIL: DSP op %u timed out\n", op);
            /* Never reuse/unmap a buffer while the DSP may still own it. */
            _Exit(3);
        }
        usleep(100);
    }
    __sync_synchronize();
    if (req->state != expected_status)
        fprintf(stderr, "FAIL: DSP op %u returned %d, expected %d\n", op, req->state, expected_status);
    return req->state;
}
static int run_op(unsigned int op, const void *params, size_t size) {
    return run_op_with_expected_status(op,params,size,0);
}
/* Canonical low-byte-first hashing of float bits enables exact A/B comparison
 * without writing a large fixture dump. This is a test digest, not a signature. */
static uint64_t output_digest(const float *values, size_t count) {
    uint64_t digest=UINT64_C(14695981039346656037);
    for(size_t i=0;i<count;++i) {
        uint32_t bits;
        memcpy(&bits,values+i,sizeof(bits));
        for(int byte=0;byte<4;++byte) {
            digest^=(bits>>(8*byte))&255u;
            digest*=UINT64_C(1099511628211);
        }
    }
    return digest;
}
static int check_tolerance(const char *name, const float *actual, const float *ref,
                           int count, float atol, float rtol) {
    float max_err = 0, max_ratio = 0;
    double sum_error2 = 0;
    int bad = 0;
    for (int i = 0; i < count; ++i) {
        float err = fabsf(actual[i] - ref[i]);
        const float bound = atol + rtol * fabsf(ref[i]);
        if (!isfinite(actual[i]) || !isfinite(ref[i]) || err > bound) {
            if (bad++ < 3) fprintf(stderr, "%s mismatch i=%d actual=%g expected=%g bound=%g\n", name, i, actual[i], ref[i], bound);
        }
        if (err > max_err) max_err = err;
        if (err / bound > max_ratio) max_ratio = err / bound;
        sum_error2 += (double) err * err;
    }
    printf("%s %s elements=%d bad=%d max_abs_error=%.8g rmse=%.8g max_error_over_bound=%.5g atol=%g rtol=%g\n",
           bad ? "FAIL" : "PASS", name, count, bad, max_err,
           sqrt(sum_error2 / count), max_ratio, atol, rtol);
    return bad != 0;
}
static int check(const char *name, const float *actual, const float *ref, int count) {
    return check_tolerance(name, actual, ref, count, 2e-4f, 2e-4f);
}
static int test_matmul_typed(enum ggml_type type, int M, int K, int N) {
    cursor=4096;
    ++tests_run;
    unsigned int op;
    switch (type) {
        case GGML_TYPE_F16: op=HTP_OPS_MAT_MUL_PERMUTED_W16A32; break;
        case GGML_TYPE_Q4_0: op=HTP_OPS_MAT_MUL_PERMUTED_W4D16A32; break;
        case GGML_TYPE_Q8_0: op=HTP_OPS_MAT_MUL_PERMUTED_W8D16A32; break;
        case GGML_TYPE_IQ4_NL: op=HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL; break;
        default: return 1;
    }
    const size_t abytes=(size_t)M*K*sizeof(float);
    const size_t wbytes=v68_test_weight_bytes(type,K,N);
    const size_t obytes=(size_t)M*N*sizeof(float);
    int ao = alloc_offset(abytes);
    int wo = alloc_offset(wbytes);
    int guard = alloc_offset(obytes+2*OUTPUT_GUARD);
    int oo = guard+OUTPUT_GUARD;
    float *a=(void *)(arena+ao), *out=(void *)(arena+oo);
    void *w=arena+wo;
    float *ref=malloc(obytes), *host_a=malloc(abytes);
    void *host_w=malloc(wbytes);
    /* Report ideal-vs-half drift for decode cases without doubling every long
     * CPU reference. Build references in cached malloc memory, not RPC uncached
     * memory, which otherwise makes the host dot-product loop extremely slow. */
    float *full=(hmx_reference && M==1) ? malloc(obytes) : NULL;
    double reference_start=now_seconds();
    if (!ref || !host_a || !host_w || (hmx_reference && M==1 && !full) ||
        v68_test_make_matmul_ex(type,M,K,N,(uint32_t)(M+17*K+N),host_a,host_w,ref,
                                hmx_reference,hmx_k_chunk,full)) {
        fprintf(stderr,"FAIL: could not generate %s matrix test\n",v68_test_type_name(type));
        free(ref); free(host_a); free(host_w); free(full);
        return 1;
    }
    double reference_ms=(now_seconds()-reference_start)*1e3;
    memcpy(a,host_a,abytes); memcpy(w,host_w,wbytes);
    free(host_a); free(host_w);
    memset(arena+guard,0xa5,OUTPUT_GUARD);
    memset(arena+oo+obytes,0xa5,OUTPUT_GUARD);
    for (int i=0;i<M*N;++i) out[i]=NAN;
    struct MatMulParams p={addr(oo),addr(ao),addr(wo),M,K,N};
    double rpc_start=now_seconds();
    int fail=run_op(op,&p,sizeof(p));
    double rpc_ms=(now_seconds()-rpc_start)*1e3;
    char name[80]; snprintf(name,sizeof(name),"%s-MATMUL M=%d K=%d N=%d",v68_test_type_name(type),M,K,N);
    /* HMX uses rounded half inputs and a half output. Against that reference,
     * 0.001 relative is approximately one FP16 ULP; 0.01 absolute allows small
     * accumulator differences under cancellation over up to 4864 terms. This
     * is a fixed acceptance envelope, not measured performance or a proof of
     * the undocumented accumulator precision. Report all errors for review.
     * HVX retains its existing FP32 reference and strict tolerance. */
    if (!fail) fail=hmx_reference ? check_tolerance(name,out,ref,M*N,0.01f,0.001f) :
                                   check(name,out,ref,M*N);
    for(int i=0;i<OUTPUT_GUARD;++i) {
        if(arena[guard+i]!=0xa5 || arena[oo+obytes+i]!=0xa5) {
            fprintf(stderr,"FAIL: %s output guard changed at %d\n",name,i);
            fail=1; break;
        }
    }
    float conversion_max=0;
    if(full) for(int i=0;i<M*N;++i) {
        float d=fabsf(ref[i]-full[i]);
        if(d>conversion_max) conversion_max=d;
    }
    printf("TIMING %s reference_ms=%.3f rpc_wall_ms=%.3f reference_k_round_chunk=%d",
           name,reference_ms,rpc_ms,hmx_k_chunk);
    if(full) printf(" expected_fp16_conversion_max_abs=%g",conversion_max);
    putchar('\n');
    if(vtcm_layout_suite)
        printf("FIXTURE %s seed=%" PRIu32 " output_fnv1a64=%016" PRIx64 "\n",
               name,(uint32_t)(M+17*K+N),output_digest(out,(size_t)M*N));
    free(ref); free(full); return fail != 0;
}
static int test_matmul(int M, int K, int N) {
    return test_matmul_typed(GGML_TYPE_F16,M,K,N);
}
static int test_matmul_invalid(int kind) {
    cursor=4096;
    ++tests_run;
    int ao=alloc_offset(32*sizeof(float));
    int wo=alloc_offset(32*32*sizeof(__fp16));
    int guard=alloc_offset(32*sizeof(float)+2*OUTPUT_GUARD);
    int oo=guard+OUTPUT_GUARD;
    memset(arena+ao,0,32*sizeof(float));
    memset(arena+wo,0,32*32*sizeof(__fp16));
    memset(arena+guard,0xa5,32*sizeof(float)+2*OUTPUT_GUARD);
    struct MatMulParams p={addr(oo),addr(ao),addr(wo),1,32,32};
    static const char *names[]={"M-zero","K-not-tile-aligned","N-not-tile-aligned","activation-not-HVX-aligned"};
    if(kind==0) p.m=0;
    else if(kind==1) p.k=31;
    else if(kind==2) p.n=31;
    else p.activation.offset+=sizeof(float);
    int status=run_op_with_expected_status(HTP_OPS_MAT_MUL_PERMUTED_W16A32,&p,sizeof(p),-1);
    int failed=status!=-1;
    for(size_t i=0;i<32*sizeof(float)+2*OUTPUT_GUARD;++i) {
        if(arena[guard+i]!=0xa5) {failed=1;break;}
    }
    printf("%s F16-INVALID case=%s status=%d expected=-1 output_unchanged=%d\n",
           failed?"FAIL":"PASS",names[kind],status,!failed);
    return failed;
}
static int test_attention_case(int Q, int KV, int H, int KH, int D,
                               int mask_kind, int half_path) {
    cursor=4096;
    ++tests_run;
    int MS=(KV+63)&~63;
    int qo=alloc_offset(Q*H*D*4), ko=alloc_offset(MS*KH*D*2), vo=alloc_offset(MS*KH*D*2);
    int mo=alloc_offset(Q*MS*2), guard=alloc_offset(Q*H*D*4+2*OUTPUT_GUARD);
    int oo=guard+OUTPUT_GUARD;
    float *q=(void *)(arena+qo), *out=(void *)(arena+oo);
    __fp16 *k=(void *)(arena+ko),*v=(void *)(arena+vo),*mask=(void *)(arena+mo);
    float *ref=calloc((size_t)Q*H*D,4),*scores=malloc(KV*4);
    if(!ref || !scores) {free(ref);free(scores);return 1;}
    memset(arena+guard,0xa5,OUTPUT_GUARD);
    memset(arena+oo+Q*H*D*4,0xa5,OUTPUT_GUARD);
    for(int i=0;i<Q*H*D;++i) {q[i]=sinf(i*0.07f);out[i]=NAN;}
    for(int i=0;i<KV*KH*D;++i) {k[i]=(__fp16)cosf(i*0.03f);v[i]=(__fp16)sinf(i*0.11f);}
    /* Deliberately poison physical tail storage: padding must never participate
     * in QK softmax or PV, including the unmasked/null-mask path. */
    for(int i=KV*KH*D;i<MS*KH*D;++i) {k[i]=(__fp16)NAN;v[i]=(__fp16)NAN;}
    for(int t=0;t<Q;++t) for(int s=0;s<MS;++s) {
        float bias=half_path ? -0.17f*(s%7)-0.031f*(t%5) : -0.01f*(s%3);
        if(s>=KV) bias=NAN;
        else if(s>KV-Q+t || (mask_kind==2 && t==0)) bias=-INFINITY;
        mask[t*MS+s]=(__fp16)bias;
    }
    for(int t=0;t<Q;++t) for(int h=0;h<H;++h) {
        float maxv=-INFINITY,sum=0;
        int kh=h/(H/KH);
        for(int s=0;s<KV;++s) {
            double dot=0;
            for(int d=0;d<D;++d) dot+=(double)q[(t*H+h)*D+d]*(float)k[(s*KH+kh)*D+d];
            scores[s]=(float)dot/sqrtf((float)D)+(mask_kind==0?0:(float)mask[t*MS+s]);
            if(scores[s]>maxv) maxv=scores[s];
        }
        if(maxv==-INFINITY) continue; /* Required zero output for all-masked row. */
        for(int s=0;s<KV;++s) {scores[s]=expf(scores[s]-maxv);sum+=scores[s];}
        for(int d=0;d<D;++d) {
            double val=0;
            for(int s=0;s<KV;++s) val+=scores[s]/sum*(float)v[(s*KH+kh)*D+d];
            ref[(t*H+h)*D+d]=(float)val;
        }
    }
    struct FlashAttnParams p={addr(oo),addr(qo),addr(ko),addr(vo),addr(mo),Q,KV,H,KH,D};
    if(mask_kind==0) {p.mask.fd=-1;p.mask.offset=0;}
    double rpc_start=now_seconds();
    int fail=run_op(HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,&p,sizeof(p));
    double rpc_ms=(now_seconds()-rpc_start)*1e3;
    char name[112]; snprintf(name,sizeof(name),"ATTENTION Q=%d KV=%d H=%d KH=%d D=%d mask=%s",
                            Q,KV,H,KH,D,mask_kind==0?"none":mask_kind==1?"causal+finite":"all-masked-row");
    /* This compares the whole HMX attention algorithm with full precision
     * softmax, so QK/P16/V16/output conversion drift is included. Inputs Q/K/V
     * stay in [-1,1]; the 0.003 absolute limit is not an arbitrary-input bound. */
    if(!fail) fail=half_path ? check_tolerance(name,out,ref,Q*H*D,0.003f,0) :
                              check(name,out,ref,Q*H*D);
    if(mask_kind==2) for(int i=0;i<H*D;++i) if(out[i]!=0.0f) {
        fprintf(stderr,"FAIL: all-masked attention output is not zero at %d\n",i);fail=1;break;
    }
    for(int i=0;i<OUTPUT_GUARD;++i) if(arena[guard+i]!=0xa5 || arena[oo+Q*H*D*4+i]!=0xa5) {
        fprintf(stderr,"FAIL: %s output guard changed\n",name);fail=1;break;
    }
    printf("TIMING %s rpc_wall_ms=%.3f\n",name,rpc_ms);
    free(scores);free(ref);return fail != 0;
}
static int test_attention(int Q, int KV, int H, int KH, int D) {
    return test_attention_case(Q,KV,H,KH,D,1,0);
}
int main(int argc, char **argv) {
    int full=0, pipeline=0, attention=0;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--hmx")) hmx_reference=1;
        else if(!strcmp(argv[i],"--full")) full=1;
        else if(!strcmp(argv[i],"--pipeline")) {pipeline=1;hmx_reference=1;}
        else if(!strcmp(argv[i],"--vtcm-layout")) {vtcm_layout_suite=1;hmx_reference=1;}
        else if(!strcmp(argv[i],"--hmx-attention")) attention=1;
        else {
            fprintf(stderr,"usage: %s [--hmx [--full] | --pipeline | --hmx-attention | --vtcm-layout]\n"
                    "default: original HVX small matrices and attention, FP32 reference\n"
                    "--hmx: F16/Q8/IQ4, half-boundary reference, small and model-size matrices\n"
                    "--full: all M={1,5,32,33} x K={896,4864}, N=896\n"
                    "--pipeline: Q8/IQ4 four-stage and K-chunked output-stationary tests\n"
                    "--vtcm-layout: F16 Qwen shapes at 32-row boundaries, partial N, invalid inputs\n"
                    "--hmx-attention: GQA, query/KV tails, null/causal/all-masked masks\n",argv[0]);
            return !strcmp(argv[i],"--help") ? 0 : 2;
        }
    }
    if(full && !hmx_reference) {fprintf(stderr,"--full requires --hmx\n");return 2;}
    if((attention && (hmx_reference || full)) || (pipeline && full) ||
       (vtcm_layout_suite && (attention || pipeline || full))) {
        fprintf(stderr,"select a single test suite\n");return 2;
    }
    setbuf(stdout,NULL); alarm(vtcm_layout_suite?600:hmx_reference?300:90);
    printf("V68_RPC_TEST reference=%s suite=%s (verify DSP dispatch logs separately)\n",
           hmx_reference?"half-input/half-output":attention?"full-softmax, HMX tolerance":"FP32",
           vtcm_layout_suite?"vtcm-layout":attention?"attention":pipeline?"pipeline":full?"full":"default");
    if(open_dsp_session(CDSP_DOMAIN_ID,1)) return 2;
    remote_handle64 h=get_global_handle();
    int err=htp_ops_init_backend(h);
    if(err) {fprintf(stderr,"init_backend failed 0x%x\n",err);return 2;}
    arena=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_FLAG_UNCACHED,ARENA_SIZE);
    if(!arena) return 2;
    arena_fd=rpcmem_to_fd(arena);
    if(arena_fd<0 || fastrpc_mmap(CDSP_DOMAIN_ID,arena_fd,arena,0,ARENA_SIZE,FASTRPC_MAP_FD)) return 2;
    if(htp_ops_create_channel(h,arena_fd,4096)) return 2;
    cursor=4096;
    int failures=0;
    if(vtcm_layout_suite) {
        const int ms[]={1,31,32,33,64,65,128,129};
        const int shapes[][2]={{896,896},{896,4864},{4864,896}};
        for(size_t mi=0;mi<sizeof(ms)/sizeof(ms[0]);++mi)
            for(size_t si=0;si<sizeof(shapes)/sizeof(shapes[0]);++si)
                failures+=test_matmul(ms[mi],shapes[si][0],shapes[si][1]);
        /* At the validated usable size these force partial final N chunks:
         * K896/N5216 tails both layouts; K4864/N928 tails the optimized one. */
        const int tail_ms[]={1,33};
        for(size_t mi=0;mi<sizeof(tail_ms)/sizeof(tail_ms[0]);++mi) {
            failures+=test_matmul(tail_ms[mi],896,5216);
            failures+=test_matmul(tail_ms[mi],4864,928);
        }
        for(int kind=0;kind<4;++kind) failures+=test_matmul_invalid(kind);
    } else if(attention) {
        const int shapes[][5]={{1,65,14,2,64},{5,67,14,2,64},{33,33,2,2,64}};
        for(int s=0;s<3;++s) for(int mask=0;mask<3;++mask)
            failures+=test_attention_case(shapes[s][0],shapes[s][1],shapes[s][2],
                                          shapes[s][3],shapes[s][4],mask,1);
    } else if(pipeline) {
        const enum ggml_type types[]={GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL};
        for(int t=0;t<2;++t) {
            hmx_k_chunk=0;
            failures+=test_matmul_typed(types[t],128,896,896);
            hmx_k_chunk=512;
            failures+=test_matmul_typed(types[t],129,1568,1056);
        }
    } else if(hmx_reference) {
        const enum ggml_type types[]={GGML_TYPE_F16,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL};
        const int ms[]={1,5,32,33}, ks[]={896,4864};
        for(int t=0;t<3;++t) {
            failures+=test_matmul_typed(types[t],5,96,96);
            for(int mi=0;mi<4;++mi) for(int ki=0;ki<2;++ki) {
                if(full || ki==(mi%2))
                    failures+=test_matmul_typed(types[t],ms[mi],ks[ki],896);
            }
        }
    } else {
        failures+=test_matmul(1,32,32);
        failures+=test_matmul(3,64,96);
        failures+=test_matmul(5,96,96);
        failures+=test_matmul(8,256,64);
        const enum ggml_type quant_types[]={GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL};
        for (int i=0;i<3;++i) {
            failures+=test_matmul_typed(quant_types[i],1,64,64);
            failures+=test_matmul_typed(quant_types[i],5,96,96);
            failures+=test_matmul_typed(quant_types[i],8,256,64);
        }
        failures+=test_attention(1,7,4,2,32);
        failures+=test_attention(3,67,4,2,64);
    }
    htp_ops_destroy_channel(h);
    close_dsp_session();
    fastrpc_munmap(CDSP_DOMAIN_ID,arena_fd,arena,ARENA_SIZE);
    rpcmem_free(arena);
    printf("V68_RPC_TEST %s failures=%d cases=%d\n",failures?"FAIL":"PASS",failures,tests_run);
    return failures?1:0;
}
