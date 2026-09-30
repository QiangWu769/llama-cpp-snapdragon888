#include "ggml-htp-op-support.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static void check(bool value, const char *message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static ggml_tensor tensor(ggml_type type, int64_t a, int64_t b, int64_t c=1, int64_t d=1) {
    ggml_tensor t{}; t.type=type; t.ne[0]=a; t.ne[1]=b; t.ne[2]=c; t.ne[3]=d;
    t.nb[0]=ggml_type_size(type); t.nb[1]=t.nb[0]*(a/ggml_blck_size(type));
    t.nb[2]=t.nb[1]*b; t.nb[3]=t.nb[2]*c;
    return t;
}
int main() {
    unsetenv("HTP_SPLIT_CPU_OPS"); unsetenv("HTP_DISABLE_FLASH_ATTN");
    check(!ggml_htp_split_cpu_ops_enabled(), "default off");
    auto w=tensor(GGML_TYPE_F16,896,128); std::strcpy(w.name,"blk.0.attn_q.weight");
    auto x=tensor(GGML_TYPE_F32,896,512); auto y=tensor(GGML_TYPE_F32,128,512);
    y.op=GGML_OP_MUL_MAT; y.src[0]=&w; y.src[1]=&x;
    auto *cpu=ggml_backend_reg_dev_get(ggml_backend_cpu_reg(),0);
    check(ggml_htp_op_layout_supported(&y), "loader probe with no buffers");
    check(ggml_backend_dev_supports_op(cpu,&y), "default CPU support unchanged");
    setenv("HTP_SPLIT_CPU_OPS","1",1);
    check(ggml_htp_split_cpu_ops_enabled(), "exact opt-in");
    check(!ggml_backend_dev_supports_op(cpu,&y), "CPU refuses packed rows");
    for (auto type : {GGML_TYPE_F16,GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL}) {
        w=tensor(type,896,128); std::strcpy(w.name,"MyHTP#blk.2.ffn_down.weight#0");
        check(ggml_htp_op_layout_supported(&y), "supported packed type");
    }
    w=tensor(GGML_TYPE_F16,896,128); std::strcpy(w.name,"blk.0.attn_q.weight");
    x.ne[1]=y.ne[1]=1; x.nb[2]=x.nb[1]; x.nb[3]=x.nb[2]; y.nb[2]=y.nb[1]; y.nb[3]=y.nb[2];
    check(ggml_htp_op_layout_supported(&y), "M1 decode");
    const auto wgood=w, xgood=x, ygood=y;
    w.ne[1]=127; y.ne[0]=127; check(!ggml_htp_op_layout_supported(&y), "N tail rejected");
    check(!ggml_backend_dev_supports_op(cpu,&y), "invalid packed shape cannot fall back");
    w=wgood; y=ygood; w.nb[1]+=2; check(!ggml_htp_op_layout_supported(&y), "strided packed weight rejected");
    w=wgood; x.ne[0]=864; check(!ggml_htp_op_layout_supported(&y), "K mismatch rejected");
    x=xgood; w.ne[2]=2; check(!ggml_htp_op_layout_supported(&y), "batched weight rejected");
    w=wgood; w.type=GGML_TYPE_F32; check(!ggml_htp_op_layout_supported(&y), "unhandled type rejected");
    w=wgood;
    for (const char *name : {"token_embd.weight","output.weight","CPU#output.weight#0","blk.0.attn_norm.weight"}) {
        std::strcpy(w.name,name); check(!ggml_htp_op_layout_supported(&y), "ordinary matrix stays CPU");
        check(ggml_backend_dev_supports_op(cpu,&y), "ordinary CPU matrix supported");
    }
#if defined(__has_include)
#if __has_include("ggml-htp-output-pack.h")
    {
        auto cached=tensor(GGML_TYPE_F16,896,75968);
        auto activation=tensor(GGML_TYPE_F32,896,1), projection=tensor(GGML_TYPE_F32,75968,1);
        projection.op=GGML_OP_MUL_MAT; projection.src[0]=&cached; projection.src[1]=&activation;
        for (const char *name : {"htp.output.packed.0.weight","htp.output.packed.1.weight",
                                "MyHTP#htp.output.packed.0.weight#0","CPU#htp.output.packed.1.weight#1"}) {
            std::strcpy(cached.name,name);
            check(ggml_htp_op_layout_supported(&projection), "output cache layout supported without buffers");
            check(!ggml_backend_dev_supports_op(cpu,&projection), "output cache CPU fallback refused");
        }
        std::strcpy(cached.name,"htp.output.packed.2.weight");
        check(!ggml_htp_op_layout_supported(&projection), "invalid output cache marker rejected");
    }
#endif
#endif
    y.op=GGML_OP_MUL; check(!ggml_htp_op_layout_supported(&y), "elementwise not advertised");
    check(ggml_backend_dev_supports_op(cpu,&y), "elementwise CPU supported");

    auto q=tensor(GGML_TYPE_F32,64,2,8), k=tensor(GGML_TYPE_F16,64,64,2), v=k;
    q.nb[1]=64*8*sizeof(float); q.nb[2]=64*sizeof(float);
    k.nb[1]=64*2*sizeof(ggml_fp16_t); k.nb[2]=64*sizeof(ggml_fp16_t); v=k;
    auto mask=tensor(GGML_TYPE_F16,64,2), out=tensor(GGML_TYPE_F32,64,8,2);
    out.op=GGML_OP_FLASH_ATTN_EXT; out.src[0]=&q; out.src[1]=&k; out.src[2]=&v; out.src[3]=&mask;
    float scale=0.125f; std::memcpy(&out.op_params[0],&scale,sizeof(scale));
    check(ggml_htp_op_layout_supported(&out), "exact flash layout without buffers");
    for (int64_t head_dim : {512,576,768}) {
        auto large_q=tensor(GGML_TYPE_F32,head_dim,2,8);
        auto large_k=tensor(GGML_TYPE_F16,head_dim,64,2);
        large_q.nb[1]=head_dim*8*sizeof(float); large_q.nb[2]=head_dim*sizeof(float);
        large_k.nb[1]=head_dim*2*sizeof(ggml_fp16_t); large_k.nb[2]=head_dim*sizeof(ggml_fp16_t);
        auto large_v=large_k,large_out=tensor(GGML_TYPE_F32,head_dim,8,2);
        large_out.op=GGML_OP_FLASH_ATTN_EXT;large_out.src[0]=&large_q;large_out.src[1]=&large_k;
        large_out.src[2]=&large_v;large_out.src[3]=&mask;
        const float large_scale=1.0f/std::sqrt((float)head_dim);
        std::memcpy(&large_out.op_params[0],&large_scale,sizeof(large_scale));
        check(ggml_htp_op_layout_supported(&large_out)==(head_dim==512), "v68 flash head limit512");
        check(ggml_backend_dev_supports_op(cpu,&large_out), "unsupported v68 heads retain CPU attention support");
    }
    auto maskgood=mask;
    mask.nb[1]+=2; check(!ggml_htp_op_layout_supported(&out), "invalid mask stride rejected"); mask=maskgood;
    scale=0.1f; std::memcpy(&out.op_params[0],&scale,sizeof(scale));
    check(!ggml_htp_op_layout_supported(&out), "unsupported flash scale rejected");
    scale=0.125f; std::memcpy(&out.op_params[0],&scale,sizeof(scale));
    out.src[3]=nullptr; check(!ggml_htp_op_layout_supported(&out), "missing mask rejected"); out.src[3]=&mask;
    setenv("HTP_DISABLE_FLASH_ATTN","1",1); check(!ggml_htp_op_layout_supported(&out), "flash disable honored");
    setenv("HTP_SPLIT_CPU_OPS","true",1); check(!ggml_htp_split_cpu_ops_enabled(), "only 1 opts in");
    check(!ggml_htp_op_layout_supported(nullptr), "null probe rejected");
    std::puts("PASS: unallocated loader probes, supported layouts, CPU fallback protection, default compatibility");
}
