#include "ggml.h"
#include "ggml-cpu.h"
#include "htp-cpu-impl.h"
#include "htp-ops.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" bool htp_ops_support_op(const ggml_tensor *) { return false; }
extern "C" bool htp_ops_has_permuted_weight(const ggml_tensor *) { return false; }
extern "C" int htp_ops_compute_op(ggml_compute_params *,ggml_tensor *) { std::abort(); }
static constexpr size_t guard=256;
static unsigned cases=0;
static void check(bool good,const char *message) {
    if(!good){std::fprintf(stderr,"FAIL: %s (case %u)\n",message,cases);std::exit(1);}
}
struct Storage {
    ggml_tensor *t;
    std::vector<unsigned char> memory;
    explicit Storage(ggml_tensor *tensor,int layout):t(tensor) {
        t->nb[0]=(layout==3?8:4);
        t->nb[1]=t->nb[0]*t->ne[0]+(layout?20:0);
        t->nb[2]=t->nb[1]*t->ne[1]+(layout?28:0);
        t->nb[3]=t->nb[2]*t->ne[2]+(layout?36:0);
        if(layout==2) {
            t->nb[2]=t->nb[0]*t->ne[0]+12;
            t->nb[1]=t->nb[2]*t->ne[2]+20;
            t->nb[3]=t->nb[1]*t->ne[1]+28;
        }
        memory.resize(ggml_nbytes(t)+2*guard,0xa5);
        t->data=memory.data()+guard;
    }
    size_t offset(const std::array<int64_t,4> &c) const {
        size_t n=guard;for(int d=0;d<4;++d)n+=(size_t)c[d]*t->nb[d];return n;
    }
};
static uint32_t pattern(const std::array<int64_t,4>&c,int part,ggml_type type) {
    uint32_t raw=(uint32_t)(c[0]+37*c[1]+503*c[2]+8099*c[3]+100000*part);
    if(type==GGML_TYPE_I32) return (raw*2654435761u)^0xa39b704du;
    float value=(float)raw/8; uint32_t bits;std::memcpy(&bits,&value,4);return bits;
}
static void populate(Storage &s,int part) {
    for(int64_t i3=0;i3<s.t->ne[3];++i3)
    for(int64_t i2=0;i2<s.t->ne[2];++i2)
    for(int64_t i1=0;i1<s.t->ne[1];++i1)
    for(int64_t i0=0;i0<s.t->ne[0];++i0){
        std::array<int64_t,4> c={i0,i1,i2,i3};auto bits=pattern(c,part,s.t->type);
        std::memcpy(s.memory.data()+s.offset(c),&bits,4);
    }
}
static void execute(ggml_context *ctx,ggml_cgraph *graph,int threads,bool hybrid) {
    auto plan=ggml_graph_plan(graph,threads,nullptr);
    std::vector<uint8_t> scratch(plan.work_size);plan.work_data=scratch.data();
    auto status=hybrid?ggml_graph_compute_htp_hybrid(graph,&plan):ggml_graph_compute(graph,&plan);
    check(status==GGML_STATUS_SUCCESS,"production graph execution");(void)ctx;
}
static void run(ggml_type type,std::array<int64_t,4>a_shape,int64_t second,int dim,
                int a_layout,int b_layout,int out_layout,int threads,bool hybrid) {
    ++cases;
    ggml_init_params init={2*1024*1024,nullptr,true};auto *ctx=ggml_init(init);check(ctx,"context");
    auto b_shape=a_shape;b_shape[dim]=second;
    Storage a(ggml_new_tensor(ctx,type,4,a_shape.data()),a_layout);
    Storage b(ggml_new_tensor(ctx,type,4,b_shape.data()),b_layout);
    populate(a,0);populate(b,1);const auto a_before=a.memory,b_before=b.memory;
    Storage out(ggml_concat(ctx,a.t,b.t,dim),out_layout);
    auto expected=out.memory;
    for(int64_t i3=0;i3<out.t->ne[3];++i3)
    for(int64_t i2=0;i2<out.t->ne[2];++i2)
    for(int64_t i1=0;i1<out.t->ne[1];++i1)
    for(int64_t i0=0;i0<out.t->ne[0];++i0){
        std::array<int64_t,4> c={i0,i1,i2,i3},src=c;int part=0;
        if(src[dim]>=a_shape[dim]){src[dim]-=a_shape[dim];part=1;}
        auto bits=pattern(src,part,type);std::memcpy(expected.data()+out.offset(c),&bits,4);
    }
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out.t);
    for(const char *mode: {"0","1"}) {
        std::fill(out.memory.begin(),out.memory.end(),0xa5);setenv("HTP_FAST_CONCAT",mode,1);
        execute(ctx,graph,threads,hybrid);
        check(out.memory==expected,"ordered bitwise concatenation, padding and guards");
        check(a.memory==a_before&&b.memory==b_before,"source remains unchanged");
    }
    ggml_free(ctx);
}
static void alias_fallback(bool hybrid) {
    // A deliberately overlapping destination must retain generic behavior;
    // new memcpy calls may not introduce an overlap precondition for callers.
    std::vector<unsigned char> baseline;
    for(const char *mode:{"0","1"}){
        ++cases;ggml_init_params init={1024*1024,nullptr,true};auto *ctx=ggml_init(init);
        auto *a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,7),*b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,5);
        Storage out(ggml_concat(ctx,a,b,0),0);
        std::vector<float> a_values={1,2,3,4,5,6,7};a->data=a_values.data();
        b->data=(char *)out.t->data+sizeof(float); // overlaps the output's first half
        for(int i=0;i<5;++i){float v=(float)(20+i);std::memcpy((char *)b->data+4*i,&v,4);}
        setenv("HTP_FAST_CONCAT",mode,1);auto *g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out.t);
        execute(ctx,g,1,hybrid);
        if(mode[0]=='0')baseline=out.memory;else check(out.memory==baseline,"overlap preserves generic-path behavior");
        ggml_free(ctx);
    }
}
int main(){
    ggml_cpu_init();
    for(bool hybrid:{false,true}){
        for(int threads:{1,2,3,8})for(auto type:{GGML_TYPE_F32,GGML_TYPE_I32}) {
            run(type,{37,1,1,1},13,0,0,0,0,threads,hybrid); // M1
            run(type,{37,32,1,1},13,0,0,0,0,threads,hybrid); // M32: ne2=1
            run(type,{11,3,4,2},19,0,1,2,1,threads,hybrid); // padded/permuted source rows
            run(type,{11,3,4,2},19,0,2,1,2,threads,hybrid); // permuted destination rows
            run(type,{11,3,4,2},19,0,1,3,1,threads,hybrid); // scalar-strided second source: fallback
            run(type,{11,3,4,2},19,0,1,1,3,threads,hybrid); // scalar-strided output: fallback
            for(int dim:{1,2,3})run(type,{7,3,4,2},5,dim,1,2,1,threads,hybrid);
        }
        run(GGML_TYPE_F32,{75968,32,1,1},75968,0,0,0,0,8,hybrid); // actual head shape
        alias_fallback(hybrid);
    }
    std::printf("PASS: %u production concat cases; normal/hybrid, F32/I32, M1/M32, full head, strided/permuted rows, fallback dimensions and overlap, bitwise output/input/guard checks\n",cases);
}
