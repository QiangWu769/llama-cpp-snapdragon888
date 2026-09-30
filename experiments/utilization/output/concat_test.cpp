#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>

static void check(bool ok,const char *message) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
static void run(int64_t first_n,int64_t second_n,int64_t m) {
    ggml_init_params init={8*1024*1024,nullptr,false};
    auto *ctx=ggml_init(init);check(ctx,"context");
    auto *a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,first_n,m);
    auto *b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,second_n,m);
    auto *ad=static_cast<float *>(a->data),*bd=static_cast<float *>(b->data);
    for(int64_t token=0;token<m;++token) {
        for(int64_t row=0;row<first_n;++row)ad[token*first_n+row]=(float)(token*100000+row);
        for(int64_t row=0;row<second_n;++row)bd[token*second_n+row]=(float)(token*100000+first_n+row);
    }
    auto *joined=ggml_concat(ctx,a,b,0);auto *graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,joined);
    check(ggml_graph_compute_with_ctx(ctx,graph,4)==GGML_STATUS_SUCCESS,"concat execution");
    check(joined->ne[0]==first_n+second_n&&joined->ne[1]==m,"[vocab,M] shape");
    auto *result=static_cast<const float *>(joined->data);
    for(int64_t token=0;token<m;++token)
        for(int64_t row=0;row<first_n+second_n;++row)
            check(result[token*(first_n+second_n)+row]==(float)(token*100000+row),"per-token vocabulary order");
    ggml_free(ctx);
}
int main(){run(32,64,1);run(64,32,32);run(256,256,32);
    std::puts("PASS: production GGML concat dim0 preserves each token's ordered vocabulary halves (M1/M32)");}
