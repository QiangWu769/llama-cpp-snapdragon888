#include "ggml-htp-output-pack.h"
#include <stdio.h>
#include <stdlib.h>

#define GUARD 64
static size_t checked_words;
static void check(bool ok, const char *message) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
/* All 65536 half encodings occur without any float conversion. Row/column
 * salting also distinguishes the two vocabulary halves and tile boundaries. */
static uint16_t bits(size_t row, size_t col, size_t k, bool every_encoding) {
    size_t index=row*k+col;
    return every_encoding ? (uint16_t)index : (uint16_t)((index*40503u)^((row>>5)*319u)^((col>>5)*127u));
}
static void run_shape(size_t k,size_t n,size_t row_offset,bool every_encoding) {
    size_t words=k*n;
    uint16_t *input=malloc((words+2*GUARD)*sizeof(*input));
    uint16_t *output=malloc((words+2*GUARD)*sizeof(*output));
    check(input&&output,"allocation");
    for(size_t i=0;i<words+2*GUARD;++i){input[i]=0xcafe;output[i]=0xbabe;}
    for(size_t row=0;row<n;++row)
        for(size_t col=0;col<k;++col) input[GUARD+row*k+col]=bits(row+row_offset,col,k,every_encoding);
    check(ggml_htp_pack_output_f16(output+GUARD,input+GUARD,k,n),"valid pack accepted");
    /* Independent inverse: decompose the packed linear address, then compare
     * its original row-major coordinate. No inverse calls the production loop. */
    for(size_t i=0;i<words;++i) {
        size_t coordinate=i;
        size_t half=coordinate%2; coordinate/=2;
        size_t lane=coordinate%32; coordinate/=32;
        size_t pair=coordinate%16; coordinate/=16;
        size_t ktile=coordinate%(k/32); coordinate/=k/32;
        size_t ntile=coordinate;
        size_t row=ntile*32+lane, col=ktile*32+pair*2+half;
        check(output[GUARD+i]==bits(row+row_offset,col,k,every_encoding),"inverse preserves exact half bits and vocabulary order");
    }
    for(size_t row=0;row<n;++row)
        for(size_t col=0;col<k;++col)
            check(input[GUARD+row*k+col]==bits(row+row_offset,col,k,every_encoding),"source unchanged");
    for(size_t i=0;i<GUARD;++i) {
        check(input[i]==0xcafe&&input[words+GUARD+i]==0xcafe,"source guards");
        check(output[i]==0xbabe&&output[words+GUARD+i]==0xbabe,"destination guards");
    }
    checked_words+=words;
    free(output);free(input);
}
static void negative_shapes(void) {
    uint16_t in[64]={0},out[64];
    for(size_t i=0;i<64;++i)out[i]=0xbeef;
    check(!ggml_htp_pack_output_f16(NULL,in,32,32),"null output");
    check(!ggml_htp_pack_output_f16(out,NULL,32,32),"null source");
    const size_t bad[][2]={{0,32},{32,0},{31,32},{32,31},{33,64},{64,33},
        {SIZE_MAX-(SIZE_MAX%32),32},{32,SIZE_MAX-(SIZE_MAX%32)}};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);++i)
        check(!ggml_htp_pack_output_f16(out,in,bad[i][0],bad[i][1]),"invalid shape rejected before write");
    for(size_t i=0;i<64;++i)check(out[i]==0xbeef,"rejections do not write output");
}
static void markers(void) {
    const struct {const char *name;int part;} good[]={
        {"htp.output.packed.0.weight",0},{"htp.output.packed.1.weight",1},
        {"MyHTP#htp.output.packed.0.weight#0",0},{"CPU#htp.output.packed.1.weight#123",1},
        {"MyHTP#htp.output.packed.0.weight",0},{"CPU#htp.output.packed.1.weight",1}};
    for(size_t i=0;i<sizeof(good)/sizeof(good[0]);++i)
        check(ggml_htp_output_weight_part(good[i].name)==good[i].part,"exact packed marker recognition");
    const char *bad[]={NULL,"","token_embd.weight","output.weight","CPU#output.weight#0", "blk.0.attn_q.weight",
        "htp.output.packed.2.weight","htp.output.packed.0.weight.extra","htp.output.packed.0.weight#0",
        "#htp.output.packed.0.weight#0","GPU#htp.output.packed.0.weight#0",
        "MyHTP2#htp.output.packed.0.weight#0","MyHTP#htp.output.packed.0.weight#",
        "CPU#htp.output.packed.0.weight#x","CPU#htp.output.packed.0.weight#0#1",
        "CPU#htp.output.packed.0.weight#-1","CPU#htp.output.packed.01.weight#0",
        "MyHTP#CPU#htp.output.packed.0.weight#0","xhtp.output.packed.0.weight"};
    for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);++i)
        check(ggml_htp_output_weight_part(bad[i])==-1,"ordinary or malformed marker rejected");
}
int main(int argc,char **argv) {
    (void)argv;
    markers();negative_shapes();
    run_shape(32,32,0,false); run_shape(64,96,0,false);run_shape(96,160,0,false);
    run_shape(1024,64,0,true);run_shape(896,96,0,false);
    /* Unequal split verifies the vocabulary concatenation offset separately. */
    run_shape(96,32,0,false);run_shape(96,64,32,false);
    if(argc>1){run_shape(896,75968,0,false);run_shape(896,75968,75968,false);}
    printf("PASS: %zu half words, bitwise inverse, input/guard integrity, split vocab order, invalid dimensions and markers\n",checked_words);
}
