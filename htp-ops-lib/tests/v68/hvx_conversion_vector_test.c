/* Integer-HVX model for executing the actual conversion wrapper source on a
 * host. This checks lane order and bit arithmetic, not instruction execution. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define _HVX_INTERNAL_H
#define HTP_HMX_V68 1
#define HVX_INLINE_ALWAYS inline __attribute__((unused))
typedef struct __attribute__((may_alias)) { uint32_t lane[32]; } HVX_Vector;
typedef struct { HVX_Vector lo, hi; } HVX_VectorPair;
typedef struct { uint32_t lane[32]; } HVX_VectorPred;
#define vmem(p) (*(HVX_Vector *)(p))
static __attribute__((unused)) HVX_Vector Q6_V_vsplat_R(int32_t x) { HVX_Vector r; for(int i=0;i<32;++i) r.lane[i]=(uint32_t)x; return r; }
static __attribute__((unused)) HVX_Vector Q6_V_vzero(void) { return Q6_V_vsplat_R(0); }
#define MODEL_BINARY(name,expression) static __attribute__((unused)) HVX_Vector name(HVX_Vector a,HVX_Vector b) { HVX_Vector r; for(int i=0;i<32;++i) r.lane[i]=(expression); return r; }
MODEL_BINARY(Q6_V_vand_VV,a.lane[i]&b.lane[i])
MODEL_BINARY(Q6_V_vor_VV,a.lane[i]|b.lane[i])
MODEL_BINARY(Q6_Vw_vadd_VwVw,a.lane[i]+b.lane[i])
MODEL_BINARY(Q6_Vw_vsub_VwVw,a.lane[i]-b.lane[i])
MODEL_BINARY(Q6_Vw_vasl_VwVw,a.lane[i]<<(b.lane[i]&31))
MODEL_BINARY(Q6_Vw_vlsr_VwVw,a.lane[i]>>(b.lane[i]&31))
static __attribute__((unused)) HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a,int n) { return Q6_Vw_vasl_VwVw(a,Q6_V_vsplat_R(n)); }
static __attribute__((unused)) HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a,int n) { return Q6_Vw_vlsr_VwVw(a,Q6_V_vsplat_R(n)); }
static __attribute__((unused)) HVX_Vector Q6_Vuw_vcl0_Vuw(HVX_Vector a) { HVX_Vector r; for(int i=0;i<32;++i) r.lane[i]=a.lane[i]?(uint32_t)__builtin_clz(a.lane[i]):32; return r; }
static __attribute__((unused)) HVX_VectorPred Q6_Q_vcmp_eq_VwVw(HVX_Vector a,HVX_Vector b) { HVX_VectorPred q; for(int i=0;i<32;++i) q.lane[i]=a.lane[i]==b.lane[i]; return q; }
static __attribute__((unused)) HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q,HVX_Vector a,HVX_Vector b) { HVX_Vector r; for(int i=0;i<32;++i) r.lane[i]=q.lane[i]?a.lane[i]:b.lane[i]; return r; }
static __attribute__((unused)) HVX_VectorPair Q6_W_vcombine_VV(HVX_Vector hi,HVX_Vector lo) { HVX_VectorPair r={lo,hi}; return r; }
static __attribute__((unused)) HVX_Vector Q6_V_lo_W(HVX_VectorPair p) { return p.lo; }
static __attribute__((unused)) HVX_Vector Q6_V_hi_W(HVX_VectorPair p) { return p.hi; }
/* Legacy qfloat helpers are parsed but unused in this v68 wrapper test. */
extern HVX_Vector Q6_Vh_vsplat_R(int);
extern HVX_Vector Q6_Vh_vadd_VhVh(HVX_Vector,HVX_Vector);
extern HVX_Vector Q6_Vsf_equals_Vqf32(HVX_Vector);
extern HVX_Vector Q6_Vqf16_vadd_VhfVhf(HVX_Vector,HVX_Vector);
#ifndef TEST_HVX_HEADER
#define TEST_HVX_HEADER "../../include/dsp/hvx_convert.h"
#endif
#include TEST_HVX_HEADER

static uint32_t random_state=1;
static uint32_t next_bits(void) { random_state=random_state*1664525u+1013904223u; return random_state; }
static uint16_t reference_half(uint32_t bits) { float f; __fp16 h; uint16_t out; memcpy(&f,&bits,4); h=(__fp16)f; memcpy(&out,&h,2); return out; }
static uint32_t reference_float(uint16_t bits) { __fp16 h; float f; uint32_t out; memcpy(&h,&bits,2); f=(float)h; memcpy(&out,&f,4); return out; }
static void check_half(uint16_t got,uint16_t expected) { if((expected&0x7fffu)>0x7c00u) assert((got&0x7fffu)>0x7c00u); else assert(got==expected); }
static void check_float(uint32_t got,uint32_t expected) { if((expected&0x7fffffffu)>0x7f800000u) assert((got&0x7fffffffu)>0x7f800000u); else assert(got==expected); }
int main(void) {
  for(uint32_t base=0;base<65536;base+=64) {
    HVX_Vector packed;
    for(int i=0;i<32;++i) packed.lane[i]=(base+2u*i)|((base+2u*i+1u)<<16);
    const HVX_VectorPair expanded=hvx_my_vhf_to_wsf(packed);
    const HVX_Vector roundtrip=hvx_my_wsf_to_vhf(expanded.hi,expanded.lo);
    for(int i=0;i<32;++i) {
      check_float(expanded.lo.lane[i],reference_float((uint16_t)(base+2u*i)));
      check_float(expanded.hi.lane[i],reference_float((uint16_t)(base+2u*i+1u)));
      check_half((uint16_t)roundtrip.lane[i],(uint16_t)(base+2u*i));
      check_half((uint16_t)(roundtrip.lane[i]>>16),(uint16_t)(base+2u*i+1u));
    }
  }
  for(int block=0;block<32768;++block) {
    HVX_Vector lo,hi;
    for(int i=0;i<32;++i) { lo.lane[i]=next_bits(); hi.lane[i]=next_bits(); }
    const HVX_Vector packed=hvx_my_wsf_to_vhf(hi,lo);
    for(int i=0;i<32;++i) {
      check_half((uint16_t)packed.lane[i],reference_half(lo.lane[i]));
      check_half((uint16_t)(packed.lane[i]>>16),reference_half(hi.lane[i]));
    }
  }
  puts("PASS integer-HVX model: all binary16 patterns, lane order, 2097152 binary32 patterns");
}
