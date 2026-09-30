#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include "calibration.h"

/* Four independent vector accumulators. The volatile register constraints
 * prohibit replacing repeated additions by an algebraic multiply. Loads and
 * stores occur only at chunk boundaries, never inside the arithmetic loop. */
__attribute__((noinline)) void calibration_hvx_chunk(uint32_t *values, unsigned iterations) {
    HVX_Vector a=*(HVX_Vector *)(values+0), b=*(HVX_Vector *)(values+32);
    HVX_Vector c=*(HVX_Vector *)(values+64), d=*(HVX_Vector *)(values+96);
    HVX_Vector x=Q6_V_vsplat_R(calibration_increment(0)), y=Q6_V_vsplat_R(calibration_increment(1));
    HVX_Vector z=Q6_V_vsplat_R(calibration_increment(2)), w=Q6_V_vsplat_R(calibration_increment(3));
    for(unsigned i=0;i<iterations;++i) {
        asm volatile("%0.w = vadd(%0.w, %4.w)\n"
                     "%1.w = vadd(%1.w, %5.w)\n"
                     "%2.w = vadd(%2.w, %6.w)\n"
                     "%3.w = vadd(%3.w, %7.w)\n"
                     : "+v"(a), "+v"(b), "+v"(c), "+v"(d)
                     : "v"(x), "v"(y), "v"(z), "v"(w));
    }
    *(HVX_Vector *)(values+0)=a; *(HVX_Vector *)(values+32)=b;
    *(HVX_Vector *)(values+64)=c; *(HVX_Vector *)(values+96)=d;
}
