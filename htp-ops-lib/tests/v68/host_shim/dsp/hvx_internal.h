#ifndef HTP_V68_HOST_HVX_SHIM_H
#define HTP_V68_HOST_HVX_SHIM_H

/* Test-only scalar semantics for the subset used by mat_mul_v68.c.
 * This does NOT emulate Hexagon qfloat precision, instructions, or timing.
 * Never put this include directory on a production DSP/Android build path. */
#include <stdint.h>
#include <string.h>
#define VLEN 128
typedef struct { float lane[32]; } HVX_Vector;
#define vmem(address) (*(HVX_Vector *) (address))
#define vmemu(address) (*(HVX_Vector *) (address))

static inline HVX_Vector Q6_V_vzero(void) {
    HVX_Vector result = {{0}};
    return result;
}
static inline HVX_Vector Q6_V_vsplat_R(int32_t bits) {
    HVX_Vector result;
    float value;
    memcpy(&value, &bits, sizeof(value));
    for (int i = 0; i < 32; ++i) result.lane[i] = value;
    return result;
}
static inline HVX_Vector Q6_Vqf32_vmpy_VsfVsf(HVX_Vector a, HVX_Vector b) {
    for (int i = 0; i < 32; ++i) a.lane[i] *= b.lane[i];
    return a;
}
static inline HVX_Vector Q6_Vqf32_vadd_Vqf32Vqf32(HVX_Vector a, HVX_Vector b) {
    for (int i = 0; i < 32; ++i) a.lane[i] += b.lane[i];
    return a;
}
static inline HVX_Vector Q6_Vsf_equals_Vqf32(HVX_Vector value) {
    return value;
}

#endif
