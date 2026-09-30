#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Test-only scalar semantics for the integer HVX instructions in the actual
 * production helper. In particular, widening multiply splits even/odd lanes,
 * unlike vunpack. This is a lane/arithmetic check, not an ISA simulator. */
typedef union {
  uint8_t b[128];
  uint16_t h[64];
  uint32_t w[32];
} HVX_Vector;
typedef struct { HVX_Vector v[2]; } HVX_VectorPair;
typedef struct { uint8_t b[128]; } HVX_VectorPred;

#define Q6_V_lo_W(x) ((x).v[0])
#define Q6_V_hi_W(x) ((x).v[1])

static inline HVX_Vector Q6_V_vzero(void) {
  HVX_Vector r = {0};
  return r;
}

static inline HVX_Vector Q6_V_vsplat_R(uint32_t x) {
  HVX_Vector r;
  for (int i = 0; i < 32; ++i) r.w[i] = x;
  return r;
}

static inline HVX_Vector Q6_Vh_vsplat_R(uint16_t x) {
  HVX_Vector r;
  for (int i = 0; i < 64; ++i) r.h[i] = x;
  return r;
}

#define BINARY_OP(name, member, count, operation)                           \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {              \
    HVX_Vector r;                                                         \
    for (int i = 0; i < count; ++i) r.member[i] = (operation);             \
    return r;                                                             \
  }

BINARY_OP(Q6_V_vand_VV, w, 32, a.w[i] & b.w[i])
BINARY_OP(Q6_V_vor_VV, w, 32, a.w[i] | b.w[i])
BINARY_OP(Q6_V_vxor_VV, w, 32, a.w[i] ^ b.w[i])
BINARY_OP(Q6_Vw_vadd_VwVw, w, 32, a.w[i] + b.w[i])
BINARY_OP(Q6_Vw_vsub_VwVw, w, 32, a.w[i] - b.w[i])
BINARY_OP(Q6_Vh_vsub_VhVh, h, 64, (uint16_t) (a.h[i] - b.h[i]))
BINARY_OP(Q6_Vuh_vmax_VuhVuh, h, 64, a.h[i] > b.h[i] ? a.h[i] : b.h[i])
BINARY_OP(Q6_Vw_vasl_VwVw, w, 32, b.w[i] >= 32 ? 0 : a.w[i] << b.w[i])
/* Every nonzero quant uses a shift in [3,10]. For zero, the production helper
 * explicitly selects zero, independently of a larger shift's ISA result. */
BINARY_OP(Q6_Vh_vlsr_VhVh, h, 64, b.h[i] >= 16 ? 0 : a.h[i] >> b.h[i])

static inline HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a, int shift) {
  for (int i = 0; i < 32; ++i) a.w[i] <<= shift;
  return a;
}

static inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, int shift) {
  for (int i = 0; i < 32; ++i) a.w[i] >>= shift;
  return a;
}

static inline HVX_Vector Q6_Vuh_vlsr_VuhR(HVX_Vector a, int shift) {
  for (int i = 0; i < 64; ++i) a.h[i] >>= shift;
  return a;
}

static inline HVX_Vector Q6_Vuw_vcl0_Vuw(HVX_Vector a) {
  for (int i = 0; i < 32; ++i) a.w[i] = a.w[i] ? __builtin_clz(a.w[i]) : 32;
  return a;
}

static inline HVX_VectorPred Q6_Q_vcmp_eq_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < 128; ++i) q.b[i] = a.w[i / 4] == b.w[i / 4];
  return q;
}

static inline HVX_VectorPred Q6_Q_vcmp_eq_VhVh(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < 128; ++i) q.b[i] = a.h[i / 2] == b.h[i / 2];
  return q;
}

static inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q, HVX_Vector a, HVX_Vector b) {
  for (int i = 0; i < 128; ++i) a.b[i] = q.b[i] ? a.b[i] : b.b[i];
  return a;
}

static inline HVX_VectorPair Q6_Wuw_vmpy_VuhVuh(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPair r;
  for (int i = 0; i < 32; ++i) {
    r.v[0].w[i] = (uint32_t) a.h[2 * i] * b.h[2 * i];
    r.v[1].w[i] = (uint32_t) a.h[2 * i + 1] * b.h[2 * i + 1];
  }
  return r;
}

/* Independent native IEEE conversion; it intentionally does not reuse the
 * production bit arithmetic in hvx_convert.h. Test that header separately. */
static inline HVX_Vector hvx_my_wsf_to_vhf(HVX_Vector high, HVX_Vector low) {
  HVX_Vector out;
  for (int i = 0; i < 32; ++i) {
    float even, odd;
    memcpy(&even, &low.w[i], sizeof(even));
    memcpy(&odd, &high.w[i], sizeof(odd));
    __fp16 even_half = (__fp16) even, odd_half = (__fp16) odd;
    memcpy(&out.h[2 * i], &even_half, sizeof(even_half));
    memcpy(&out.h[2 * i + 1], &odd_half, sizeof(odd_half));
  }
  return out;
}

/* The runner extracts these functions from src/dsp/ops/mat_mul.c. */
#include "integer_dequant_under_test.h"

static int is_half_nan(uint16_t bits) {
  return (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
}

int main(void) {
  uint64_t count = 0;
  for (int phase = 0; phase < 4; ++phase) {
    for (unsigned base = 0; base < 65536; ++base) {
      HVX_Vector values, scales;
      uint16_t reference[64];
      for (int lane = 0; lane < 64; ++lane) {
        const int quant = phase * 64 + lane - 128;
        __fp16 half_quant = (__fp16) quant, half_scale;
        memcpy(&values.h[lane], &half_quant, sizeof(half_quant));
        /* Every quant sees every scale, with different neighboring scales to
         * catch accidental contiguous-vs-even/odd lane interpretation. */
        scales.h[lane] = (uint16_t) (base + lane * 997u);
        memcpy(&half_scale, &scales.h[lane], sizeof(half_scale));
        __fp16 expected = (__fp16) ((float) quant * (float) half_scale);
        memcpy(&reference[lane], &expected, sizeof(expected));
      }
      const HVX_Vector actual = matmul_dequant_multiply_fp16_integer(values, scales);
      for (int lane = 0; lane < 64; ++lane) {
        const int reference_nan = is_half_nan(reference[lane]);
        if ((reference_nan && !is_half_nan(actual.h[lane])) ||
            (!reference_nan && actual.h[lane] != reference[lane])) {
          fprintf(stderr, "FAIL q=%d scale=0x%04x got=0x%04x expected=0x%04x lane=%d\n",
                  phase * 64 + lane - 128, scales.h[lane], actual.h[lane], reference[lane], lane);
          return 1;
        }
        ++count;
      }
    }
  }
  printf("PASS %llu products: every INT8 value x every FP16 scale bit pattern.\n",
         (unsigned long long) count);
  puts("Finite/zero/Inf bits and NaN classification agree with independent IEEE arithmetic.");
  puts("Source-level lane semantics only; real Hexagon compilation and DSP execution remain necessary.");
  return 0;
}
