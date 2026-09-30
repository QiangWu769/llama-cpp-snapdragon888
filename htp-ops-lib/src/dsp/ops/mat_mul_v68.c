/*
 * Snapdragon 888 / Hexagon v68 fallback for the HTP weight layout.
 *
 * This is a DSP HVX implementation, not an HMX implementation. The exported
 * hmx_* names are retained solely for compatibility with the existing RPC
 * dispatcher. No HMX instructions, VTCM allocation or DMA engine are required.
 *
 * The converter stores each 32 x 32 (output x input) weight tile as
 * [input-pair][output][input-within-pair]. Quantized input additionally uses
 * the 256-element superblock layout produced by repack_*_super_block_hvx().
 * Unpacking is scalar; the matrix multiply and accumulation operate on 32
 * output channels at a time with HVX FP32/qfloat32 instructions. Four input
 * rows share each unpacked weight pair. In contrast to the HMX path, neither
 * FP32 activations nor dequantized weights are rounded to FP16 before the
 * multiplication. Results therefore need a numerical, not bitwise, comparison.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dsp/hvx_internal.h"
#include "dsp/ops.h"

#define HVX_MATMUL_CHANNELS 32
#define HVX_MATMUL_ROWS 4

static const int8_t hvx_iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

/* All callers pass an offset aligned to 64 elements: one input pair for all
 * 32 output channels. A pair is wholly contained in one quant superblock. */
static inline void hvx_unpack_weight_pair(float *w0, float *w1,
                                          const void *weight, size_t offset,
                                          enum ggml_type type) {
    if (type == GGML_TYPE_F16) {
        const __fp16 *p = (const __fp16 *) weight + offset;
        for (int lane = 0; lane < HVX_MATMUL_CHANNELS; ++lane) {
            w0[lane] = (float) p[2 * lane];
            w1[lane] = (float) p[2 * lane + 1];
        }
        return;
    }

    const size_t block = offset / QK_K;
    const unsigned int first = (unsigned int) (offset % QK_K);
    if (type == GGML_TYPE_Q8_0) {
        const my_block_q8_0 *p = (const my_block_q8_0 *) weight + block;
        /* A quant scale covers 32 elements, or 16 output-channel pairs. */
        const float d0 = (float) p->scales[first / QK_0];
        const float d1 = (float) p->scales[first / QK_0 + 1];
        for (int lane = 0; lane < HVX_MATMUL_CHANNELS; ++lane) {
            const unsigned int local = first + 2 * lane;
            const float d = lane < 16 ? d0 : d1;
            w0[lane] = d * (float) p->quants[local];
            w1[lane] = d * (float) p->quants[local + 1];
        }
        return;
    }

    const my_block_q4_0 *p = (const my_block_q4_0 *) weight + block;
    const float d0 = (float) p->scales[first / QK_0];
    const float d1 = (float) p->scales[first / QK_0 + 1];
    const unsigned int shift = (first / 128) * 4;
    const unsigned int parity = (first / 64) & 1;
    for (int lane = 0; lane < HVX_MATMUL_CHANNELS; ++lane) {
        /* Repacked bytes hold elements (j,j+128), then (j+64,j+192).
         * Here first is a multiple of 64, so j is simply 2*lane or 2*lane+1. */
        const unsigned int q0 = (p->quants[4 * lane + parity] >> shift) & 15;
        const unsigned int q1 = (p->quants[4 * lane + 2 + parity] >> shift) & 15;
        const float d = lane < 16 ? d0 : d1;
        if (type == GGML_TYPE_IQ4_NL) {
            w0[lane] = d * (float) hvx_iq4_nl_values[q0];
            w1[lane] = d * (float) hvx_iq4_nl_values[q1];
        } else {
            w0[lane] = d * (float) ((int) q0 - 8);
            w1[lane] = d * (float) ((int) q1 - 8);
        }
    }
}

static inline HVX_Vector hvx_splat_float(float value) {
    int32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return Q6_V_vsplat_R(bits);
}

static int hvx_mat_mul_permuted_v68(float *restrict dst, const float *activation,
                                   const void *weight, int m, int k, int n,
                                   enum ggml_type type) {
    if (!dst || !activation || !weight || m <= 0 || k <= 0 || n <= 0 ||
        k % HVX_MATMUL_CHANNELS || n % HVX_MATMUL_CHANNELS) {
        return -1;
    }
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_Q4_0 &&
        type != GGML_TYPE_Q8_0 && type != GGML_TYPE_IQ4_NL) {
        return -1;
    }
    /* Avoid wrapping address calculations on the 32-bit DSP. This pointer-only
     * API cannot check buffer lengths; callers must supply the full matrices. */
    if ((size_t) m > SIZE_MAX / sizeof(float) / (size_t) k ||
        (size_t) m > SIZE_MAX / sizeof(float) / (size_t) n ||
        (size_t) n > SIZE_MAX / sizeof(__fp16) / (size_t) k) {
        return -1;
    }

    float unpacked0[HVX_MATMUL_CHANNELS] __attribute__((aligned(VLEN)));
    float unpacked1[HVX_MATMUL_CHANNELS] __attribute__((aligned(VLEN)));

    for (int row = 0; row < m; row += HVX_MATMUL_ROWS) {
        const int rows = m - row < HVX_MATMUL_ROWS ? m - row : HVX_MATMUL_ROWS;
        for (int out = 0; out < n; out += HVX_MATMUL_CHANNELS) {
            HVX_Vector sums[HVX_MATMUL_ROWS];
            for (int r = 0; r < rows; ++r) {
                sums[r] = Q6_V_vzero();
            }
            for (int in_tile = 0; in_tile < k; in_tile += HVX_MATMUL_CHANNELS) {
                const size_t tile_offset = (size_t) out * (size_t) k +
                                           (size_t) in_tile * HVX_MATMUL_CHANNELS;
                for (int pair = 0; pair < HVX_MATMUL_CHANNELS / 2; ++pair) {
                    hvx_unpack_weight_pair(unpacked0, unpacked1, weight,
                                           tile_offset + (size_t) pair * 64, type);
                    const HVX_Vector w0 = vmem(unpacked0);
                    const HVX_Vector w1 = vmem(unpacked1);
                    for (int r = 0; r < rows; ++r) {
                        const float *a = activation + (size_t) (row + r) * (size_t) k +
                                         in_tile + 2 * pair;
                        const HVX_Vector a0 = hvx_splat_float(a[0]);
                        const HVX_Vector a1 = hvx_splat_float(a[1]);
                        sums[r] = Q6_Vqf32_vadd_Vqf32Vqf32(
                            sums[r], Q6_Vqf32_vmpy_VsfVsf(a0, w0));
                        sums[r] = Q6_Vqf32_vadd_Vqf32Vqf32(
                            sums[r], Q6_Vqf32_vmpy_VsfVsf(a1, w1));
                    }
                }
            }
            for (int r = 0; r < rows; ++r) {
                /* The RPC normally supplies aligned buffers. Unaligned vector
                 * stores also support standalone tests and sliced destinations. */
                vmemu(dst + (size_t) (row + r) * (size_t) n + out) =
                    Q6_Vsf_equals_Vqf32(sums[r]);
            }
        }
    }
    return 0;
}

/* Compatibility entry points. The v68 build selects this file instead of the
 * HMX mat_mul.c implementation. */
int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *activation,
                               const __fp16 *permuted_weight, int m, int k, int n) {
    return hvx_mat_mul_permuted_v68(dst, activation, permuted_weight, m, k, n,
                                  GGML_TYPE_F16);
}

int hmx_mat_mul_permuted_qk_0_d16a32(float *restrict dst, const float *activation,
                                   const uint8_t *permuted_weight, int m, int k, int n,
                                   enum ggml_type weight_type) {
    if (weight_type != GGML_TYPE_Q4_0 && weight_type != GGML_TYPE_Q8_0 &&
        weight_type != GGML_TYPE_IQ4_NL) {
        return -1;
    }
    return hvx_mat_mul_permuted_v68(dst, activation, permuted_weight, m, k, n,
                                  weight_type);
}
