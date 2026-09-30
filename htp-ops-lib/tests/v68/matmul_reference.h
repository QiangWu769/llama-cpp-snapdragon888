#ifndef HTP_V68_MATMUL_REFERENCE_H
#define HTP_V68_MATMUL_REFERENCE_H

/* Test-only generator and reference. Packing follows the converter tensor axes
 * and the upstream GGML repacker, independently of the DSP random-access decoder.
 * The expected result is an ordinary [M,K] x [N,K]^T double-precision dot product.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "dsp/quants.h"

static inline uint32_t v68_test_random(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static inline const char *v68_test_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        default: return "INVALID";
    }
}

static inline size_t v68_test_weight_bytes(enum ggml_type type, int k, int n) {
    const size_t count = (size_t) k * (size_t) n;
    switch (type) {
        case GGML_TYPE_F16: return count * sizeof(__fp16);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL: return count / 256 * sizeof(my_block_q4_0);
        case GGML_TYPE_Q8_0: return count / 256 * sizeof(my_block_q8_0);
        default: return 0;
    }
}

/* hmx_fp16 models the explicit HMX conversion boundaries, not the DSP kernel:
 * A32 -> A16; (quant * scale16) -> W16; exact dot -> O16 -> O32.
 * The optional full_precision output keeps A32 and dequantized W32, making the
 * expected conversion drift visible separately from implementation error.
 * The old wrapper below retains the original HVX data and numerical contract.
 */
static inline int v68_test_make_matmul_ex(enum ggml_type type, int m, int k, int n,
                                        uint32_t seed, float *activation,
                                        void *packed, float *reference,
                                        int hmx_fp16, int hmx_k_chunk, float *full_precision) {
    static const int values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    if (m <= 0 || k <= 0 || n <= 0 || k % 32 || n % 32 ||
        !v68_test_weight_bytes(type, k, n) || !activation || !packed || !reference) {
        return -1;
    }
    const size_t count = (size_t) k * (size_t) n;
    float *logical = (float *) malloc(count * sizeof(float));
    float *permuted = (float *) malloc(count * sizeof(float));
    if (!logical || !permuted) {
        free(logical);
        free(permuted);
        return -1;
    }
    uint32_t random = seed ? seed : 1;

    if (type == GGML_TYPE_F16) {
        __fp16 *p = (__fp16 *) packed;
        for (size_t i = 0; i < count; ++i) {
            p[i] = (__fp16) (((int) (v68_test_random(&random) % 1025) - 512) /
                             (hmx_fp16 ? 257.0f : 256.0f));
            permuted[i] = (float) p[i];
        }
    } else {
        for (size_t block = 0; block < count / 256; ++block) {
            __fp16 scales[8];
            uint8_t codes[256];
            for (int j = 0; j < 8; ++j) {
                /* Include negative, positive and zero scales. */
                scales[j] = (__fp16) (((int) (v68_test_random(&random) % 17) - 8) /
                                      (hmx_fp16 ? 127.0f : 128.0f));
            }
            for (int j = 0; j < 256; ++j) {
                codes[j] = (uint8_t) v68_test_random(&random);
            }
            if (type == GGML_TYPE_Q8_0) {
                my_block_q8_0 *p = (my_block_q8_0 *) packed + block;
                for (int j = 0; j < 8; ++j) p->scales[j] = scales[j];
                for (int j = 0; j < 256; ++j) {
                    const int value = (int) codes[j] - 128;
                    p->quants[j] = (int8_t) value;
                    permuted[block * 256 + j] = (float) scales[j / 32] * value;
                }
            } else {
                my_block_q4_0 *p = (my_block_q4_0 *) packed + block;
                uint8_t unpacked[256];
                for (int j = 0; j < 8; ++j) {
                    p->scales[j] = scales[j];
                    /* Build standard GGML 32-element blocks, whose low/high
                     * nibbles hold the first/second half of the block. */
                    uint8_t standard[16];
                    for (int x = 0; x < 16; ++x) {
                        standard[x] = (codes[j * 32 + x] & 15) |
                                      ((codes[j * 32 + x + 16] & 15) << 4);
                    }
                    for (int x = 0; x < 16; ++x) {
                        unpacked[j * 32 + x] = standard[x] & 15;
                        unpacked[j * 32 + x + 16] = standard[x] >> 4;
                    }
                }
                /* Sequential repacking as in repack_q4_0_super_block_hvx,
                 * not the decoder's per-element byte/nibble formula. */
                for (int j = 0; j < 64; ++j) {
                    p->quants[2 * j] = (unpacked[j + 128] << 4) | unpacked[j];
                    p->quants[2 * j + 1] = (unpacked[j + 192] << 4) | unpacked[j + 64];
                }
                for (int j = 0; j < 256; ++j) {
                    const int value = type == GGML_TYPE_IQ4_NL ?
                        values[unpacked[j]] : (int) unpacked[j] - 8;
                    permuted[block * 256 + j] = (float) scales[j / 32] * value;
                }
            }
        }
    }

    /* Walk [N/32,K/32,16,32,2] sequentially to reconstruct ordinary [N,K]. */
    size_t index = 0;
    for (int nb = 0; nb < n / 32; ++nb)
        for (int kb = 0; kb < k / 32; ++kb)
            for (int pair = 0; pair < 16; ++pair)
                for (int lane = 0; lane < 32; ++lane)
                    for (int half = 0; half < 2; ++half)
                        logical[(size_t) (nb * 32 + lane) * k + kb * 32 + pair * 2 + half] =
                            permuted[index++];

    for (size_t i = 0; i < (size_t) m * k; ++i) {
        activation[i] = ((int) (v68_test_random(&random) % 1025) - 512) /
                        (hmx_fp16 ? 251.0f : 256.0f);
    }
    /* Cache the rounded operands: conversion in the O(M*N*K) loop would make
     * a correctness test unnecessarily expensive on the Android host. */
    float *rounded_a = hmx_fp16 ? (float *) malloc((size_t) m * k * sizeof(float)) : NULL;
    if (hmx_fp16 && !rounded_a) {
        free(logical);
        free(permuted);
        return -1;
    }
    if (hmx_fp16) {
        for (size_t i = 0; i < (size_t) m * k; ++i)
            rounded_a[i] = (float) (__fp16) activation[i];
        /* Reuse the packing scratch for the rounded logical matrix. */
        for (size_t i = 0; i < count; ++i)
            permuted[i] = (float) (__fp16) logical[i];
    }
    const float *dot_a = hmx_fp16 ? rounded_a : activation;
    const float *dot_w = hmx_fp16 ? permuted : logical;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int inner = 0; inner < k; ++inner) {
                sum += (double) dot_a[(size_t) row * k + inner] *
                       dot_w[(size_t) col * k + inner];
                /* Output-stationary HMX reloads its half output through an
                 * identity tile between explicit K chunks. This is a declared
                 * conversion boundary; no packed-layout decoder is reused. */
                if (hmx_fp16 && hmx_k_chunk > 0 && (inner + 1) % hmx_k_chunk == 0)
                    sum = (double) (__fp16) sum;
            }
            reference[(size_t) row * n + col] = hmx_fp16 ? (float) (__fp16) sum : (float) sum;
            if (full_precision) {
                double full = 0;
                for (int inner = 0; inner < k; ++inner)
                    full += (double) activation[(size_t) row * k + inner] *
                            logical[(size_t) col * k + inner];
                full_precision[(size_t) row * n + col] = (float) full;
            }
        }
    }
    free(rounded_a);
    free(logical);
    free(permuted);
    return 0;
}

static inline int v68_test_make_matmul(enum ggml_type type, int m, int k, int n,
                                      uint32_t seed, float *activation,
                                      void *packed, float *reference) {
    return v68_test_make_matmul_ex(type, m, k, n, seed, activation, packed,
                                   reference, 0, 0, NULL);
}

#endif
