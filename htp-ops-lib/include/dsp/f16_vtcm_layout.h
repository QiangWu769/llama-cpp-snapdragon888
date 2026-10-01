#pragma once

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Pure size arithmetic: no Hexagon SDK dependency, allocation or ISA changes.
 * Physical HMX FP16 tiles contain 32 rows and 32 columns even when logical M=1.
 * Every matrix area is a whole number of 2048-byte tiles. */
typedef struct {
  size_t m_chunk_n_rows;
  size_t n_chunk_n_cols;
  size_t weight_size;
  size_t activation_size;
  size_t output_size;
  size_t total_size;
} htp_f16_vtcm_layout;

static inline bool htp_f16_plan_vtcm(size_t usable_size, int m, int k, int n,
                                    size_t activation_limit, htp_f16_vtcm_layout *out) {
  if (!out) {
    return false;
  }
  *out = (htp_f16_vtcm_layout) { 0 };
  if (m <= 0 || k <= 0 || n <= 0 || k % 32 || n % 32 || !activation_limit) {
    return false;
  }

  const uint64_t mk = (uint64_t) m * (uint64_t) k;
  const uint64_t mn = (uint64_t) m * (uint64_t) n;
  const uint64_t kn = (uint64_t) k * (uint64_t) n;
  if (mk > INT_MAX || mn > INT_MAX || kn > INT_MAX ||
      mk > SIZE_MAX / 4 || mn > SIZE_MAX / 4 || kn > SIZE_MAX / 2 || usable_size < 256) {
    return false;
  }

  /* Preserve the established row cap whenever one physical row tile fits.
   * Wider K needs at least 32 rows even for logical M=1. Expand only to that
   * minimum, then require weight, activation, output and control to fit the
   * actual VTCM together; this does not change the quantized-path areas. */
  const uint64_t minimum_activation_size = (uint64_t) k * 2 * 32;
  const uint64_t activation_budget = (uint64_t) activation_limit > minimum_activation_size
                                       ? (uint64_t) activation_limit : minimum_activation_size;
  const uint64_t row_limit = activation_budget / ((uint64_t) k * 2) / 32 * 32;
  uint64_t requested_rows = (uint64_t) m < row_limit ? (uint64_t) m : row_limit;
  requested_rows = (requested_rows + 31) / 32 * 32;
  if (!requested_rows) {
    return false;
  }

  uint64_t best_chunks = UINT64_MAX;
  for (uint64_t rows = 32; rows <= requested_rows; rows += 32) {
    const uint64_t activation_size = rows * (uint64_t) k * 2;
    if (activation_size > (uint64_t) usable_size - 256) {
      break;
    }
    /* Weight and output grow together with N: 2*N*(K+physical_M).
     * Division before multiplication bounds every selected allocation. */
    uint64_t cols = ((uint64_t) usable_size - 256 - activation_size) /
                    (2 * ((uint64_t) k + rows));
    if (cols > (uint64_t) n) {
      cols = (uint64_t) n;
    }
    /* dma_desc_1d.length is 24 bits. The tested phone is below that limit,
     * but do not silently truncate a larger manager-reported allocation.
     * HMX output tile offset helpers also use signed int element indices. */
    const uint64_t dma_cols = UINT64_C(0xFFFFFF) / ((uint64_t) k * 2);
    if (cols > dma_cols) cols = dma_cols;
    if (cols > (uint64_t) INT_MAX / rows) cols = (uint64_t) INT_MAX / rows;
    cols = cols / 32 * 32;
    if (!cols) {
      continue;
    }

    const uint64_t chunks = (((uint64_t) m + rows - 1) / rows) *
                            (((uint64_t) n + cols - 1) / cols);
    if (chunks > best_chunks ||
        (chunks == best_chunks && rows <= out->m_chunk_n_rows)) {
      continue;
    }
    const uint64_t weight_size = cols * (uint64_t) k * 2;
    const uint64_t output_size = rows * cols * 2;
    const uint64_t total_size = weight_size + activation_size + output_size + 256;
    if (total_size > usable_size || total_size > SIZE_MAX) {
      continue;
    }
    best_chunks = chunks;
    out->m_chunk_n_rows = (size_t) rows;
    out->n_chunk_n_cols = (size_t) cols;
    out->weight_size = (size_t) weight_size;
    out->activation_size = (size_t) activation_size;
    out->output_size = (size_t) output_size;
    out->total_size = (size_t) total_size;
  }
  return out->m_chunk_n_rows != 0;
}

static inline bool htp_f16_layout_address_fits(uintptr_t base, size_t usable_size,
                                               const htp_f16_vtcm_layout *layout) {
  return base != 0 && base % 2048 == 0 && layout != NULL && layout->total_size != 0 &&
         layout->total_size <= usable_size && layout->total_size <= UINTPTR_MAX - base;
}
