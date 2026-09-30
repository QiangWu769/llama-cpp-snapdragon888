#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/f16_vtcm_layout.h"

enum { ACTIVATION_LIMIT = 512 * 1024, PHONE_USABLE = 4 * 1024 * 1024 - 256 * 1024 };

/* Independently enumerate feasible pairs, including smaller N blocks, rather
 * than reproducing the production division that selects the largest N. */
static bool oracle(size_t usable, int m, int k, int n, size_t *best_rows, size_t *best_cols) {
  size_t row_limit = ACTIVATION_LIMIT / ((size_t) k * 2) / 32 * 32;
  size_t requested = ((size_t) m + 31) / 32 * 32;
  if (row_limit > requested) row_limit = requested;
  uint64_t best_cost = UINT64_MAX;
  *best_rows = *best_cols = 0;
  for (size_t rows = 32; rows <= row_limit; rows += 32) {
    for (size_t cols = 32; cols <= (size_t) n; cols += 32) {
      uint64_t needed = 2 * ((uint64_t) rows * k + (uint64_t) cols * k + (uint64_t) rows * cols) + 256;
      if (needed > usable || (uint64_t) cols * k * 2 > UINT64_C(0xFFFFFF) ||
          (uint64_t) rows * cols > INT_MAX) continue;
      uint64_t cost = (((size_t) m + rows - 1) / rows) * (((size_t) n + cols - 1) / cols);
      if (cost < best_cost || (cost == best_cost && rows > *best_rows) ||
          (cost == best_cost && rows == *best_rows && cols > *best_cols)) {
        best_cost = cost;
        *best_rows = rows;
        *best_cols = cols;
      }
    }
  }
  return *best_rows != 0;
}

static void check(int m, int k, int n, size_t usable) {
  htp_f16_vtcm_layout l;
  size_t rows, cols;
  bool expected = oracle(usable, m, k, n, &rows, &cols);
  assert(htp_f16_plan_vtcm(usable, m, k, n, ACTIVATION_LIMIT, &l) == expected);
  if (!expected) return;
  assert(l.m_chunk_n_rows == rows && l.n_chunk_n_cols == cols);
  assert(rows % 32 == 0 && cols % 32 == 0 && cols <= (size_t) n);
  assert(l.weight_size == cols * (size_t) k * 2);
  assert(l.activation_size == rows * (size_t) k * 2);
  assert(l.output_size == rows * cols * 2);
  assert(l.total_size == l.weight_size + l.activation_size + l.output_size + 256);
  assert(l.total_size <= usable && l.activation_size <= ACTIVATION_LIMIT);
  assert(l.weight_size % 2048 == 0 && l.activation_size % 2048 == 0 && l.output_size % 2048 == 0);

  /* Use the actual sequential allocation boundaries, guard both ends and
   * reserved VTCM, and exercise every physical output tile including tails. */
  uint8_t *allocation = malloc(usable + 4096 + 64);
  uint8_t *visits = calloc((size_t) m * n, 1);
  assert(allocation && visits);
  memset(allocation, 0xA7, usable + 4096 + 64);
  uint8_t *base = (uint8_t *) (((uintptr_t) allocation + 2048 + 2047) & ~(uintptr_t) 2047);
  assert(htp_f16_layout_address_fits((uintptr_t) base, usable, &l));
  uint8_t *weight = base;
  uint8_t *activation = weight + l.weight_size;
  uint8_t *output = activation + l.activation_size;
  uint8_t *scales = output + l.output_size;
  assert((uintptr_t) activation % 2048 == 0 && (uintptr_t) output % 2048 == 0);
  assert((uintptr_t) scales % 256 == 0);
  memset(weight, 0x11, l.weight_size);
  memset(activation, 0x22, l.activation_size);
  memset(scales, 0x44, 256);
  for (size_t mr = 0; mr < (size_t) m; mr += rows) {
    size_t valid_rows = (size_t) m - mr < rows ? (size_t) m - mr : rows;
    for (size_t nc = 0; nc < (size_t) n; nc += cols) {
      size_t valid_cols = (size_t) n - nc < cols ? (size_t) n - nc : cols;
      size_t row_tiles = (valid_rows + 31) / 32, col_tiles = valid_cols / 32;
      for (size_t r = 0; r < row_tiles; ++r) {
        /* Last K tile of every full physical activation row tile. */
        assert(((r + 1) * (size_t) k / 32 * 2048) <= l.activation_size);
        for (size_t c = 0; c < col_tiles; ++c) {
          size_t tile_offset = (r * col_tiles + c) * 2048;
          assert(tile_offset + 2048 <= l.output_size);
          memset(output + tile_offset, 0x33, 2048);
        }
      }
      for (size_t r = 0; r < valid_rows; ++r) {
        for (size_t c = 0; c < valid_cols; ++c) ++visits[(mr + r) * (size_t) n + nc + c];
      }
    }
  }
  for (size_t i = 0; i < (size_t) m * n; ++i) assert(visits[i] == 1);
  for (uint8_t *p = allocation; p < base; ++p) assert(*p == 0xA7);
  for (size_t i = 0; i < l.weight_size; ++i) assert(weight[i] == 0x11);
  for (size_t i = 0; i < l.activation_size; ++i) assert(activation[i] == 0x22);
  for (size_t i = 0; i < 256; ++i) assert(scales[i] == 0x44);
  for (uint8_t *p = base + l.total_size; p < allocation + usable + 4096 + 64; ++p) assert(*p == 0xA7);
  free(visits);
  free(allocation);
}

int main(void) {
  const int ms[] = { 1, 2, 5, 31, 32, 33, 63, 64, 65, 128, 129, 288, 289 };
  const int shapes[][2] = { {32,32}, {96,96}, {896,128}, {896,896}, {896,4864}, {4864,896}, {8192,32} };
  const size_t capacities[] = { 0, 6399, 6400, 512 * 1024, 1536 * 1024 + 256, PHONE_USABLE };
  unsigned cases = 0;
  for (size_t a = 0; a < sizeof(ms) / sizeof(ms[0]); ++a)
    for (size_t b = 0; b < sizeof(shapes) / sizeof(shapes[0]); ++b)
      for (size_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); ++c) {
        check(ms[a], shapes[b][0], shapes[b][1], capacities[c]);
        ++cases;
      }

  htp_f16_vtcm_layout l;
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 0, 896, 896, ACTIVATION_LIMIT, &l));
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 1, 895, 896, ACTIVATION_LIMIT, &l));
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 1, 896, 895, ACTIVATION_LIMIT, &l));
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 1, 896, 896, ACTIVATION_LIMIT, NULL));
  assert(!htp_f16_plan_vtcm(SIZE_MAX, INT_MAX, 32, 32, ACTIVATION_LIMIT, &l));
  assert(!htp_f16_plan_vtcm(SIZE_MAX, 1, INT_MAX - 31, INT_MAX - 31, ACTIVATION_LIMIT, &l));
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 1, 896, 896, 0, &l));
  assert(!htp_f16_plan_vtcm(PHONE_USABLE, 1, 8224, 32, ACTIVATION_LIMIT, &l));
  assert(htp_f16_plan_vtcm(SIZE_MAX, 1, 32, 1048576, ACTIVATION_LIMIT, &l));
  assert(l.weight_size <= 0xFFFFFF && l.n_chunk_n_cols == 262112);
  assert(htp_f16_plan_vtcm(PHONE_USABLE, 1, 4864, 896, ACTIVATION_LIMIT, &l));
  assert(l.m_chunk_n_rows == 32 && l.n_chunk_n_cols == 352);
  assert(l.activation_size == 32 * 4864 * 2); /* Never allocate only one row. */
  assert(!htp_f16_layout_address_fits(0, PHONE_USABLE, &l));
  assert(!htp_f16_layout_address_fits(2049, PHONE_USABLE, &l));
  assert(!htp_f16_layout_address_fits(2048, l.total_size - 1, &l));
  assert(!htp_f16_layout_address_fits(UINTPTR_MAX & ~(uintptr_t) 2047, PHONE_USABLE, &l));
  assert(htp_f16_layout_address_fits(2048, l.total_size, &l));
  printf("FP16 VTCM planner PASS: %u shape/capacity cases; full-tile bounds, reserved-memory guards and address checks\n", cases);
  return 0;
}
