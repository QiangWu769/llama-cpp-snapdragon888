/* Host-side attention layout/numerical tests; no Hexagon SDK or HVX stubs. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp/ops.h"

_Static_assert(sizeof(__fp16) == 2, "attention requires IEEE binary16 storage");
_Static_assert(sizeof(float) == 4, "attention requires IEEE binary32 storage");

static uint32_t random_state = 1;
static double max_error, max_normal_error;
#if defined(ATTENTION_FP16_TILES)
static const double normal_tolerance = 0.002;
static const double large_tolerance = 0.025;
#else
static const double normal_tolerance = 0.0002;
static const double large_tolerance = 0.0002;
#endif

static float random_float(void) {
  random_state = random_state * 1664525u + 1013904223u;
  return ((float) ((random_state >> 8) % 2001) - 1000.0f) / 1000.0f;
}

static void *checked_malloc(size_t size) {
  void *p = malloc(size);
  assert(p != NULL);
  return p;
}

/* The reference materializes one double score row and computes softmax in
 * double, independently of the FP32 accumulation in the production kernel. */
static void run_case(int nq, int nk, int nh, int nkh, int d, int mask_mode,
                     float magnitude) {
  const size_t qsize = (size_t) nq * nh * d;
  const size_t ksize = (size_t) nk * nkh * d;
  const size_t mask_stride = ((size_t) nk + 63) & ~(size_t) 63;
  float *q = checked_malloc(qsize * sizeof(float));
  float *o = checked_malloc(qsize * sizeof(float));
  float *scalar_o = checked_malloc(qsize * sizeof(float));
  __fp16 *k = checked_malloc(ksize * sizeof(__fp16));
  __fp16 *v = checked_malloc(ksize * sizeof(__fp16));
  __fp16 *mask = mask_mode ? checked_malloc((size_t) nq * mask_stride * sizeof(__fp16)) : NULL;
  double *scores = checked_malloc((size_t) nk * sizeof(double));

  for (size_t i = 0; i < qsize; ++i) q[i] = random_float() * magnitude;
  for (size_t i = 0; i < ksize; ++i) {
    k[i] = (__fp16) (random_float() * magnitude);
    v[i] = (__fp16) random_float();
  }
  if (mask) {
    for (int a = 0; a < nq; ++a) {
      for (size_t b = 0; b < mask_stride; ++b) {
        /* NaN padding detects accidental inclusion of padded keys. */
        float bias = b >= (size_t) nk ? NAN : (mask_mode == 3 ? -2.0f * (float) (b % 5) : 0.0f);
        if (b < (size_t) nk && ((mask_mode == 1 && b > (size_t) a) ||
                               (mask_mode == 2 && a == 0) ||
                               (mask_mode == 3 && b % 7 == 0))) {
          bias = -INFINITY;
        }
        mask[(size_t) a * mask_stride + b] = (__fp16) bias;
      }
    }
  }

  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, mask,
                           nq, nk, nh, nkh, d) == 0);
  assert(naive_flash_attn(scalar_o, q, k, v, mask, nq, nk, nh, nkh, d) == 0);
  for (int a = 0; a < nq; ++a) {
    for (int h = 0; h < nh; ++h) {
      const int kh = h / (nh / nkh);
      double maximum = -INFINITY;
      double total = 0;
      for (int b = 0; b < nk; ++b) {
        double dot = 0;
        for (int x = 0; x < d; ++x) {
          dot += (double) q[((size_t) a * nh + h) * d + x] *
                 (double) k[((size_t) b * nkh + kh) * d + x];
        }
        scores[b] = dot / sqrt((double) d) +
                    (mask ? (double) mask[(size_t) a * mask_stride + b] : 0.0);
        if (scores[b] > maximum) maximum = scores[b];
      }
      for (int b = 0; b < nk; ++b) {
        scores[b] = maximum == -INFINITY ? 0 : exp(scores[b] - maximum);
        total += scores[b];
      }
      for (int x = 0; x < d; ++x) {
        double expected = 0;
        for (int b = 0; b < nk; ++b) {
          expected += scores[b] * (double) v[((size_t) b * nkh + kh) * d + x];
        }
        if (total > 0) expected /= total;
        const size_t i = ((size_t) a * nh + h) * d + x;
        const double error = fabs(o[i] - expected);
        if (error > max_error) max_error = error;
        if (magnitude <= 1 && error > max_normal_error) max_normal_error = error;
        if (!isfinite(o[i]) || error > (magnitude > 1 ? large_tolerance : normal_tolerance) || o[i] != scalar_o[i]) {
          fprintf(stderr, "FAIL nq=%d nk=%d nh=%d nkh=%d d=%d mask=%d "
                          "i=%zu got=%g ref=%.12g error=%g\n",
                  nq, nk, nh, nkh, d, mask_mode, i, o[i], expected, error);
          abort();
        }
      }
    }
  }
  free(q);
  free(o);
  free(scalar_o);
  free(k);
  free(v);
  free(mask);
  free(scores);
}

static void special_cases(void) {
  enum { D = 33, NK = 3, NQ = 2, MASK_STRIDE = 64 };
  float q[NQ * D] = { 0 };
  float o[NQ * D];
  __fp16 k[NK * D];
  __fp16 v[NK * D];
  __fp16 mask[NQ * MASK_STRIDE];
  for (int b = 0; b < NK; ++b) {
    for (int d = 0; d < D; ++d) {
      k[b * D + d] = (__fp16) (b == 2 ? NAN : 0.0f);
      v[b * D + d] = (__fp16) (b == 2 ? NAN : (float) (b + 1) * 0x1p-24f);
    }
  }
  for (int i = 0; i < NQ * MASK_STRIDE; ++i) mask[i] = (__fp16) -INFINITY;
  mask[0] = mask[1] = (__fp16) 0;
  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, mask,
                           NQ, NK, 1, 1, D) == 0);
  for (int d = 0; d < D; ++d) {
    #if defined(ATTENTION_FP16_TILES)
    assert(fabsf(o[d] - 1.5f * 0x1p-24f) <= 0x1p-24f);
#else
    assert(o[d] == 1.5f * 0x1p-24f);  /* Preserve FP16 subnormal V values. */
#endif
    assert(o[D + d] == 0.0f);         /* All masked, including poisoned K/V. */
  }

  /* Positive-infinite mask entries divide probability equally. */
  mask[0] = mask[1] = (__fp16) INFINITY;
  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, mask,
                           NQ, NK, 1, 1, D) == 0);
  for (int d = 0; d < D; ++d) {
#if defined(ATTENTION_FP16_TILES)
    assert(fabsf(o[d] - 1.5f * 0x1p-24f) <= 0x1p-24f);
#else
    assert(o[d] == 1.5f * 0x1p-24f);
#endif
  }

  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, NULL,
                           1, 1, 1, 0, 1) == -1);
  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, NULL,
                           1, 1, 3, 2, 1) == -1);
  assert(simple_flash_attn((__fp16 *) o, (const __fp16 *) q, k, v, NULL,
                           1, 1, 1, 1, 0) == -1);
  assert(simple_flash_attn(NULL, (const __fp16 *) q, k, v, NULL,
                           1, 1, 1, 1, 1) == -1);
}

int main(void) {
  const int dims[] = { 1, 7, 31, 32, 33, 64, 65, 80, 96, 128, 256 };
  const int lengths[] = { 1, 7, 63, 64, 65, 129, 257 };
  int cases = 0;
  for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); ++di) {
    for (size_t ki = 0; ki < sizeof(lengths) / sizeof(lengths[0]); ++ki) {
      for (int mask = 0; mask < 4; ++mask) {
        run_case(3, lengths[ki], 8, 2, dims[di], mask, 1);
        run_case(1, lengths[ki], 4, 4, dims[di], mask, 8);
        cases += 2;
      }
    }
  }
  /* Cross the 32-row tile boundary and use an odd GQA group (Qwen2.5). */
  run_case(5, 67, 14, 2, 64, 3, 1);
  run_case(33, 33, 2, 2, 64, 1, 1);
  cases += 2;
  special_cases();
  printf("PASS %d attention cases; max absolute error %.9g; "
         "normal-input max %.9g; subnormal/masked-NaN/infinite-mask/invalid-input checks passed\n", cases, max_error, max_normal_error);
  return 0;
}
