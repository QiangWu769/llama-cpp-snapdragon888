/*
 * v68 attention fallback: HVX FP32 arithmetic, without HMX, DMA or VTCM.
 *
 * Keep the existing simple_flash_attn ABI: despite its __fp16 pointer types,
 * Q and O contain float elements. Layouts (no batch dimension) are:
 *   Q/O [qo_len][n_heads][head_dim]
 *   K/V [kv_len][n_kv_heads][head_dim]
 *   mask [qo_len][round_up(kv_len, 64)], broadcast across heads.
 * The ABI has no strides/scale arguments: inputs must have these layouts and
 * the attention scale is 1/sqrt(head_dim). Mask values are additive logits in
 * natural-log units; a -infinity entry excludes that key. An entirely masked
 * query produces zero. No implicit causal mask is applied.
 *
 * This is a correctness-oriented fallback, not a tiled FlashAttention kernel.
 * Each worker stores one FP32 score row (kv_len elements), uses a stable
 * two-pass softmax, and accumulates O in FP32. Half-to-float conversion stays
 * scalar to preserve FP16 subnormals and avoid newer vector FP16 instructions.
 * The non-Hexagon build provides the same scalar implementation for host-side
 * layout/bounds tests; it does not emulate or validate HVX instructions.
 */

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__hexagon__)
#include "dsp/hvx_internal.h"
#include "dsp/worker_pool.h"
#endif

typedef struct {
  float *o;
  const float *q;
  const __fp16 *k;
  const __fp16 *v;
  const __fp16 *mask;
  size_t qo_stride;
  size_t kv_stride;
  size_t mask_stride;
  unsigned int n_tasks;
  int kv_len;
  int n_heads;
  int group_size;
  int head_dim;
  float scale;
} fa_v68_args_t;

static int fa_v68_init(fa_v68_args_t *a, float *o, const float *q,
                       const __fp16 *k, const __fp16 *v, const __fp16 *mask,
                       int qo_len, int kv_len, int n_heads, int n_kv_heads,
                       int head_dim) {
  if (!o || !q || !k || !v || qo_len <= 0 || kv_len <= 0 || n_heads <= 0 ||
      n_kv_heads <= 0 || head_dim <= 0 || n_heads % n_kv_heads != 0) {
    return -1;
  }
  /* Leave room for every worker's final atomic increment past the last row. */
  if ((uint64_t) qo_len * (uint64_t) n_heads > UINT_MAX - 6u ||
      (size_t) n_heads > SIZE_MAX / sizeof(float) / (size_t) head_dim ||
      (size_t) n_kv_heads > SIZE_MAX / sizeof(__fp16) / (size_t) head_dim) {
    return -1;
  }

  a->qo_stride = (size_t) n_heads * (size_t) head_dim;
  a->kv_stride = (size_t) n_kv_heads * (size_t) head_dim;
  a->mask_stride = ((size_t) kv_len + 63u) & ~(size_t) 63u;
  if ((size_t) qo_len > SIZE_MAX / sizeof(float) / a->qo_stride ||
      (size_t) kv_len > SIZE_MAX / sizeof(__fp16) / a->kv_stride ||
      (size_t) qo_len > SIZE_MAX / sizeof(__fp16) / a->mask_stride) {
    return -1;
  }

  a->o = o;
  a->q = q;
  a->k = k;
  a->v = v;
  a->mask = mask;
  a->n_tasks = (unsigned int) qo_len * (unsigned int) n_heads;
  a->kv_len = kv_len;
  a->n_heads = n_heads;
  a->group_size = n_heads / n_kv_heads;
  a->head_dim = head_dim;
  a->scale = 1.0f / sqrtf((float) head_dim);
  return 0;
}

static float fa_v68_dot_scalar(const float *q, const __fp16 *k, int n) {
  float sum = 0.0f;
  for (int d = 0; d < n; ++d) {
    sum += q[d] * (float) k[d];
  }
  return sum;
}

static void fa_v68_axpy_scalar(float *o, const __fp16 *v, float p, int n) {
  for (int d = 0; d < n; ++d) {
    o[d] += p * (float) v[d];
  }
}

#if defined(__hexagon__)
static float fa_v68_dot_hvx(const float *q, const __fp16 *k, int n) {
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector sum = zero;
  _Alignas(128) float converted[32];
  int d = 0;
  for (; d <= n - 32; d += 32) {
    for (int j = 0; j < 32; ++j) {
      converted[j] = (float) k[d + j];
    }
    const HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(vmemu(q + d), vmem(converted));
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, prod);
  }
  for (int shift = 64; shift >= 4; shift >>= 1) {
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, zero, shift));
  }
  /* qf32 is not IEEE FP32: convert before reading a lane as a C float. */
  vmem(converted) = Q6_Vsf_equals_Vqf32(sum);
  float result = converted[31];
  for (; d < n; ++d) {
    result += q[d] * (float) k[d];
  }
  return result;
}

static void fa_v68_axpy_hvx(float *o, const __fp16 *v, float p, int n) {
  uint32_t bits;
  memcpy(&bits, &p, sizeof(bits));
  const HVX_Vector weight = Q6_V_vsplat_R((int32_t) bits);
  _Alignas(128) float converted[32];
  int d = 0;
  for (; d <= n - 32; d += 32) {
    for (int j = 0; j < 32; ++j) {
      converted[j] = (float) v[d + j];
    }
    const HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(vmem(converted), weight);
    const HVX_Vector sum = Q6_Vqf32_vadd_Vqf32Vsf(prod, vmemu(o + d));
    vmemu(o + d) = Q6_Vsf_equals_Vqf32(sum);
  }
  fa_v68_axpy_scalar(o + d, v + d, p, n - d);
}
#endif

static void fa_v68_row(const fa_v68_args_t *a, unsigned int task, float *scores,
                       bool use_hvx) {
  const size_t query = task / (unsigned int) a->n_heads;
  const size_t head = task % (unsigned int) a->n_heads;
  const size_t kv_head = head / (unsigned int) a->group_size;
  const size_t offset = query * a->qo_stride + head * (size_t) a->head_dim;
  const float *q = a->q + offset;
  float *o = a->o + offset;
  const __fp16 *mask = a->mask ? a->mask + query * a->mask_stride : NULL;
  const __fp16 *k = a->k + kv_head * (size_t) a->head_dim;
  const __fp16 *v = a->v + kv_head * (size_t) a->head_dim;
  float maximum = -INFINITY;

#if !defined(__hexagon__)
  (void) use_hvx;
#endif
  for (int c = 0; c < a->kv_len; ++c) {
    const float bias = mask ? (float) mask[c] : 0.0f;
    if (bias == -INFINITY) {
      scores[c] = -INFINITY;
      continue;
    }
    const __fp16 *key = k + (size_t) c * a->kv_stride;
    float dot;
#if defined(__hexagon__)
    if (use_hvx) {
      dot = fa_v68_dot_hvx(q, key, a->head_dim);
    } else
#endif
    {
      dot = fa_v68_dot_scalar(q, key, a->head_dim);
    }
    scores[c] = dot * a->scale + bias;
    maximum = fmaxf(maximum, scores[c]);
  }

  memset(o, 0, (size_t) a->head_dim * sizeof(float));
  if (maximum == -INFINITY) {
    return;
  }

  float denominator = 0.0f;
  for (int c = 0; c < a->kv_len; ++c) {
    /* Also give a well-defined equal weighting to positive-infinite logits. */
    const float p = maximum == INFINITY ? (scores[c] == INFINITY ? 1.0f : 0.0f)
                                       : expf(scores[c] - maximum);
    scores[c] = p;
    denominator += p;
  }
  const float inv_denominator = 1.0f / denominator;
  for (int c = 0; c < a->kv_len; ++c) {
    const float p = scores[c] * inv_denominator;
    if (p == 0.0f) {
      continue;  /* Masked values need not be initialized or finite. */
    }
    const __fp16 *value = v + (size_t) c * a->kv_stride;
#if defined(__hexagon__)
    if (use_hvx) {
      fa_v68_axpy_hvx(o, value, p, a->head_dim);
    } else
#endif
    {
      fa_v68_axpy_scalar(o, value, p, a->head_dim);
    }
  }
}

static int fa_v68_serial(const fa_v68_args_t *a) {
  if ((size_t) a->kv_len > SIZE_MAX / sizeof(float)) {
    return -1;
  }
  float *scores = (float *) malloc((size_t) a->kv_len * sizeof(float));
  if (!scores) {
    return -1;
  }
  for (unsigned int task = 0; task < a->n_tasks; ++task) {
    fa_v68_row(a, task, scores, false);
  }
  free(scores);
  return 0;
}

#if defined(__hexagon__)
typedef struct {
  const fa_v68_args_t *args;
  worker_synctoken_t sync;
  unsigned int next_task;
} fa_v68_shared_t;

typedef struct {
  fa_v68_shared_t *shared;
  float *scores;
} fa_v68_job_t;

static void fa_v68_worker(void *data, int worker_index) {
  fa_v68_job_t *job = (fa_v68_job_t *) data;
  fa_v68_shared_t *shared = job->shared;
  (void) worker_index;  /* Scratch belongs to the submitted job, not the thread. */
  for (;;) {
    unsigned int task = worker_pool_atomic_inc_return(&shared->next_task) - 1u;
    if (task >= shared->args->n_tasks) {
      break;
    }
    fa_v68_row(shared->args, task, job->scores, true);
  }
  worker_pool_synctoken_jobdone(&shared->sync);
}
#endif

int simple_flash_attn(__fp16 *restrict O, const __fp16 *restrict Q,
                      const __fp16 *restrict K, const __fp16 *restrict V,
                      const __fp16 *restrict mask, int qo_len, int kv_len,
                      int n_heads, int n_kv_heads, int head_dim) {
  fa_v68_args_t args;
  if (fa_v68_init(&args, (float *) O, (const float *) Q, K, V, mask,
                  qo_len, kv_len, n_heads, n_kv_heads, head_dim) != 0) {
    return -1;
  }
#if defined(__hexagon__)
  unsigned int n_jobs = num_hvx128_contexts;
  if (n_jobs > num_workers) n_jobs = num_workers;
  if (n_jobs > MAX_NUM_WORKERS) n_jobs = MAX_NUM_WORKERS;
  if (n_jobs > args.n_tasks) n_jobs = args.n_tasks;
  if (n_jobs == 0 ||
      (size_t) kv_len > SIZE_MAX / sizeof(float) / n_jobs) {
    return -1;
  }

  float *scratch = (float *) malloc((size_t) kv_len * n_jobs * sizeof(float));
  if (!scratch) {
    return -1;
  }
  fa_v68_shared_t shared = { .args = &args, .next_task = 0 };
  fa_v68_job_t jobs[MAX_NUM_WORKERS];
  worker_pool_synctoken_init(&shared.sync, n_jobs);
  int result = 0;
  for (unsigned int i = 0; i < n_jobs; ++i) {
    jobs[i].shared = &shared;
    jobs[i].scores = scratch + (size_t) i * (size_t) kv_len;
    worker_pool_job_t job = { .fptr = fa_v68_worker, .dptr = &jobs[i] };
    if (worker_pool_submit(NULL, job) != AEE_SUCCESS) {
      /* Account for an unsubmitted job, otherwise the waiter could deadlock. */
      worker_pool_synctoken_jobdone(&shared.sync);
      result = -1;
    }
  }
  worker_pool_synctoken_wait(&shared.sync);
  free(scratch);
  return result;
#else
  return fa_v68_serial(&args);
#endif
}

int naive_flash_attn(float *restrict O, const float *restrict Q,
                     const __fp16 *restrict K, const __fp16 *restrict V,
                     const __fp16 *restrict mask, int qo_len, int kv_len,
                     int n_heads, int n_kv_heads, int head_dim) {
  fa_v68_args_t args;
  if (fa_v68_init(&args, O, Q, K, V, mask, qo_len, kv_len, n_heads,
                  n_kv_heads, head_dim) != 0) {
    return -1;
  }
  return fa_v68_serial(&args);
}
