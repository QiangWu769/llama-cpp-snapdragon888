#pragma once

#include <stdbool.h>

#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

bool htp_ops_support_op(const struct ggml_tensor * dst);
// The HTP converter permutes these weights; ordinary CPU matmul cannot read them.
bool htp_ops_has_permuted_weight(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);
// Caller holds compute_mutex; this takes request_mutex through release/free.
void htp_ops_retire_and_free_rpcmem(void *base);

#ifdef __cplusplus
}
#endif
