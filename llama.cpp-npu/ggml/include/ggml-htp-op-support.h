#pragma once

#include "ggml.h"
#include "ggml-htp-output-pack.h"
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* Experimental partitioning for the converter's packed GGUFs. Evaluate the
 * toggle before creating a model/context and leave it fixed for that lifetime. */
static inline bool ggml_htp_split_cpu_ops_enabled(void) {
    const char *value = getenv("HTP_SPLIT_CPU_OPS");
    return value && strcmp(value, "1") == 0;
}

static inline bool ggml_htp_has_permuted_weight(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT || !dst->src[0]) {
        return false;
    }
    // This is the converter's layout contract, not a generic GGUF type property.
    // In particular, output/token embeddings remain in ordinary row-major layout.
    const char * name = dst->src[0]->name;
    if (ggml_htp_output_weight_part(name) >= 0) return true;
    if (strncmp(name, "blk.", 4) != 0) {
        // The GGML scheduler names tensor copies "<backend>#<original>#<copy>".
        const char * original = strstr(name, "#blk.");
        if (!original) {
            return false;
        }
        name = original + 1;
    }
    const char * suffix = name + 4;
    if (*suffix < '0' || *suffix > '9') {
        return false;
    }
    do {
        ++suffix;
    } while (*suffix >= '0' && *suffix <= '9');
    static const char * const permuted_suffixes[] = {
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_up.weight", ".ffn_down.weight", ".ffn_gate.weight",
    };
    for (size_t i = 0; i < sizeof(permuted_suffixes) / sizeof(permuted_suffixes[0]); ++i) {
        const char *expected = permuted_suffixes[i];
        const size_t length = strlen(expected);
        if (strncmp(suffix, expected, length) != 0) {
            continue;
        }
        const char * end = suffix + length;
        while (*end == '#') {
            ++end;
            if (*end < '0' || *end > '9') {
                return false;
            }
            do {
                ++end;
            } while (*end >= '0' && *end <= '9');
        }
        return *end == '\0';
    }
    return false;
}


static inline bool ggml_htp_interleaved_attention_layout(const struct ggml_tensor *tensor, size_t element_size) {
    return tensor && tensor->ne[3] == 1 && tensor->nb[0] == element_size &&
           tensor->nb[1] == element_size * tensor->ne[0] * tensor->ne[2] &&
           tensor->nb[2] == element_size * tensor->ne[0];
}

/* Capability-only checks: no DSP initialization and no buffer requirements.
 * The loader and scheduler call supports_op before all buffers exist. */
static inline bool ggml_htp_op_layout_supported(const struct ggml_tensor *dst) {
    if (!dst) return false;
    if (dst->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor *weight = dst->src[0];
        const struct ggml_tensor *activation = dst->src[1];
        if (!weight || !activation || !ggml_htp_has_permuted_weight(dst)) return false;
        const int64_t k = weight->ne[0], n = weight->ne[1];
        const bool shape_ok = k > 0 && n > 0 && k <= INT_MAX && n <= INT_MAX &&
            activation->ne[1] > 0 && activation->ne[1] <= INT_MAX && k % 32 == 0 && n % 32 == 0 &&
            weight->ne[2] == 1 && weight->ne[3] == 1 &&
            ggml_nrows(dst) == dst->ne[1] && ggml_nrows(activation) == activation->ne[1] &&
            activation->ne[0] == k && dst->ne[0] == n && dst->ne[1] == activation->ne[1] &&
            ggml_is_contiguous(weight) && ggml_is_contiguous(activation) && ggml_is_contiguous(dst);
        const bool weight_type = weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_Q4_0 ||
                                 weight->type == GGML_TYPE_Q8_0 || weight->type == GGML_TYPE_IQ4_NL;
        return shape_ok && weight_type && dst->type == GGML_TYPE_F32 && activation->type == GGML_TYPE_F32;
    }
    if (dst->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const char *disabled = getenv("HTP_DISABLE_FLASH_ATTN");
    if (disabled && disabled[0] && strcmp(disabled, "0") != 0) return false;
    const struct ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mask = dst->src[3];
    if (!q || !k || !v || !mask) return false;
    float scale, max_bias, logit_softcap;
    memcpy(&scale, &dst->op_params[0], sizeof(scale));
    memcpy(&max_bias, &dst->op_params[1], sizeof(max_bias));
    memcpy(&logit_softcap, &dst->op_params[2], sizeof(logit_softcap));
    const int64_t dim = q->ne[0];
    // Snapdragon 888 v68 fa68_init supports at most 512 elements per head.
    if (dim <= 0 || dim > 512 || q->ne[1] <= 0 || q->ne[1] > INT_MAX ||
        q->ne[2] <= 0 || q->ne[2] > INT_MAX || k->ne[1] <= 0 || k->ne[1] > INT_MAX ||
        k->ne[2] <= 0 || k->ne[2] > INT_MAX) return false;
    const float expected_scale = 1.0f / sqrtf((float)dim);
    const int64_t mask_stride = ((k->ne[1] + 63) / 64) * 64;
    return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
           v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 && max_bias == 0 && logit_softcap == 0 &&
           fabsf(scale - expected_scale) <= 1e-6f * expected_scale &&
           k->ne[0] == dim && v->ne[0] == dim && v->ne[1] == k->ne[1] &&
           v->ne[2] == k->ne[2] && q->ne[2] % k->ne[2] == 0 &&
           ggml_htp_interleaved_attention_layout(q, sizeof(float)) &&
           ggml_htp_interleaved_attention_layout(k, sizeof(ggml_fp16_t)) &&
           ggml_htp_interleaved_attention_layout(v, sizeof(ggml_fp16_t)) &&
           dst->ne[0] == dim && dst->ne[1] == q->ne[2] && dst->ne[2] == q->ne[1] &&
           dst->ne[3] == 1 && ggml_is_contiguous(dst) &&
           mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] &&
           mask->ne[2] == 1 && mask->ne[3] == 1 && mask->nb[0] == sizeof(ggml_fp16_t) &&
           mask->nb[1] == mask_stride * sizeof(ggml_fp16_t);
}
