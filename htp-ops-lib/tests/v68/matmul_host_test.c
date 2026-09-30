#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "dsp/ops.h"
#include "matmul_reference.h"

static int test_case(enum ggml_type type, int m, int k, int n, int shifted) {
    const size_t elements = (size_t) m * n;
    const size_t guard = 32 + shifted;
    float *activation = malloc((size_t) m * k * sizeof(float));
    float *reference = malloc(elements * sizeof(float));
    float *allocation = malloc((elements + 96) * sizeof(float));
    void *weight = malloc(v68_test_weight_bytes(type, k, n));
    if (!activation || !reference || !allocation || !weight) abort();
    float *output = allocation + guard;
    for (size_t i = 0; i < elements + 96; ++i) allocation[i] = -9876.5f;
    for (size_t i = 0; i < elements; ++i) output[i] = NAN;
    int failed = v68_test_make_matmul(type, m, k, n, (uint32_t) (m + 17 * k + n),
                                      activation, weight, reference);
    if (!failed) {
        failed = type == GGML_TYPE_F16 ?
            hmx_mat_mul_permuted_w16a32(output, activation, weight, m, k, n) :
            hmx_mat_mul_permuted_qk_0_d16a32(output, activation, weight, m, k, n, type);
    }
    for (size_t i = 0; !failed && i < elements; ++i) {
        if (!isfinite(output[i]) || fabsf(output[i] - reference[i]) >
            2e-4f + 2e-4f * fabsf(reference[i])) {
            fprintf(stderr, "FAIL %s M=%d K=%d N=%d shifted=%d index=%zu got=%g expected=%g\n",
                    v68_test_type_name(type), m, k, n, shifted, i, output[i], reference[i]);
            failed = 1;
        }
    }
    for (size_t i = 0; i < guard; ++i) if (allocation[i] != -9876.5f) failed = 1;
    for (size_t i = guard + elements; i < elements + 96; ++i)
        if (allocation[i] != -9876.5f) failed = 1;
    free(activation);
    free(reference);
    free(allocation);
    free(weight);
    return failed != 0;
}

int main(void) {
    const enum ggml_type types[] = {GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL};
    const int ms[] = {1, 2, 3, 4, 5, 9}, ks[] = {32, 64, 96, 256}, ns[] = {32, 64, 96};
    int cases = 0;
    for (int t = 0; t < 4; ++t)
        for (int m = 0; m < 6; ++m)
            for (int k = 0; k < 4; ++k)
                for (int n = 0; n < 3; ++n)
                    for (int shifted = 0; shifted < 2; ++shifted) {
                        if (test_case(types[t], ms[m], ks[k], ns[n], shifted)) return 1;
                        ++cases;
                    }
    float output[32] = {0}, activation[32] = {0};
    __fp16 weight[1024] = {0};
    if (hmx_mat_mul_permuted_w16a32(output, activation, weight, 1, 31, 32) != -1 ||
        hmx_mat_mul_permuted_w16a32(output, activation, weight, 1, 32, 31) != -1 ||
        hmx_mat_mul_permuted_w16a32(NULL, activation, weight, 1, 32, 32) != -1 ||
        hmx_mat_mul_permuted_w16a32(output, activation, weight, 0, 32, 32) != -1 ||
        hmx_mat_mul_permuted_qk_0_d16a32(output, activation, (uint8_t *) weight,
                                       1, 32, 32, GGML_TYPE_Q2_K) != -1) {
        fprintf(stderr, "FAIL invalid-input rejection\n");
        return 1;
    }
    printf("PASS %d matmul layout/control-flow cases + guards + invalid-input checks\n", cases);
    puts("Host scalar semantics only: this does not validate Hexagon qfloat precision or hardware execution.");
    return 0;
}
