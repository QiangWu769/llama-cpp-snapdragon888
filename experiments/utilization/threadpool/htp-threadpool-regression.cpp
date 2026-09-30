#include "ggml.h"
#include "ggml-cpu.h"
#include "htp-cpu-impl.h"
#include "htp-ops.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" bool htp_ops_support_op(const ggml_tensor *) { return false; }
extern "C" bool htp_ops_has_permuted_weight(const ggml_tensor *) { return false; }
extern "C" int htp_ops_compute_op(ggml_compute_params *, ggml_tensor *) { std::abort(); }
static bool abort_once(void * opaque) {
    auto * calls = static_cast<int *>(opaque);
    return ++*calls == 1;
}
int main() {
    ggml_cpu_init();
    unsigned graphs = 0;
    for (const int threads : {4, 2, 1, 6, 3, 4}) {
        auto * pool = ggml_htp_threadpool_new(threads);
        std::fprintf(stderr, "threads=%d\n", threads);
        for (int pass = 0; pass < 100; ++pass) {
            ggml_init_params ip{8 * 1024 * 1024, nullptr, false};
            auto * context = ggml_init(ip);
            assert(context);
            auto * a = ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, 32 + pass % 5);
            auto * b = ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, 32 + pass % 5);
            for (int64_t i = 0; i < ggml_nelements(a); ++i) {
                static_cast<float *>(a->data)[i] = float((i + pass) % 113) / 113.0f;
                static_cast<float *>(b->data)[i] = float((i + 2 * pass) % 97) / 97.0f;
            }
            auto * sum = ggml_add(context, a, b);
            auto * result = ggml_rms_norm(context, ggml_scale(context, sum, 0.7f), 1e-5f);
            auto * graph = ggml_new_graph(context);
            ggml_build_forward_expand(graph, result);
            const int active_threads = 1 + pass % threads;
            auto plan = ggml_graph_plan(graph, active_threads, nullptr);
            std::vector<uint8_t> scratch(plan.work_size);
            plan.work_data = scratch.data();
            assert(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_SUCCESS);
            std::vector<float> baseline(ggml_nelements(result));
            std::memcpy(baseline.data(), result->data, baseline.size() * sizeof(float));
            std::memset(result->data, 0, baseline.size() * sizeof(float));
            plan.threadpool = pool;
            assert(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_SUCCESS);
            assert(std::memcmp(baseline.data(), result->data, baseline.size() * sizeof(float)) == 0);
            if (pass % 20 == 0) {
                int calls = 0;
                plan.abort_callback = abort_once;
                plan.abort_callback_data = &calls;
                assert(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_ABORTED);
                plan.abort_callback = nullptr;
                plan.abort_callback_data = nullptr;
                assert(ggml_graph_compute_htp_hybrid(graph, &plan) == GGML_STATUS_SUCCESS);
                assert(std::memcmp(baseline.data(), result->data, baseline.size() * sizeof(float)) == 0);
            }
            // Immediately free the old graph/plan backing storage to exercise
            // the reused-pool completion barrier under rapid graph changes.
            ggml_free(context);
            ++graphs;
        }
        ggml_htp_threadpool_free(pool);
    }
    ggml_htp_threadpool_free(nullptr);
    std::printf("PASS: %u fresh graphs, six pool thread counts, bit-identical disposable/reused results, abort/recovery and teardown\n", graphs);
}
