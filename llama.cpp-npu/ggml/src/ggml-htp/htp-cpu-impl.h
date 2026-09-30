#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// These pools belong to the hybrid executor, whose worker implementation is
// separate from ggml-cpu. Do not attach a pool created by ggml_threadpool_new.
struct ggml_threadpool * ggml_htp_threadpool_new(int n_threads);
void ggml_htp_threadpool_free(struct ggml_threadpool * threadpool);

enum ggml_status ggml_graph_compute_htp_hybrid(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

#ifdef __cplusplus
}
#endif
