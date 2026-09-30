#pragma once

#include "ggml-cpp.h"
#include <array>

/* Model-owned, runtime-only tensors. Buffers are destroyed before metadata;
 * the HTP allocator installs a synchronized mapping-retirement callback. */
struct llama_htp_output_cache {
    std::array<ggml_context_ptr, 2> contexts;
    std::array<ggml_backend_buffer_ptr, 2> buffers;
    std::array<ggml_tensor *, 2> weights = {{ nullptr, nullptr }};
};
