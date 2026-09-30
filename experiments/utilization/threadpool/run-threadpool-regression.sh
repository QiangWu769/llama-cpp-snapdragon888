#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "${experiment_dir}/../../.." && pwd)"
source_dir="${1:-${repository_dir}/llama.cpp-npu}"
build_dir="${2:-${experiment_dir}/build-regression}"

# Build the ordinary CPU dependencies, but compile only the hybrid executor
# for this test. Its DSP operation entry points are replaced by test stubs.
cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "${build_dir}" --target ggml-cpu -j 6

"${CC:-cc}" -std=gnu11 -O2 -D_GNU_SOURCE -D_DARWIN_C_SOURCE \
    -I"${source_dir}/ggml/include" -I"${source_dir}/ggml/src" \
    -c "${source_dir}/ggml/src/ggml-htp/htp-cpu-impl.c" \
    -o "${build_dir}/htp-cpu-regression.o"
"${CXX:-c++}" -std=c++17 -O2 -pthread \
    -I"${source_dir}/ggml/include" -I"${source_dir}/ggml/src" \
    -I"${source_dir}/ggml/src/ggml-htp" \
    "${experiment_dir}/htp-threadpool-regression.cpp" \
    "${build_dir}/htp-cpu-regression.o" \
    -L"${build_dir}/ggml/src" -lggml-cpu -lggml-base \
    -Wl,-rpath,"${build_dir}/ggml/src" \
    -o "${build_dir}/htp-threadpool-regression"
"${build_dir}/htp-threadpool-regression"
