#!/usr/bin/env bash
set -euo pipefail
test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$test_dir/../../.." && pwd)"
source_dir="${1:-$repo_dir/llama.cpp-npu}"
if [[ -n "${2:-}" ]]; then
    build_dir="$2"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/htp-split-regression.XXXXXX")"
    trap 'rm -rf -- "$build_dir"' EXIT
fi
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$build_dir" --target ggml-cpu -j 6
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations -fsanitize=undefined \
    -I"$source_dir/ggml/include" "$test_dir/support_test.cpp" \
    -L"$build_dir/ggml/src" -lggml-cpu -lggml-base \
    -Wl,-rpath,"$build_dir/ggml/src" -o "$build_dir/split-support-test"
"$build_dir/split-support-test"
