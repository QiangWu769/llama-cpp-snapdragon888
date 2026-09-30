#!/usr/bin/env bash
set -euo pipefail
review_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$review_dir/../../.." && pwd)"
source_dir="${1:-$repo_dir/llama.cpp-npu}"
if [[ -n "${2:-}" ]]; then
    build_dir="$2"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/htp-concat-regression.XXXXXX")"
    trap 'rm -rf -- "$build_dir"' EXIT
fi
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS= -DCMAKE_CXX_FLAGS= \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF -DGGML_METAL=OFF \
    -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$build_dir" --target ggml-cpu -j 6
# Instrument both actual executor translation units. Keep GGML base as a
# normal dependency: its graph-size helper intentionally offsets a NULL pointer,
# which full-library UBSan rejects before the concat test can run.
"${CC:-cc}" -std=gnu11 -O2 -D_GNU_SOURCE -D_DARWIN_C_SOURCE -fsanitize=undefined \
    -fno-sanitize-recover=undefined -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" \
    -I"$source_dir/ggml/src/ggml-cpu" -c "$source_dir/ggml/src/ggml-cpu/ggml-cpu.c" \
    -o "$build_dir/concat-standard.o"
"${CC:-cc}" -std=gnu11 -O2 -D_GNU_SOURCE -D_DARWIN_C_SOURCE -fsanitize=undefined \
    -fno-sanitize-recover=undefined -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" \
    -Dggml_get_type_traits_cpu=concat_test_hybrid_type_traits \
    -Dggml_threadpool_free=concat_test_hybrid_pool_free \
    -Dggml_threadpool_pause=concat_test_hybrid_pool_pause \
    -Dggml_threadpool_resume=concat_test_hybrid_pool_resume \
    -Dggml_arm_arch_features=concat_test_hybrid_arm_features \
    -c "$source_dir/ggml/src/ggml-htp/htp-cpu-impl.c" -o "$build_dir/concat-hybrid.o"
"${CXX:-c++}" -std=c++17 -O2 -pthread -fsanitize=undefined -fno-sanitize-recover=undefined \
    -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-htp" \
    "$review_dir/concat_test.cpp" "$build_dir/concat-standard.o" "$build_dir/concat-hybrid.o" \
    -L"$build_dir/ggml/src" -lggml-cpu -lggml-base \
    -Wl,-rpath,"$build_dir/ggml/src" -o "$build_dir/concat-test"
"$build_dir/concat-test"

"${CXX:-c++}" -std=c++17 -O2 -pthread -Wno-deprecated-declarations \
    -fsanitize=undefined -fno-sanitize-recover=undefined -I"$source_dir/ggml/include" \
    "$review_dir/support_test.cpp" -L"$build_dir/ggml/src" -lggml-cpu -lggml-base \
    -Wl,-rpath,"$build_dir/ggml/src" -o "$build_dir/concat-support-test"
"$build_dir/concat-support-test"
