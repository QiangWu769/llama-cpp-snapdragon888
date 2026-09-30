#!/bin/sh
set -eu
test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
llama_source=${LLAMA_SOURCE_DIR:-"$repo_dir/llama.cpp-npu"}
dsp_source=${HTP_SOURCE_DIR:-"$repo_dir/htp-ops-lib"}
test_output=$(mktemp -d "${TMPDIR:-/tmp}/htp-output-regressions.XXXXXX")
trap 'rm -rf "$test_output"' EXIT HUP INT TERM
compiler_c=${CC:-cc}
compiler_cpp=${CXX:-c++}

"$compiler_c" -std=c11 -O2 -Wall -Wextra -Werror -fsanitize=undefined -fno-sanitize-recover=all \
    -I"$llama_source/ggml/include" "$test_dir/packing_test.c" -o "$test_output/packing"
"$test_output/packing" --full-qwen

"$compiler_cpp" -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined -fno-sanitize-recover=all \
    -I"$test_dir/mock" -I"$dsp_source/include" \
    "$test_dir/dsp_mapping_test.cc" "$dsp_source/src/dsp/mmap_mgr.cc" -o "$test_output/dsp-mapping"
"$test_output/dsp-mapping"

"$compiler_cpp" -std=c++17 -Wall -Wextra -Werror -Wno-unused-variable -Wno-unused-label \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -I"$llama_source/ggml/include" -I"$llama_source/ggml/src" -I"$llama_source/ggml/src/ggml-htp" \
    "$test_dir/host_mapping_test.cc" "$llama_source/ggml/src/ggml-htp/rpcmem_mapper.cc" \
    -o "$test_output/host-mapping"
"$test_output/host-mapping"

# Optional tests execute the actual host GGML implementation. The required
# matching GGML libraries must already be built; no SDK or phone is involved.
if [ -n "${GGML_HOST_BUILD:-}" ]; then
    library_root="$GGML_HOST_BUILD/ggml/src"
    cpu_library_root="$library_root"
    if [ -d "$library_root/ggml-cpu" ]; then
        if [ -f "$library_root/ggml-cpu/libggml-cpu.so" ] || [ -f "$library_root/ggml-cpu/libggml-cpu.dylib" ]; then
            cpu_library_root="$library_root/ggml-cpu"
        fi
    fi
    "$compiler_cpp" -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined -fno-sanitize-recover=all \
        -I"$llama_source/ggml/include" "$test_dir/concat_test.cpp" \
        -L"$library_root" -L"$cpu_library_root" -lggml-cpu -lggml-base \
        -Wl,-rpath,"$library_root" -Wl,-rpath,"$cpu_library_root" -o "$test_output/concat"
    "$test_output/concat"
    "$compiler_cpp" -std=c++17 -O2 -Wall -Wextra -Werror -Wno-unused-variable -Wno-unused-label \
        -Wno-deprecated-declarations -fsanitize=undefined -fno-sanitize-recover=all \
        -I"$llama_source/ggml/include" -I"$llama_source/ggml/src" -I"$llama_source/ggml/src/ggml-htp" \
        "$test_dir/mapper_test.cpp" "$llama_source/ggml/src/ggml-htp/rpcmem_mapper.cc" \
        -L"$library_root" -lggml-base -Wl,-rpath,"$library_root" -o "$test_output/mapper-ggml"
    "$test_output/mapper-ggml"
else
    printf '%s\n' 'SKIP: actual GGML concat/mapper tests (set GGML_HOST_BUILD to a matching host build)'
fi
