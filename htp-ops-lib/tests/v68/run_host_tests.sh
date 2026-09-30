#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-v68-host-test.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
"${CC:-clang}" -std=c11 -O1 -g -Wall -Wextra -Werror \
    -fsanitize=undefined -fno-omit-frame-pointer \
    -I"$repo_dir/tests/v68/host_shim" -I"$repo_dir/include" \
    "$repo_dir/src/dsp/ops/mat_mul_v68.c" \
    "$repo_dir/tests/v68/matmul_host_test.c" \
    -lm -o "$build_dir/matmul_host_test"
"$build_dir/matmul_host_test"
