#!/bin/sh
set -eu

attention_test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
attention_repo_dir=$(CDPATH= cd -- "$attention_test_dir/../.." && pwd)
attention_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-attention-tests.XXXXXX")
trap 'rm -rf "$attention_build_dir"' EXIT HUP INT TERM

attention_cc=${CC:-clang}
attention_sanitizers=${SANITIZERS:-undefined}
"$attention_cc" -std=c11 -O2 -g -Wall -Wextra -Werror \
  -fsanitize="$attention_sanitizers" -fno-sanitize-recover=all \
  -I "$attention_repo_dir/include" \
  "$attention_repo_dir/src/dsp/ops/flash_attn_v68.c" \
  "$attention_test_dir/attention_portable_test.c" \
  -lm -o "$attention_build_dir/attention_portable_test"
"$attention_build_dir/attention_portable_test"
