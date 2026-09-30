#!/bin/sh
set -eu
attention_test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
attention_repo_dir=$(CDPATH= cd -- "$attention_test_dir/../.." && pwd)
attention_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-attention-hmx-tests.XXXXXX")
trap 'rm -rf "$attention_build_dir"' EXIT HUP INT TERM
attention_cc=${CC:-clang}
attention_sanitizers=${SANITIZERS:-undefined}
"$attention_cc" -std=c11 -O2 -g -Wall -Wextra -Werror \
  -fsanitize="$attention_sanitizers" -fno-sanitize-recover=all \
  -DHTP_HMX_V68=1 -DFA_V68_HMX_PORTABLE_TEST=1 -DATTENTION_FP16_TILES=1 \
  -I "$attention_repo_dir/include" \
  "$attention_repo_dir/src/dsp/ops/flash_attn.c" \
  "$attention_test_dir/attention_portable_test.c" \
  -lm -o "$attention_build_dir/attention_hmx_test"
"$attention_build_dir/attention_hmx_test"
