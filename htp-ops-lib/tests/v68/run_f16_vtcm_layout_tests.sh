#!/bin/sh
set -eu
layout_test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
layout_repo_dir=$(CDPATH= cd -- "$layout_test_dir/../.." && pwd)
layout_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-f16-vtcm-tests.XXXXXX")
trap 'rm -rf "$layout_build_dir"' EXIT HUP INT TERM
"${CC:-clang}" -std=c11 -O1 -g -Wall -Wextra -Werror \
  -fsanitize="${SANITIZERS:-address,undefined}" -fno-sanitize-recover=all \
  -I "$layout_repo_dir/include" "$layout_test_dir/f16_vtcm_layout_test.c" \
  -o "$layout_build_dir/f16_vtcm_layout_test"
"$layout_build_dir/f16_vtcm_layout_test"
