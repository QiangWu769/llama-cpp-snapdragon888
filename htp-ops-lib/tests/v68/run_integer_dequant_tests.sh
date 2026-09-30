#!/bin/sh
set -eu
dequant_test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dequant_repo_dir=$(CDPATH= cd -- "$dequant_test_dir/../.." && pwd)
dequant_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-v68-dequant.XXXXXX")
trap 'rm -rf "$dequant_build_dir"' EXIT HUP INT TERM

# Test the production integer arithmetic, without including SDK-only DSP code.
python3 - "$dequant_repo_dir/src/dsp/ops/mat_mul.c" "$dequant_build_dir/integer_dequant_under_test.h" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index('static inline HVX_Vector matmul_v68_integer_product_bits')
end = source.index('\n#endif', start)
Path(sys.argv[2]).write_text(source[start:end])
PY

"${CC:-clang}" -std=c11 -O2 -Wall -Wextra -Werror \
  -fsanitize="${SANITIZERS:-undefined}" -fno-sanitize-recover=all \
  -I "$dequant_build_dir" "$dequant_test_dir/integer_dequant_test.c" \
  -o "$dequant_build_dir/integer_dequant_test"
"$dequant_build_dir/integer_dequant_test"
