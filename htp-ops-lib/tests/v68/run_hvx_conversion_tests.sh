#!/bin/sh
set -eu
conversion_test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
conversion_repo_dir=$(CDPATH= cd -- "$conversion_test_dir/../.." && pwd)
conversion_header=${CONVERSION_HEADER:-"$conversion_repo_dir/include/dsp/hvx_convert.h"}
conversion_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/htp-v68-fp-bits.XXXXXX")
trap 'rm -rf "$conversion_build_dir"' EXIT HUP INT TERM
# Isolate the actual integer helpers from their SDK-only HVX wrappers. This
# tests the production bit arithmetic; vector lane layout still needs the DSP.
python3 - "$conversion_header" "$conversion_build_dir/v68_fp_bits.h" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
start = source.index('static HVX_INLINE_ALWAYS uint32_t hvx_v68_hf_bits_to_sf')
end = source.index('\n#endif', start)
Path(sys.argv[2]).write_text(source[start:end].replace('HVX_INLINE_ALWAYS', 'inline'))
PY
"${CC:-clang}" -std=c11 -O2 -Wall -Wextra -Werror \
  -fsanitize="${SANITIZERS:-undefined}" -fno-sanitize-recover=all \
  -I "$conversion_build_dir" "$conversion_test_dir/hvx_conversion_test.c" \
  -lm -o "$conversion_build_dir/conversion_test"
"$conversion_build_dir/conversion_test"

"${CC:-clang}" -std=c11 -O2 -Wall -Wextra -Werror \
  -fsanitize="${SANITIZERS:-undefined}" -fno-sanitize-recover=all \
  "-DTEST_HVX_HEADER=\"$conversion_header\"" -I "$conversion_repo_dir/include" \
  "$conversion_test_dir/hvx_conversion_vector_test.c" \
  -lm -o "$conversion_build_dir/vector_conversion_test"
"$conversion_build_dir/vector_conversion_test"
