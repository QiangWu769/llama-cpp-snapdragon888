#!/bin/sh
set -eu
probe_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
calibration_test_directory=$(mktemp -d)
trap 'rm -rf "$calibration_test_directory"' EXIT HUP INT TERM
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -fsanitize=undefined \
  -I"$probe_directory" "$probe_directory/host_oracle_test.c" \
  -o "$calibration_test_directory/oracle"
"$calibration_test_directory/oracle"
