#!/usr/bin/env bash
# Emit independent Burg cepstral feature vectors from the pinned scalar C code.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
cc -O2 -std=c99 -DOPUS_BUILD -DVAR_ARRAYS -DHAVE_LRINTF \
  -U__SSE2__ -U__AVX__ -ffp-contract=off -fno-fast-math -fno-tree-vectorize \
  -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$source_dir" -I"$source_dir/include" -I"$source_dir/celt" \
  -I"$source_dir/silk" -I"$source_dir/dnn" \
  tools/reference/burg_cepstrum_vectors.c "$source_dir/dnn/freq.c" \
  "$source_dir/dnn/burg.c" "$source_dir/dnn/lpcnet_tables.c" \
  "$source_dir/libopus.a" -lm -o target/reference/burg-reference
./target/reference/burg-reference
