#!/usr/bin/env bash
# Emit independent scalar feature-extraction fixtures from the pinned C checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
cc -O2 -std=c99 -DOPUS_BUILD -DVAR_ARRAYS -DENABLE_OSCE -DENABLE_OSCE_BWE \
  -U__SSE2__ -U__AVX__ -DSUPPRESS_PERF_WARNINGS -ffp-contract=off \
  -fno-fast-math -fno-tree-vectorize -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$source_dir" -I"$source_dir/include" -I"$source_dir/celt" -I"$source_dir/silk" \
  -I"$source_dir/dnn" -I"$OSCE_MODEL_DIRECTORY" \
  tools/reference/osce_features_vectors.c "$source_dir/dnn/osce_features.c" \
  "$source_dir/dnn/freq.c" "$source_dir/dnn/lpcnet_tables.c" \
  "$source_dir/libopus.a" -lm -o target/reference/osce-features-reference
./target/reference/osce-features-reference
