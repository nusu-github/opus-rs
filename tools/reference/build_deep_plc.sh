#!/usr/bin/env bash
# Emit independent analysis and neural PLC fixtures with the current pinned model.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source "$root/tools/reference/dnn_profile.sh"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
args=()
for name in lpcnet_plc lpcnet_enc pitchdnn fargan freq lpcnet_tables burg nnet nnet_default parse_lpcnet_weights; do args+=("$source_dir/dnn/$name.c"); done
for name in plc pitchdnn fargan; do args+=("$OSCE_MODEL_DIRECTORY/${name}_data.c"); done
cc "${dnn_flags[@]}" -O2 -std=c99 -DOPUS_BUILD -DVAR_ARRAYS -DENABLE_DEEP_PLC -DHAVE_LRINTF \
  -U__SSE2__ -U__AVX__ -DSUPPRESS_PERF_WARNINGS -ffp-contract=off \
  -fno-fast-math -fno-tree-vectorize -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$source_dir" -I"$source_dir/include" -I"$source_dir/celt" -I"$source_dir/silk" \
  -I"$source_dir/dnn" -I"$OSCE_MODEL_DIRECTORY" \
  tools/reference/deep_plc_vectors.c "${args[@]}" \
  "$source_dir/libopus.a" -lm -o "target/reference/deep-plc-reference$dnn_suffix"
"./target/reference/deep-plc-reference$dnn_suffix"
