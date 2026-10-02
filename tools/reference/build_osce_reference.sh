#!/usr/bin/env bash
# Build the independent scalar OSCE component oracle from the pinned C checkout.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source "$root/tools/reference/dnn_profile.sh"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
cc "${dnn_flags[@]}" -O2 -std=c99 -U__SSE2__ -U__AVX__ -ffp-contract=off -fno-fast-math -fno-tree-vectorize \
  -ffunction-sections -fdata-sections -Wl,--gc-sections -DVAR_ARRAYS -DOPUS_BUILD \
  -DENABLE_OSCE -DENABLE_OSCE_BWE -DHAVE_LRINTF \
  -I"$source_dir/include" -I"$source_dir" -I"$source_dir/celt" -I"$source_dir/silk" \
  -I"$source_dir/dnn" -I"$OSCE_MODEL_DIRECTORY" \
  tools/reference/osce_vectors.c "$source_dir/dnn/nndsp.c" "$source_dir/dnn/nnet.c" \
  "$source_dir/dnn/nnet_default.c" "$source_dir/dnn/parse_lpcnet_weights.c" \
  "$OSCE_MODEL_DIRECTORY/lace_data.c" "$OSCE_MODEL_DIRECTORY/nolace_data.c" \
  "$OSCE_MODEL_DIRECTORY/bbwenet_data.c" "$source_dir/libopus.a" -lm \
  -o "target/reference/osce-reference$dnn_suffix"
"./target/reference/osce-reference$dnn_suffix"
