#!/usr/bin/env bash
# Emit current DRED packet fixtures without using the Rust implementation.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source "$root/tools/reference/dnn_profile.sh"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
cc "${dnn_flags[@]}" -O2 -std=c99 -DOPUS_BUILD -DVAR_ARRAYS -U__SSE2__ -U__AVX__ \
  -ffp-contract=off -fno-fast-math -fno-tree-vectorize \
  -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$source_dir" -I"$source_dir/include" -I"$source_dir/celt" -I"$source_dir/dnn" \
  -I"$OSCE_MODEL_DIRECTORY" tools/reference/dred_packet_vectors.c \
  "$source_dir/dnn/dred_encoder.c" "$source_dir/dnn/dred_decoder.c" "$source_dir/dnn/dred_coding.c" \
  "$source_dir/dnn/nnet.c" "$source_dir/dnn/nnet_default.c" \
  "$OSCE_MODEL_DIRECTORY/dred_rdovae_stats_data.c" \
  "$source_dir/libopus.a" -lm -o "target/reference/dred-packets-reference$dnn_suffix"
"./target/reference/dred-packets-reference$dnn_suffix"
