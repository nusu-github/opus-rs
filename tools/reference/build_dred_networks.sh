#!/usr/bin/env bash
# Emit independent RDOVAE fixtures using verified generated model exports.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source "$root/tools/reference/dnn_profile.sh"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
source_dir="$root/target/reference/source"
if [[ ! -f "$source_dir/libopus.a" ]]; then bash tools/reference/build.sh; fi
cc "${dnn_flags[@]}" -O2 -std=c99 -DOPUS_BUILD -DVAR_ARRAYS -U__SSE2__ -U__AVX__ \
  -DSUPPRESS_PERF_WARNINGS -ffp-contract=off -fno-fast-math -fno-tree-vectorize \
  -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$source_dir" -I"$source_dir/include" -I"$source_dir/celt" -I"$source_dir/dnn" \
  -I"$OSCE_MODEL_DIRECTORY" tools/reference/dred_network_vectors.c \
  "$source_dir/dnn/dred_rdovae_enc.c" "$source_dir/dnn/dred_rdovae_dec.c" \
  "$source_dir/dnn/nnet.c" "$source_dir/dnn/nnet_default.c" \
  "$source_dir/dnn/parse_lpcnet_weights.c" \
  "$OSCE_MODEL_DIRECTORY/dred_rdovae_enc_data.c" "$OSCE_MODEL_DIRECTORY/dred_rdovae_dec_data.c" \
  "$source_dir/libopus.a" -lm -o "target/reference/dred-networks-reference$dnn_suffix"
"./target/reference/dred-networks-reference$dnn_suffix"
