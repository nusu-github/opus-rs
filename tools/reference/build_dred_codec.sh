#!/usr/bin/env bash
# Build a scalar DRED and Deep PLC profile with verified model parameters.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
source "$root/tools/reference/dnn_profile.sh"
: "${OSCE_MODEL_DIRECTORY:?Set OSCE_MODEL_DIRECTORY to the verified generated dnn model directory}"
weights=$(cd "$OSCE_MODEL_DIRECTORY" && pwd)
qext_flags=
qext_suffix=
case ${OPUS_REFERENCE_QEXT:-0} in
  0) ;;
  1) qext_flags=-DENABLE_QEXT; qext_suffix=-qext ;;
  *) printf '%s\n' 'OPUS_REFERENCE_QEXT must be 0 or 1.' >&2; exit 2 ;;
esac
build="$root/target/reference-dred$dnn_suffix$qext_suffix"
revision=503d81b138d76621aae4b12786e90de48aa8db3a
mkdir -p "$build/source"
if [[ ! -e "$build/source/.reference-revision" ]]; then
    git archive "$revision" | tar -x -C "$build/source"
    printf '%s\n' "$revision" > "$build/source/.reference-revision"
fi
flags="${dnn_flags[*]} $qext_flags -O2 -std=c99 -DVAR_ARRAYS -DOPUS_BUILD -DHAVE_LRINTF -DENABLE_DRED -DENABLE_DEEP_PLC -U__SSE2__ -U__AVX__ -ffp-contract=off -fno-fast-math -fno-tree-vectorize -I. -Iinclude -Isilk -Icelt -Isilk/float -Idnn -I$weights"
make -s -C "$build/source" -f Makefile.unix clean > /dev/null 2>&1
make -s -C "$build/source" -f Makefile.unix lib "CFLAGS=$flags" -j "${OPUS_REFERENCE_JOBS:-2}" > "$build/build.log" 2>&1
(
  cd "$build/source"
  for name in nnet nnet_default parse_lpcnet_weights lpcnet_enc lpcnet_plc pitchdnn fargan freq lpcnet_tables burg dred_rdovae_enc dred_rdovae_dec dred_encoder dred_decoder dred_coding; do
    cc $flags -c "dnn/$name.c" -o "dnn/$name.o"
    ar r libopus.a "dnn/$name.o"
  done
  for name in plc pitchdnn fargan dred_rdovae_enc dred_rdovae_dec dred_rdovae_stats; do
    cc $flags -c "$weights/${name}_data.c" -o "dnn/${name}_data.o"
    ar r libopus.a "dnn/${name}_data.o"
  done
)
cc $flags -I"$build/source/src" -I"$build/source/dnn" -I"$build/source/include" -I"$build/source/celt" -I"$build/source/silk" tools/reference/oracle.c "$build/source/libopus.a" -lm -o "$build/opus-reference"
printf '%s\n' "$build/opus-reference"
