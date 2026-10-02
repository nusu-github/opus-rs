#!/bin/sh
# Regenerate the fixed-point QEXT energy-math vectors from the pinned C source.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
reference="$root/target/reference-fixed-qext"
output=${1:-"$root/rust/tests/fixtures/reference/fixed-qext-energy-math.bin"}
if [ ! -f "$reference/source/libopus.a" ]; then
    "$root/tools/reference/build.sh" fixed-qext
fi
"${CC:-cc}" -O2 -std=c99 -DVAR_ARRAYS -DOPUS_BUILD -DFIXED_POINT -DENABLE_QEXT \
    -I"$reference/source/include" -I"$reference/source/celt" \
    "$root/tools/reference/qext_energy_math.c" "$reference/source/libopus.a" -lm \
    -o "$reference/qext-energy-math"
"$reference/qext-energy-math" > "$output"
