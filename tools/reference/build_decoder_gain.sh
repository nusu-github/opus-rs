#!/bin/sh
# Generate independent integer decoder gain vectors for both PCM widths.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
for profile in fixed fixed-res24; do
    reference="$root/target/reference-$profile"
    if [ ! -f "$reference/source/libopus.a" ]; then
        bash "$root/tools/reference/build.sh" "$profile"
    fi
    extra_flags=
    if [ "$profile" = fixed-res24 ]; then extra_flags=-DENABLE_RES24; fi
    "${CC:-cc}" -O2 -std=c99 -DVAR_ARRAYS -DOPUS_BUILD -DFIXED_POINT $extra_flags \
        -ffp-contract=off -fno-fast-math \
        -I"$reference/source/include" -I"$reference/source/celt" \
        "$root/tools/reference/decoder_gain_vectors.c" "$reference/source/libopus.a" -lm \
        -o "$reference/decoder-gain-vectors"
    "$reference/decoder-gain-vectors" > "$root/rust/tests/fixtures/reference/$profile-decoder-gain.tsv"
done
