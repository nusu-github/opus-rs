#!/bin/sh
# Build the test-only scalar C oracle from an immutable Git object.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
revision=503d81b138d76621aae4b12786e90de48aa8db3a
profile=${1:-float}
case "$profile" in
    float) suffix=; fixed_point= ;;
    fixed) suffix=-fixed; fixed_point=1 ;;
    fixed-res24) suffix=-fixed-res24; fixed_point=1 ;;
    pfa) suffix=-pfa; fixed_point= ;;
    fixed-pfa|fixed-res24-pfa) suffix=-$profile; fixed_point=1 ;;
    qext|qext-pfa) suffix=-$profile; fixed_point= ;;
    fixed-qext|fixed-qext-pfa|fixed-res24-qext|fixed-res24-qext-pfa) suffix=-$profile; fixed_point=1 ;;
    *) printf '%s\n' 'Usage: build.sh [float|fixed|fixed-res24|pfa|fixed-pfa|fixed-res24-pfa|qext|qext-pfa|fixed-qext|fixed-qext-pfa|fixed-res24-qext|fixed-res24-qext-pfa]' >&2; exit 2 ;;
esac
build=${OPUS_REFERENCE_BUILD_DIR:-"$root/target/reference$suffix"}
mkdir -p "$build/source"
if [ ! -f "$build/source/.reference-revision" ]; then
    git -C "$root" archive "$revision" | tar -x -C "$build/source"
    printf '%s\n' "$revision" > "$build/source/.reference-revision"
fi
if [ "$(cat "$build/source/.reference-revision")" != "$revision" ]; then
    printf '%s\n' 'Reference build directory contains a different revision.' >&2
    exit 1
fi
flags='-O2 -std=c99 -DVAR_ARRAYS -DOPUS_BUILD -DHAVE_LRINTF -ffp-contract=off -fno-fast-math -fno-tree-vectorize -Iinclude -Isilk -Icelt -Isilk/float'
if [ -n "$fixed_point" ]; then
    flags="$flags -DFIXED_POINT -Isilk/fixed"
    case "$profile" in fixed-res24*) flags="$flags -DENABLE_RES24" ;; esac
fi
case "$profile" in
    *pfa) flags="$flags -DENABLE_PFA" ;;
esac
oracle_flags=
case "$profile" in
    *qext*) flags="$flags -DENABLE_QEXT"; oracle_flags=-DENABLE_QEXT ;;
esac
make -s -C "$build/source" -f Makefile.unix lib "CC=${CC:-cc}" "CFLAGS=$flags" "FIXED_POINT=$fixed_point" -j "${OPUS_REFERENCE_JOBS:-2}" > "$build/build.log" 2>&1 || {
    cat "$build/build.log" >&2
    exit 1
}
if [ -n "$fixed_point" ]; then
    # Makefile.unix omits analysis when FIXED_POINT is set because its default
    # disables the float API. This oracle retains that API for cross-checking.
    (
        cd "$build/source"
        for unit in analysis mlp mlp_data; do
            "${CC:-cc}" $flags -c "src/$unit.c" -o "src/$unit.o"
            ar r libopus.a "src/$unit.o"
        done
    )
fi
"${CC:-cc}" -O2 -std=c99 -ffp-contract=off -fno-fast-math -DOPUS_BUILD $oracle_flags \
    -I "$build/source/include" -I "$build/source/celt" -I "$build/source/src" \
    -I "$build/source/silk" "$root/tools/reference/oracle.c" \
    "$build/source/libopus.a" -lm -o "$build/opus-reference"
printf '%s\n' "$build/opus-reference"
