#!/bin/sh
# Build the pinned scalar C reference with dynamic Opus Custom modes enabled.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
revision=503d81b138d76621aae4b12786e90de48aa8db3a
profile=${1:-float}
base_profile=${profile%-pfa}
base_profile=${base_profile%-qext}
case "$base_profile" in
    float) fixed_point= ;;
    fixed|fixed24) fixed_point=1 ;;
    *) printf '%s\n' 'Usage: build_custom.sh float|fixed|fixed24[-qext][-pfa]' >&2; exit 2 ;;
esac
suffix=-$profile
if [ "$profile" = float ]; then suffix=; fi
build="$root/target/reference-custom$suffix"
mkdir -p "$build/source"
if [ ! -f "$build/source/.reference-revision" ]; then
    git -C "$root" archive "$revision" | tar -x -C "$build/source"
    printf '%s\n' "$revision" > "$build/source/.reference-revision"
fi
test "$(cat "$build/source/.reference-revision")" = "$revision"
flags='-O2 -std=c99 -DVAR_ARRAYS -DOPUS_BUILD -DHAVE_LRINTF -DCUSTOM_MODES -DENABLE_OPUS_CUSTOM_API -ffp-contract=off -fno-fast-math -fno-tree-vectorize -Iinclude -Isilk -Icelt -Isilk/float'
if [ "$base_profile" != float ]; then flags="$flags -DFIXED_POINT -Isilk/fixed"; fi
if [ "$base_profile" = fixed24 ]; then flags="$flags -DENABLE_RES24"; fi
case "$profile" in *-pfa) flags="$flags -DENABLE_PFA" ;; esac
case "$profile" in *-qext*) flags="$flags -DENABLE_QEXT" ;; esac
make -s -C "$build/source" -f Makefile.unix lib "CC=${CC:-cc}" "CFLAGS=$flags" "FIXED_POINT=$fixed_point" -j "${OPUS_REFERENCE_JOBS:-2}" > "$build/build.log" 2>&1 || {
    cat "$build/build.log" >&2
    exit 1
}
printf '%s\n' "$build/source/libopus.a"
