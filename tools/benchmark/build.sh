#!/usr/bin/env bash
# Build independent native C and safe Rust timing adapters.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
profile=${1:-float}
features=()
defines=()
case "$profile" in
    float) suffix= ;;
    fixed) suffix=-fixed; features=(--features fixed_point) ;;
    fixed-res24) suffix=-fixed-res24; features=(--features enable_res24) ;;
    pfa) suffix=-pfa; features=(--features pfa) ;;
    qext) suffix=-qext; features=(--features enable_qext); defines=(-DENABLE_QEXT) ;;
    fixed-qext) suffix=-fixed-qext; features=(--features fixed_point,enable_qext); defines=(-DENABLE_QEXT) ;;
    fixed-res24-qext) suffix=-fixed-res24-qext; features=(--features enable_res24,enable_qext); defines=(-DENABLE_QEXT) ;;
    *) printf '%s\n' 'Supported profiles: float, fixed, fixed-res24, pfa, qext, fixed-qext, fixed-res24-qext.' >&2; exit 2 ;;
esac
reference="$root/target/reference$suffix/source"
if [[ ! -f "$reference/libopus.a" ]]; then bash tools/reference/build.sh "$profile"; fi
test "$(cat "$reference/.reference-revision")" = 503d81b138d76621aae4b12786e90de48aa8db3a
output="$root/target/benchmark/binaries"
mkdir -p "$output"
"${CC:-cc}" -O2 -std=c99 -ffp-contract=off -fno-fast-math -DOPUS_BUILD "${defines[@]}" \
    -I"$reference/include" -I"$reference/celt" -I"$reference/src" -I"$reference/silk" \
    tools/benchmark/codec_bench.c "$reference/libopus.a" -lm -o "$output/c-$profile"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$root/target/benchmark/build"}
export CARGO_INCREMENTAL=0
cargo build --locked --release "${features[@]}" --bin opus-rs-bench
cp "$CARGO_TARGET_DIR/release/opus-rs-bench" "$output/rust-$profile"
printf 'C: %s\nRust: %s\n' "$output/c-$profile" "$output/rust-$profile"
