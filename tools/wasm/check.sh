#!/bin/sh
# Compare the same Rust codec on native and WebAssembly targets.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
rust_host=$(rustc -vV | sed -n 's/^host: //p')
features=${OPUS_RUST_FEATURES:-}
target_dir=${CARGO_TARGET_DIR:-target}
case "$target_dir" in
    /*) ;;
    *) target_dir="$root/$target_dir" ;;
esac
export OPUS_WASM_ORACLE="$target_dir/wasm32-wasip1/release/opus-rs-oracle.wasm"
case ${OPUS_WASM_COMPARISON:-codec} in
    codec) comparison=tools/reference/compare_codec.py ;;
    qext) comparison=tools/reference/compare_qext.py ;;
    qext_speech) comparison=tools/reference/compare_qext_speech.py ;;
    decode) comparison=tools/reference/compare_decode.py ;;
    sequence|sequences) comparison=tools/reference/compare_sequences.py ;;
    *) printf '%s\n' 'OPUS_WASM_COMPARISON must be codec, qext, qext_speech, decode, or sequences.' >&2; exit 2 ;;
esac
if [ -n "$features" ]; then
    cargo check --locked --lib --target wasm32-unknown-unknown --features "$features"
    cargo build --locked --release --bin opus-rs-oracle --features "$features" \
        --target "$rust_host" --target wasm32-wasip1
else
    cargo check --locked --lib --target wasm32-unknown-unknown
    cargo build --locked --release --bin opus-rs-oracle \
        --target "$rust_host" --target wasm32-wasip1
fi
python3 "$comparison" \
    --reference "$target_dir/$rust_host/release/opus-rs-oracle" \
    --candidate tools/wasm/oracle.sh "$@"
