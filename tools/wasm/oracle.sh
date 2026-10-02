#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
target_dir=${CARGO_TARGET_DIR:-target}
case "$target_dir" in
    /*) ;;
    *) target_dir="$root/$target_dir" ;;
esac
module=${OPUS_WASM_ORACLE:-$target_dir/wasm32-wasip1/release/opus-rs-oracle.wasm}
exec node --no-warnings "$root/tools/wasm/run.mjs" "$module" "$@"
