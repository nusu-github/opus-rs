#!/usr/bin/env bash
# Compare optional neural models with scalar C and the same Rust build on WASI.
# Verified models must be supplied through DNN_WEIGHTS_PATH and DRED_WEIGHTS_PATH.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
profile=${1:-quantized}
component=${OPUS_NEURAL_COMPONENT:-all}
case "$component" in all|deep-plc|dred|osce) ;; *) echo 'OPUS_NEURAL_COMPONENT must be all, deep-plc, dred, or osce.' >&2; exit 2 ;; esac
case "$profile" in
    quantized) extra_features=; reference_suffix=-quantized; osce_weights="$root/target/reference/osce-weights-quantized.bin" ;;
    debug-float) extra_features=,dnn_debug_float; reference_suffix=; osce_weights="$root/target/reference/osce-weights.bin" ;;
    *) echo 'Usage: check_neural.sh [quantized|debug-float]' >&2; exit 2 ;;
esac
rust_host=$(rustc -vV | sed -n 's/^host: //p')
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$root/target/neural-wasm"}
case "$CARGO_TARGET_DIR" in /*) ;; *) CARGO_TARGET_DIR="$root/$CARGO_TARGET_DIR" ;; esac
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
artifact_dir=${OPUS_NEURAL_ARTIFACT_DIR:-"$root/target/neural-wasm-binaries"}
output_dir=${OPUS_NEURAL_OUTPUT_DIR:-"$root/target/neural-wasm-$profile"}
mkdir -p "$artifact_dir" "$output_dir"
unset OPUS_ORACLE_DNN_BLOB

build_profile() {
    local name=$1 features=$2
    native="$artifact_dir/$name-$profile"
    module="$artifact_dir/$name-$profile.wasm"
    if [[ ${OPUS_NEURAL_SKIP_BUILD:-0} != 1 ]]; then
        cargo check --locked --release --lib --target wasm32-unknown-unknown --features "$features" \
            > "$output_dir/$name-unknown-check.log" 2>&1
        cargo build --locked --release --bin opus-rs-oracle --features "$features" \
            --target "$rust_host" --target wasm32-wasip1 \
            > "$output_dir/$name-build.log" 2>&1
        cp "$CARGO_TARGET_DIR/$rust_host/release/opus-rs-oracle" "$native"
        cp "$CARGO_TARGET_DIR/wasm32-wasip1/release/opus-rs-oracle.wasm" "$module"
    fi
    test -x "$native"
    test -f "$module"
    export OPUS_WASM_ORACLE="$module"
}

if [[ "$component" != osce ]]; then
    : "${DNN_WEIGHTS_PATH:?Provide the verified model directory or tarball}"
    export DRED_WEIGHTS_PATH=${DRED_WEIGHTS_PATH:-"$DNN_WEIGHTS_PATH"}
    build_profile deep-plc "deep_plc_weights$extra_features"
    reference="$root/target/reference-dred$reference_suffix/opus-reference"
    test -x "$reference"
    if [[ "$component" == all || "$component" == deep-plc ]]; then
        decode_options=(--quick)
        if [[ ${OPUS_NEURAL_FULL_DECODE:-0} == 1 ]]; then decode_options=(); fi
        OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_decode.py \
            --reference "$reference" --candidate "$native" --loss "${decode_options[@]}" \
            | tee "$output_dir/deep-plc-native.log"
        OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_sequences.py \
            --reference "$reference" --candidate "$native" --decode-only \
            | tee "$output_dir/deep-plc-sequences-native.log"
        OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_decode.py \
            --reference "$native" --candidate tools/wasm/oracle.sh --loss "${decode_options[@]}" \
            | tee "$output_dir/deep-plc-wasi.log"
        OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_sequences.py \
            --reference "$native" --candidate tools/wasm/oracle.sh --decode-only \
            | tee "$output_dir/deep-plc-sequences-wasi.log"
    fi
    if [[ "$component" == all || "$component" == dred ]]; then
        python3 tools/reference/compare_dred.py --reference "$reference" --candidate "$native" \
            | tee "$output_dir/dred-native.log"
        python3 tools/reference/compare_dred.py --reference "$native" --candidate tools/wasm/oracle.sh \
            | tee "$output_dir/dred-wasi.log"
    fi
fi

if [[ "$component" == all || "$component" == osce ]]; then
    build_profile osce "osce$extra_features"
    reference="$root/target/reference-osce$reference_suffix/opus-reference"
    test -x "$reference"
    osce_weights=${OPUS_OSCE_WEIGHTS:-"$osce_weights"}
    python3 tools/reference/compare_osce.py --reference "$reference" --candidate "$native" \
        --weights "$osce_weights" | tee "$output_dir/osce-native.log"
    python3 tools/reference/compare_osce.py --reference "$native" --candidate tools/wasm/oracle.sh \
        --weights "$osce_weights" --reference-weights "$osce_weights" \
        | tee "$output_dir/osce-wasi.log"
fi
