#!/usr/bin/env bash
# Compare 96 kHz neural/QEXT combinations on the native host and WASI.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
profile=${1:-quantized}
case "$profile" in
    quantized) extra_features=; reference_suffix=-quantized; osce_weights="$root/target/reference/osce-weights-quantized.bin" ;;
    debug-float) extra_features=,dnn_debug_float; reference_suffix=; osce_weights="$root/target/reference/osce-weights.bin" ;;
    *) echo 'Usage: check_neural_qext.sh [quantized|debug-float]' >&2; exit 2 ;;
esac
: "${DNN_WEIGHTS_PATH:?Provide the verified model directory or tarball}"
export DRED_WEIGHTS_PATH=${DRED_WEIGHTS_PATH:-"$DNN_WEIGHTS_PATH"}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$root/target/neural-wasm"}
case "$CARGO_TARGET_DIR" in /*) ;; *) CARGO_TARGET_DIR="$root/$CARGO_TARGET_DIR" ;; esac
export CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0
artifact_dir=${OPUS_NEURAL_ARTIFACT_DIR:-"$root/target/neural-wasm-binaries"}
case "$artifact_dir" in /*) ;; *) artifact_dir="$root/$artifact_dir" ;; esac
output_dir=${OPUS_NEURAL_OUTPUT_DIR:-"$root/target/neural-wasm-$profile"}
mkdir -p "$artifact_dir" "$output_dir"
rust_host=$(rustc -vV | sed -n 's/^host: //p')
unset OPUS_ORACLE_DNN_BLOB

for name in deep-plc osce; do
    features="osce,enable_qext$extra_features"
    if [[ "$name" == deep-plc ]]; then features="deep_plc_weights,enable_qext$extra_features"; fi
    native="$artifact_dir/qext-$name-$profile"
    module="$native.wasm"
    if [[ ${OPUS_NEURAL_SKIP_BUILD:-0} != 1 ]]; then
        cargo check --locked --release --lib --target wasm32-unknown-unknown --features "$features" \
            > "$output_dir/qext-$name-unknown-check.log" 2>&1
        cargo build --locked --release --bin opus-rs-oracle --features "$features" \
            --target "$rust_host" --target wasm32-wasip1 \
            > "$output_dir/qext-$name-build.log" 2>&1
        cp "$CARGO_TARGET_DIR/$rust_host/release/opus-rs-oracle" "$native"
        cp "$CARGO_TARGET_DIR/wasm32-wasip1/release/opus-rs-oracle.wasm" "$module"
    fi
    test -x "$native"
    test -f "$module"
    # Separate launchers preserve the model choice for each oracle invocation.
    printf '#!/usr/bin/env bash\nexport OPUS_WASM_ORACLE=%q\nexec %q "$@"\n' \
        "$module" "$root/tools/wasm/oracle.sh" > "$native-wasi.sh"
    chmod +x "$native-wasi.sh"
done

deep_native="$artifact_dir/qext-deep-plc-$profile"
osce_native="$artifact_dir/qext-osce-$profile"
osce_weights=${OPUS_OSCE_WEIGHTS:-"$osce_weights"}
if [[ ${OPUS_NEURAL_SKIP_C:-0} != 1 ]]; then
    python3 tools/reference/compare_qext_neural.py \
        --dred-reference "$root/target/reference-dred$reference_suffix-qext/opus-reference" \
        --osce-reference "$root/target/reference-osce$reference_suffix-qext/opus-reference" \
        --candidate "$deep_native" --osce-candidate "$osce_native" \
        --osce-weights "$osce_weights" | tee "$output_dir/qext-native.log"
fi
python3 tools/reference/compare_qext_neural.py \
    --dred-reference "$deep_native" --osce-reference "$osce_native" \
    --candidate "$deep_native-wasi.sh" --osce-candidate "$osce_native-wasi.sh" \
    --osce-weights "$osce_weights" --osce-reference-weights "$osce_weights" \
    | tee "$output_dir/qext-wasi.log"
