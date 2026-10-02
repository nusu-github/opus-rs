#!/usr/bin/env bash
# Run the model-dependent exact OSCE network tests with an explicitly supplied blob.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
: "${OSCE_WEIGHTS_PATH:?Set OSCE_WEIGHTS_PATH to a blob created by export_osce_weights.py}"
features=osce
if [[ ${OPUS_DNN_DEBUG_FLOAT:-0} == 1 ]]; then features+=,dnn_debug_float; fi
cargo test --locked --features "$features" --lib osce::tests::networks_match_pinned_scalar_c -- --ignored --exact
