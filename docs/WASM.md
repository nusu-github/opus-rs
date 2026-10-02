# WebAssembly numerical compatibility

The codec library uses `no_std` with `alloc` and contains no unsafe Rust. The
process oracle uses WASI for command-line arguments and fixture files; this
adapter is separate from the portable codec algorithms.

Install the `wasm32-unknown-unknown` and `wasm32-wasip1` Rust targets and Node.js
24 or newer, then run:

```sh
./tools/wasm/check.sh
```

The script checks the library for a browser-compatible Wasm target, builds the
same process oracle for the native host and WASI, and executes the complete
302-case codec matrix on both. It compares packet bytes, encoder and decoder
final ranges, integer PCM, and the exact bit patterns of floating-point PCM.
The independent C comparison remains a separate check; see
`tools/reference/compare_codec.py` and `tools/reference/compare_qext.py`.

`OPUS_RUST_FEATURES` selects Cargo features for both builds.
`OPUS_WASM_COMPARISON` selects the matrix; arguments after `check.sh` are passed
to that matrix unchanged.

| Comparison | Scope |
| --- | --- |
| `codec` (default) | Complete single-stream encoding and decoding |
| `qext` | QEXT at 48 and 96 kHz, multiple bitrates and frame durations |
| `qext_speech` | SILK and hybrid coding at 96 kHz |
| `decode` | Cross-decoding; add `--loss` for PLC and FEC |
| `sequences` | Mode transitions, reset, gain changes, and packet loss |

Examples:

```sh
OPUS_RUST_FEATURES=fixed_point ./tools/wasm/check.sh
OPUS_RUST_FEATURES=enable_res24 ./tools/wasm/check.sh
OPUS_RUST_FEATURES=pfa ./tools/wasm/check.sh
OPUS_RUST_FEATURES=enable_qext OPUS_WASM_COMPARISON=qext ./tools/wasm/check.sh
OPUS_RUST_FEATURES=enable_qext,enable_res24 OPUS_WASM_COMPARISON=qext ./tools/wasm/check.sh
OPUS_RUST_FEATURES=enable_qext OPUS_WASM_COMPARISON=qext_speech ./tools/wasm/check.sh
OPUS_RUST_FEATURES=fixed_point OPUS_WASM_COMPARISON=decode ./tools/wasm/check.sh --loss
OPUS_WASM_COMPARISON=sequences ./tools/wasm/check.sh
```

`CARGO_TARGET_DIR` is respected by both builds and the WASI launcher. Distinct
build directories keep concurrent feature-profile comparisons independent.
`tools/wasm/oracle.sh` also accepts `OPUS_WASM_ORACLE` when running a previously
built `.wasm` module directly.

The WebAssembly workflow executes nine profiles: floating point, fixed point,
and fixed point with 24-bit internal samples, each with the default transform,
the PFA transform, or QEXT. The six baseline/PFA profiles also exercise stateful
mode transitions, decoder gain changes, resets, and extended packet loss.
The three QEXT profiles also compare SILK and hybrid coding at 96 kHz.

The checked-in [Wasm manifest](../tests/fixtures/reference/wasm-manifest.json)
records 3,408 exact comparisons across these nine profiles, including their
arithmetic, transform, feature, source, and toolchain provenance. Neural model
comparisons and independent scalar C results are recorded separately.

Optional neural models have a separate runtime check. It compares each model
profile against scalar C, then compares native Rust against WASI. Both the
standard quantized models and the diagnostic floating-point models are tested.
Prepare the verified model archive and references before running it:

```sh
python3 tools/reference/fetch_models.py
export DNN_WEIGHTS_PATH="$PWD/target/reference/models/dnn"
export DRED_WEIGHTS_PATH="$DNN_WEIGHTS_PATH"
export OSCE_MODEL_DIRECTORY="$DNN_WEIGHTS_PATH"
export OPUS_DNN_DEBUG_FLOAT=0
bash tools/reference/build_dred_codec.sh
bash tools/reference/build_osce_codec.sh
python3 tools/reference/export_osce_weights.py "$OSCE_MODEL_DIRECTORY" \
    target/reference/osce-weights-quantized.bin
OPUS_NEURAL_FULL_DECODE=1 ./tools/wasm/check_neural.sh quantized
```

For the diagnostic profile, set `OPUS_DNN_DEBUG_FLOAT=1`, rebuild both C
references, export `target/reference/osce-weights.bin` with the exporter's
`--debug-float` option, and run `check_neural.sh debug-float`. The script adds
the corresponding `dnn_debug_float` Cargo feature automatically.

Each profile covers 906 Deep PLC decoding configurations, 24 stateful loss
recovery sequences, 204 DRED recovery configurations, and 80 OSCE decoding
configurations. Without `OPUS_NEURAL_FULL_DECODE=1`, Deep PLC uses a 21-case
smoke matrix. `OPUS_NEURAL_COMPONENT` can select `deep-plc`, `dred`, or `osce`;
the default is `all`. Deep PLC/DRED and OSCE use separate builds to preserve
the corresponding C configurations. Every build also checks
`wasm32-unknown-unknown`.

The dedicated neural WebAssembly workflow runs both complete profiles.
Artifacts are copied to `target/neural-wasm-binaries`, so feature builds can
share a Cargo cache without replacing binaries used by a running comparison.
Set `OPUS_NEURAL_SKIP_BUILD=1` to reuse these copied artifacts.

The neural workflow also covers the supported 96 kHz QEXT combinations in a
separate 104-case matrix for each model profile. After preparing the models,
build the matching C references with QEXT and run:

```sh
OPUS_REFERENCE_QEXT=1 bash tools/reference/build_dred_codec.sh
OPUS_REFERENCE_QEXT=1 bash tools/reference/build_osce_codec.sh
./tools/wasm/check_neural_qext.sh quantized
```

Use `OPUS_DNN_DEBUG_FLOAT=1` and `check_neural_qext.sh debug-float` for the
diagnostic profile. This matrix exercises DRED encoding and recovery, ordinary
PLC with a loaded neural model, and OSCE at 96 kHz. C intentionally uses
ordinary CELT PLC at that rate, even when a neural model is loaded.

The fixed stereo-to-mono regression runs directly under WASI against independent
C fixtures. All eight fixed/24-bit, scalar/PFA, and standard/QEXT profiles pass
40 cases, including actual 48/96 kHz QEXT extension payloads. The
[downmix manifest](../tests/fixtures/reference/downmix-wasm-manifest.json)
records commands, source and fixture hashes, modules, logs, and toolchain
provenance. For example:

```sh
CARGO_TARGET_WASM32_WASIP1_RUNNER='node --no-warnings tools/wasm/run.mjs' \
  cargo test --locked --target wasm32-wasip1 --features enable_res24,enable_qext,pfa \
  --test codec_downmix_reference
```
