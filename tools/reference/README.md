# Independent Opus compatibility oracle

The oracle is a **test-only executable**, compiled from the original C Git object
`503d81b138d76621aae4b12786e90de48aa8db3a`. The Rust library does not link to C,
use FFI, launch the oracle, or require a C toolchain. Committed fixtures can be
consumed by ordinary Rust tests without building C.

## Reference arithmetic profile

`build.sh` extracts the pinned source into `target/reference/source` and builds
its scalar floating-point implementation with a C99 compiler and GNU Make. It
sets `OPUS_BUILD`, `VAR_ARRAYS`, and `HAVE_LRINTF`, and compiles with
`-O2 -ffp-contract=off -fno-fast-math -fno-tree-vectorize`. No SIMD dispatch,
`FLOAT_APPROX`, fixed-point mode, DRED, OSCE, or experimental PFA is enabled.
Floating-point contraction is disabled so separate multiplies and additions
remain separate operations. Integer PCM conversion uses `lrintf` under the
process's default rounding mode. This profile is deliberately explicit: different
Opus arithmetic profiles must not silently share golden results.

The build requires `git`, `tar`, `make`, and a GCC-compatible C compiler. Set `CC`,
`OPUS_REFERENCE_JOBS`, or `OPUS_REFERENCE_BUILD_DIR` to override defaults.
Generated build files belong under `target/`, not in the source tree.

## Reproduce CI

`verify.sh` runs the same native comparisons as the Rust CI workflow:

```sh
bash tools/reference/verify.sh baseline float
bash tools/reference/verify.sh baseline fixed
bash tools/reference/verify.sh baseline fixed-res24
bash tools/reference/verify.sh pfa fixed-res24
bash tools/reference/verify.sh custom float
bash tools/reference/verify.sh custom-pfa fixed
```

The baseline suite runs 302 encoder configurations, 906 normal/loss/FEC
cross-decoder cases, 28 persistent-state sequences, 132 encoder controls with
and without in-band FEC, 112 DTX/VBR/application cases, and the 120 official
RFC 8251 vector/rate/channel combinations. Packet, multistream, and projection
process-oracle tests run once in the float job. PFA jobs regenerate transform
fixtures and compare all 98 CELT encoder and 294 decoder configurations plus
the state sequences. The six Custom jobs regenerate C mode/PCM fixtures and
test float, fixed, and 24-bit arithmetic with and without PFA.
Fixed and 24-bit baseline jobs also run their complete unit suites. Both
baseline and PFA fixed profiles check their own PLC and integer-fade fixtures.

The runner builds optimized Rust binaries, retains test debug assertions, and
writes logs and profile-labeled manifests to `target/verification/`. It clears
ambient `OPUS_ORACLE_*` diagnostic controls. `OPUS_VERIFY_CANDIDATE` and
`OPUS_VERIFY_REFERENCE` can select already-built executables; their arithmetic
profiles must match the requested profile. `OPUS_VERIFY_DRY_RUN=1` prints the
commands without executing them. `OPUS_VERIFY_VECTORS=0` explicitly omits the
official corpus for an offline partial run; CI always runs it.

The separate Wasm workflow owns native/WASI runtime comparisons, avoiding
duplicate runtime jobs in this workflow.

```sh
./tools/reference/build.sh
cargo build --bin opus-rs-oracle
python3 tools/reference/compare_codec.py --candidate target/debug/opus-rs-oracle --quick
python3 tools/reference/compare_decode.py --candidate target/debug/opus-rs-oracle --quick --loss
```

Remove `--quick` to cover all 302 encoder configurations. They span all five
standard sample rates, mono and stereo, SILK, hybrid, CELT, automatic mode,
2.5 through 120 ms durations where the requested mode supports them, and
integer-generated mixed, silent, impulse, alternating-extrema, and noise input.
Each case encodes three consecutive frames by default, preserving state.
The cross-decoder uses five frames by default; `--loss` adds a dropped frame and
a decode-FEC recovery request. These recovery requests also test the fallback
behavior when the encoder did not send redundancy. This is a finite regression
matrix, not a proof of compatibility for every valid Opus input or feature.

A successful comparison requires exact packet bytes, exact integer PCM samples,
identical IEEE-754 float bit patterns, identical sample counts, and identical
encoder/decoder final entropy ranges. There is no tolerance or quality-score
threshold. Failures report the first mismatching field and sample/byte index;
they are never filtered out. Nonzero process exits, including panics, fail the
comparison. A reference self-consistency run establishes fixture generation
only; it does not establish Rust compatibility.

## Complete golden vectors

```sh
python3 tools/reference/generate_codec_fixture.py
python3 tools/reference/compare_codec.py --record tests/fixtures/reference/codec-manifest.json
```

`rust/tests/fixtures/reference/celt-stereo-20ms.pcm` is integer-generated input,
not third-party audio. Its `.tsv` file contains all reference packets and decoded
samples for three 48 kHz stereo 20 ms frames, forced CELT, 96 kbit/s CBR,
complexity 10, audio application. `rust/tests/codec_reference.rs` independently
checks encoder packets and ranges, decoder integer PCM, and decoder float bits.
The separate JSON manifest records input/output SHA-256 digests for the full C
matrix; a manifest generated without `--candidate` is not a Rust test report.

To preserve complete cross-decoding evidence for debugging:

```sh
python3 tools/reference/compare_decode.py --quick --loss --output-dir target/reference/corpus
python3 tools/reference/compare_decode.py --candidate target/debug/opus-rs-oracle --quick --loss --output-dir target/reference/comparison
```

## SILK encoder controls

`rust/tests/silk_encoder_reference.rs` checks 420 independent C-generated
configurations per profile from `fixtures/reference/silk-encoder{,-fixed,-fixed-res24}.tsv`: all standard SILK
sample rates and packet durations, each complexity level from 0 through 10,
CBR and VBR, mono and stereo, and in-band FEC with 15 percent expected loss.
Each configuration keeps encoder history for three or five packets. The fixture
contains every packet byte and final entropy range; ordinary `cargo test`
requires no C compiler.

```sh
python3 tools/reference/generate_silk_encoder_vectors.py
python3 tools/reference/generate_silk_encoder_vectors.py --profile fixed
python3 tools/reference/generate_silk_encoder_vectors.py --profile fixed-res24
python3 tools/reference/compare_controls.py --candidate target/debug/opus-rs-oracle
OPUS_ORACLE_FEC=1 OPUS_ORACLE_LOSS=15 python3 tools/reference/compare_controls.py --candidate target/debug/opus-rs-oracle
```

## Subprocess protocol

All decimal integers and hexadecimal strings use the C locale. Outputs are
UTF-8 tab-separated records, terminated by newlines. PCM input is signed 16-bit
little-endian, interleaved by channel. Float output uses each sample's raw
32-bit word as eight hexadecimal digits, independent of host byte order.

### Codec

```text
opus-reference codec RATE CHANNELS FRAME_SIZE FRAME_COUNT MODE BITRATE INPUT_I16LE
```

`MODE` is `-1000` (automatic), `1000` (SILK), `1001` (hybrid), or `1002` (CELT).
The application is audio; VBR is disabled and complexity is 10 by default.
Set `OPUS_ORACLE_VBR=1` or `OPUS_ORACLE_COMPLEXITY=0` through `10` to
exercise another configuration in both oracle processes. `OPUS_ORACLE_FEC=1`
and `OPUS_ORACLE_LOSS=15` enable redundancy with an expected 15 percent packet
loss; their defaults are zero. Forced SILK
bandwidth is narrowband at 8 kHz, mediumband at 12 kHz, or wideband otherwise.
Forced hybrid bandwidth is superwideband at 24 kHz or fullband at 48 kHz.
Each process owns one encoder and separate integer/float decoders.

| Field | Meaning |
| --- | --- |
| `C` | Record type |
| index | Zero-based frame index |
| packet length | Encoded byte count |
| samples | Decoded samples per channel |
| encoder range | `OPUS_GET_FINAL_RANGE` from the encoder |
| integer decoder range | Final range from the integer decoder |
| float decoder range | Final range from the float decoder |
| packet | Encoded bytes as hexadecimal |
| integer PCM | Interleaved signed 16-bit little-endian samples as hexadecimal |
| float PCM | Interleaved IEEE-754 32-bit words as hexadecimal |

### Cross-decoding

```text
opus-reference decode RATE CHANNELS PACKETS_FILE
```

Each input line is `FRAME_SIZE FEC HEX`, separated by whitespace. `-` in place of
`HEX` requests packet loss concealment. `FEC` is 0 or 1. Output fields are
`D`, index, decoded samples (or negative Opus error), integer decoder range,
float decoder range, integer PCM hexadecimal, float PCM hexadecimal. Decoder
state persists across all lines. Negative decode errors have empty PCM fields.

### Packet parsing

```text
opus-reference packet HEX RATE
```

The first record is `P`, parse result, TOC, payload offset, bandwidth, channels,
samples per frame, reported frame count, and packet sample count. Empty packets
use `OPUS_BAD_ARG` for helpers that require a TOC byte. Successful parsing adds
one `F`, byte offset, length record per frame. Outputs initialized to zero on
parse failure are oracle conventions, not additional C API guarantees.

### Entropy coding

```text
opus-reference entropy SCRIPT
```

A script starts with `size N` and then one operation per line:

```text
uint VALUE TOTAL
bits VALUE BITS
bit VALUE LOGP
icdf SYMBOL
icdf16 SYMBOL
encode LOW HIGH TOTAL
bin LOW HIGH BITS
patch VALUE BITS
shrink SIZE
```

`icdf` uses `[192, 128, 64, 0]` with 8-bit precision. `icdf16` uses
`[30000, 20000, 10000, 0]` with 15-bit precision. Scripts must satisfy the C
entropy API's preconditions. Comments start with `#`.

Each encoder operation produces `E`, operation index, tell, fractional tell,
range, and error. Finishing produces `B`, final storage size, error, and the
complete packet buffer. Decoding the resulting packet produces `D`, operation
index, decoded value, tell, fractional tell, range, and error. Patch and shrink
have no decoder operation. General-interval decode values are the C cumulative
frequency, not necessarily the interval's lower bound.

### Optional speech enhancement models

The `osce` Rust feature enables the LACE, NoLACE, and BBWENet implementations.
Models are supplied through the decoder's model-loading API as an owned,
little-endian Opus weight blob. The source archive used for the reference
contains generated `lace_data`, `nolace_data`, and `bbwenet_data` exports.
The runtime does not compile or call C, download models, or retain borrowed
references to the caller's blob.

Generate the model metadata and portable blob from the verified model exports:

```sh
python3 tools/reference/generate_osce_specs.py "$OSCE_MODEL_DIRECTORY"
python3 tools/reference/export_osce_weights.py "$OSCE_MODEL_DIRECTORY" target/reference/osce-weights-quantized.bin
OSCE_MODEL_DIRECTORY="$OSCE_MODEL_DIRECTORY" tools/reference/build_osce_reference.sh > target/reference/osce-network-vectors.txt
cmp target/reference/osce-network-vectors.txt rust/tests/fixtures/reference/osce-networks-quantized.txt
OSCE_WEIGHTS_PATH="$PWD/target/reference/osce-weights-quantized.bin" tools/reference/check_osce.sh
```

The external-model test is explicitly ignored by ordinary test discovery;
`check_osce.sh` runs it with the required model artifact. It compares all output
float bits for five consecutive frames of each enhancement network against the
independent scalar C implementation. The C build disables vector helper selection
in DNN `vec.h`, in addition to disabling contraction and automatic vectorization.

### Official packets, state transitions, and fixed arithmetic

`fetch_vectors.py` verifies and extracts the twelve official RFC 8251 packet
streams. `compare_vectors.py` compares their decoded PCM and final ranges with
both independent scalar implementations at every supported output rate:

```sh
python3 tools/reference/fetch_vectors.py
python3 tools/reference/compare_vectors.py --candidate target/debug/opus-rs-oracle --rates 8000 12000 16000 24000 48000 --record target/reference/rfc8251-results.json
python3 tools/reference/compare_sequences.py --candidate target/debug/opus-rs-oracle --decode-only
```

These comparisons require exact PCM values and exact float bits, rather than the
quality tolerance used for the RFC's `.dec` audio files. The sequence runner
exercises mode changes, reset, gain changes, FEC, consecutive packet losses,
periodic-to-noise concealment, and recovery. The `decode` protocol also accepts
`reset` and `gain SIGNED_Q8_DB` control lines; these do not emit a PCM row.

Build a separate fixed-point C profile with `tools/reference/build.sh fixed`.
Its executable and source archive live in `target/reference-fixed`. Compare it
with a Rust executable built using `--features fixed_point`; a passing float
profile is not evidence for the fixed-point profile.

The `codec_float` and `codec24` commands share the `codec` arguments, accepting
little-endian float32 and signed 24-bit samples stored in int32 respectively.
`codec_transition` changes mode every four frames through SILK, hybrid, CELT,
hybrid, and SILK, with FEC enabled and a 15% expected packet-loss setting.

The ordinary `osce` feature tests also use committed, independent feature and
adaptive-filter fixtures. The feature fixture covers eight successive frames,
10/20 ms durations, voiced/unvoiced/inactive speech, both LPC orders, bandwidth
extension state, and float/integer crossfades. Regenerate the feature fixture:

```sh
tools/reference/build_osce_features.sh > target/reference/osce-features.txt
cmp target/reference/osce-features.txt rust/tests/fixtures/reference/osce-features.txt
cargo test --features osce --lib osce::features::tests::features_match_pinned_scalar_c
cargo test --features osce --lib osce::nndsp::tests::matches_pinned_scalar_c_adaptive_filters
```

The feature FFT and DCT use the pinned `dnn/lpcnet_tables.c` coefficients. The
adaptive filter fixture is produced by `nndsp_vectors.c` and the same scalar
DNN compilation settings. These fixtures compare IEEE-754 bits, including
state carried across frames; model inference has its own separate fixture.

### Current model provenance and DRED

The pinned reference's `autogen.sh` selects model archive
`opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz`.
`fetch_models.py` verifies that SHA-256 before extracting it. The earlier
`4ec556...` model used by the original Rust donor has a different DRED network
and packet version; it is not a substitute for this revision.

The standard neural profile uses the reference's `DISABLE_DEBUG_FLOAT` setting:
layers with quantized int8 weights use those weights, while layers without a
quantized representation retain their float parameters. This is the default
for the C builders, Rust bundled weights, and OSCE blob exporter. Quantized
fixtures and C build directories use a `-quantized` suffix.

The diagnostic full-precision model profile is separate. Set
`OPUS_DNN_DEBUG_FLOAT=1` for the C builders, enable Rust's `dnn_debug_float`
feature alongside the desired neural features, and pass `--debug-float` to
`export_osce_weights.py`. Its fixtures and reference directories retain their
unsuffixed names. The neural CI matrix verifies both profiles independently.

For combined neural and quality-extension checks, set `OPUS_REFERENCE_QEXT=1`
when running `build_dred_codec.sh` or `build_osce_codec.sh`. These builds enable
96 kHz support and append `-qext` to the selected reference directory, keeping
the ordinary neural oracles intact. Enable Rust's `enable_qext` feature alongside
the corresponding neural features and model arithmetic profile.

The combined `deep_plc_weights,osce,enable_qext` configuration requires a blob
containing both OSCE and Deep PLC parameters. An OSCE-only blob cannot initialize
the Deep PLC model. `--all-models` includes LACE, NoLACE, BBWENet, PLC, PitchDNN,
FARGAN, and the DRED encoder/decoder arrays without changing the exporter's
default OSCE-only output.

```sh
OPUS_DNN_DEBUG_FLOAT=0 OPUS_REFERENCE_QEXT=1 bash tools/reference/build_combined_neural_codec.sh
python3 tools/reference/export_osce_weights.py "$OSCE_MODEL_DIRECTORY" \
  target/reference/all-models-quantized.bin --all-models
cargo build --locked --release --features deep_plc_weights,osce,enable_qext --bin opus-rs-oracle
python3 tools/reference/compare_combined_neural.py \
  --candidate target/release/opus-rs-oracle \
  --weights target/reference/all-models-quantized.bin \
  --record target/reference/combined-neural-quantized.json
```

This selected matrix checks 110 cases at 48 and 96 kHz: complete encoding,
LACE/NoLACE and bandwidth extension, normal decoding, in-band FEC, 400 ms loss
bursts, reset and gain changes, and immediate/deferred DRED recovery. Its
reset cases retain the bandwidth-extension history that precedes the pinned
C decoder's reset boundary. To test diagnostic models, build C with
`OPUS_DNN_DEBUG_FLOAT=1`, export with `--debug-float`, enable Rust's
`dnn_debug_float`, and select
`--reference target/reference-neural-combined-qext/opus-reference`. The neural
CI workflow tests both model arithmetic profiles independently.

For the combined PFA profile, also set `OPUS_REFERENCE_PFA=1` when building C,
add Rust's `pfa` feature, and append `-pfa` to the reference directory. The same
110 cases check the interaction between the alternate transform and neural
feature extraction, prediction, and recovery in both model arithmetic profiles.

```sh
python3 tools/reference/fetch_models.py
export OSCE_MODEL_DIRECTORY="$PWD/target/reference/models/dnn"
export DRED_WEIGHTS_PATH="$OSCE_MODEL_DIRECTORY"
export DNN_WEIGHTS_PATH="$OSCE_MODEL_DIRECTORY"
tools/reference/build_dred_networks.sh > target/reference/dred-networks-quantized.txt
cmp target/reference/dred-networks-quantized.txt rust/tests/fixtures/reference/dred-networks-quantized.txt
tools/reference/build_dred_packets.sh > target/reference/dred-packets-quantized.txt
cmp target/reference/dred-packets-quantized.txt rust/tests/fixtures/reference/dred-packets-quantized.txt
tools/reference/build_deep_plc.sh > target/reference/deep-plc-quantized.txt
cmp target/reference/deep-plc-quantized.txt rust/tests/fixtures/reference/deep-plc-quantized.txt
cargo test --features deep_plc_weights --lib pinned_scalar_c
cargo test --features dred --lib current_packet_reference
```

The RDOVAE fixture checks sixteen stateful encode/decode steps, including
reinitialization, and every latent, state, and output-feature bit. Its decoder
input includes the current quantizer-conditioning channel. The packet fixture
covers 48 independent quantizer, activity, budget, and redundancy-offset
settings and compares complete payloads and decoded states. The neural PLC
fixture checks 64 successive feature/update/concealment/recovery frames, including
a 240 ms loss burst that reaches the neural attenuation floor.

The local optional weight packages convert the verified generated parameter
arrays into Rust data at build time. They do not compile or link C. Model
inference and all codec algorithms remain in safe Rust.

The Burg cepstrum has a separate 32-frame fixture to isolate feature rounding
from neural prediction. Regenerate it with `build_burg_cepstrum.sh`. The complete
OSCE decoder comparison uses loaded models for 80 rate/channel/mode/complexity
and bandwidth-extension cases, including loss:

```sh
bash tools/reference/build_osce_codec.sh
cargo build --features osce --bin opus-rs-oracle
python3 tools/reference/compare_osce.py --candidate target/debug/opus-rs-oracle
bash tools/reference/build_dred_codec.sh
cargo build --features deep_plc_weights --bin opus-rs-oracle
OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_decode.py --reference target/reference-dred-quantized/opus-reference --candidate target/debug/opus-rs-oracle --loss
OPUS_ORACLE_DECODER_COMPLEXITY=10 python3 tools/reference/compare_sequences.py --reference target/reference-dred-quantized/opus-reference --candidate target/debug/opus-rs-oracle --decode-only
python3 tools/reference/compare_dred.py --candidate target/debug/opus-rs-oracle
```

`dred_decode RATE CHANNELS SCRIPT` accepts `FRAME_SIZE OFFSET HEX` records.
An offset of `-1` decodes the packet normally; a nonnegative offset decodes that
many samples before the packet using its DRED redundancy. Recovery emits an
`R` record containing the frame index, available DRED samples, and DRED end,
followed by the ordinary `D` PCM record. `OPUS_ORACLE_DRED_DEFER=1` separates
parsing from neural processing. The recovery matrix tests both paths with
one, three, and eight consecutive lost packets at every supported output rate.
The complete matrix has 204 cases across SILK, hybrid, CELT, and automatic mode,
with one or two channels and immediate or deferred redundancy processing.
Each input also produces a `Q` record containing the exact signed 24-bit PCM
words and the decoder range from an independent 24-bit decoder state.
It uses decoder complexity zero, so the recovered audio depends on the DRED
feature queue rather than ordinary neural concealment activation.

The separate `compare_qext_neural.py` matrix covers 104 combined-feature cases
at 96 kHz. Supply a `deep_plc_weights,enable_qext` executable with `--candidate`
and an `osce,enable_qext` executable with `--osce-candidate`, plus the matching
`-qext` C references and OSCE blob. Keeping these executables separate allows
the OSCE-only blob to load without requiring the independent Deep PLC models.

The neural CI workflow fetches the verified model archive, regenerates the C
fixtures, runs these comparisons, and compiles the neural algorithms for Wasm.

### Fixed-point decoder regressions

`generate_downmix_vectors.py PROFILE` regenerates the exact fixed CELT
stereo-to-mono regression for `fixed` and `fixed-res24`, optionally suffixed
with `-qext`, `-pfa`, or both. Its two packets are the first packets of official
RFC 8251 `testvector01.bit` and `testvector11.bit` from the archive identified
above. Each profile checks 8 and 48 kHz output, both PCM APIs and final ranges.
QEXT profiles also include generated 48 and 96 kHz stereo packets with actual
extension payloads, decoded to mono at the corresponding rate.
The Rust `codec_downmix_reference` test checks the complete PCM hashes; it
guards the reference's separate arithmetic shifts before channel addition.

`build_decoder_gain.sh` regenerates the independent fixed-point decoder gain
fixtures for 16-bit and 24-bit PCM. Each profile covers 272 combinations of
sample boundaries and Q8 gain values, including zero, extreme attenuation and
amplification, and rounding boundaries around one decibel. The Rust
`fixed_decoder_gain_matches_pinned_c` test compares the exact gain coefficient
and output PCM bits, including the reference's integer saturation behavior.

### PFA transform profiles

`build.sh pfa`, `build.sh fixed-pfa`, and `build.sh fixed-res24-pfa` create separate scalar reference builds
with `ENABLE_PFA`. `generate_pfa_transform_vectors.py` and its `--fixed` variant
record all four canonical FFT sizes and MDCT shifts, four input patterns, both
transform directions, and contiguous/interleaved MDCT layouts. Floating-point
vectors store exact IEEE-754 bits; fixed-point vectors store exact integers.

### Complete QEXT codec profiles

The QEXT workflow compares full packets, encoder and decoder ranges, and every
decoded sample in the `qext`, `fixed-qext`, and `fixed-res24-qext` scalar profiles.
Each profile runs 384 CELT configurations, 92 SILK/hybrid configurations at
96 kHz, and 4,230 multistream/surround/projection frames, including maximum packet
budgets. Use the matching Rust features and reference directory:

| C profile | Rust features |
| --- | --- |
| `qext` | `enable_qext` |
| `fixed-qext` | `fixed_point,enable_qext` |
| `fixed-res24-qext` | `enable_res24,enable_qext` |

```sh
bash tools/reference/build.sh qext
cargo build --locked --release --features enable_qext --bin opus-rs-oracle
python3 tools/reference/compare_qext.py --candidate target/release/opus-rs-oracle
python3 tools/reference/compare_qext_speech.py --candidate target/release/opus-rs-oracle
cargo test --locked --release --features enable_qext --test multistream_reference \
  -- --ignored --nocapture --test-threads=1
```

For fixed arithmetic, select the corresponding `--reference` path for both
Python commands. The Rust multistream test selects its reference directory from
its Cargo features. These checks are separate from the native/WASI comparisons.

Custom QEXT modes have six independent profiles: `float-qext`, `fixed-qext`,
and `fixed24-qext`, each with an optional `-pfa` suffix. Regenerate the C fixtures
before checking them, then add `custom_modes` and, for PFA, `pfa` to the Rust
features above. For example:

```sh
bash tools/reference/build_custom.sh float-qext-pfa
python3 tools/reference/generate_custom_vectors.py float-qext-pfa
cargo test --locked --release --features custom_modes,enable_qext,pfa --test custom_reference
```

Each custom profile checks 51 streams and 255 normal/loss frames with integer,
24-bit integer, and floating-point PCM. CI also verifies regenerated mode
tables and fixture drift.

### QEXT vector quantization

Build the `qext`, `fixed-qext`, or `fixed-res24-qext` C profile, then run
`generate_qext_vq_vectors.py` with no option, `--fixed`, or `--res24` respectively.
Each profile contains 1,888 independent cases: refined algebraic coding at
2–14 extra bits and cubic coding at resolutions 1–14, for dimensions 2, 3, 8,
and 24. The cases vary block layouts, spreading, gain, resynthesis, zero/sparse
inputs, pulse counts, and extension storage. Tight extension storage exercises
the uniform refinement branch; generous storage exercises entropy weighting.

`qext-vq.script` records the operation, name, dimension, pulse count or cubic
resolution, spreading, blocks, Q15 gain, resynthesis flag, extra bits, base and
extension capacities, then Q24 input integers. Float inputs divide those
integers by 2^24. Results contain collapse masks, both coder ranges and tells,
encoder and decoder coefficient words, and both complete packet buffers.
The C driver checks encoder errors and confirms both decoder final ranges.

`generate_qext_bands_vectors.py` accepts the same profile options and records
288 complete band quantization cases per profile. These cover all four frame
scales, long and short blocks, mono/stereo, dual and intensity stereo, theta
rate-distortion decisions, and zero/two/six extension bits per coefficient.
The core mode uses 48 kHz; additional bands use both 48 and 96 kHz modes.
Each binary fixture begins with little-endian words `0x51424e44` and the case
count. Each case stores its controls, per-band allocations and time/frequency
settings, exact C-normalized input coefficients, encoder and decoder output
coefficients, collapse masks, entropy states, random seeds, and complete packet
buffers. The record layout is defined by `qext_bands_vectors.c` and includes
explicit array and buffer lengths. Coefficients use IEEE-754 bits in float
profiles and signed integer bits in fixed profiles.

The encoder edge corpus is regenerated with
`python3 tools/reference/generate_encoder_edge_vectors.py` and checked with
`cargo test --test encoder_edges_reference`. Its 112 configurations cover audio
and voice applications, signal hints, constrained and unconstrained VBR, and
50-frame DTX silence/activity transitions in each coding mode. Every packet and
entropy range is compared exactly; 112 selected DTX boundary frames additionally
compare every integer PCM sample and floating-point PCM bit pattern. This corpus
uses the default floating-point reference profile without OSCE.

## QEXT band allocation and fine energy

`qext_allocation_vectors.c` generates 432 cases for each scalar C profile with
`ENABLE_QEXT`. The checked-in `qext-allocation.tsv` and
`fixed-qext-allocation.tsv` cover both 48 and 96 kHz modes, one and two channels,
all four frame scales, optional extension bands, low and high bit budgets,
tonality weighting, and successive fine-energy quantization through 14 bits.
The Rust component test compares every packet byte, entropy state, allocation,
and reconstructed energy; it also decodes each C packet independently.

```sh
python3 tools/reference/generate_qext_allocation_vectors.py
cargo test --features enable_qext --lib allocation_and_fine_energy_match_c
cargo test --features enable_qext,fixed_point --lib allocation_and_fine_energy_match_c
```

## QEXT SILK resampling

`qext-resampler.tsv` contains 24 stateful C vectors for 96-to-16 kHz encoding
and 8/12/16-to-96 kHz decoding. Each conversion retains its filter history over
six alternating 10 and 20 ms input blocks. Every output sample is compared
exactly, including the 44-sample encoder input delay at 96 kHz.

```sh
python3 tools/reference/generate_qext_resampler_vectors.py
cargo test --features enable_qext --lib qext_96khz_matches_c_stateful_vectors
```

## SILK stereo transitions

`silk-stereo.tsv` contains 72 stateful C vectors covering 8/12/16 kHz, 10/20 ms
frames, changing low and high bitrates, and forced mono transitions. The test
compares the width, predictor, amplitude history, rate allocation, transmitted
indices, and every transformed mid/side sample. It includes reduced-width
operation, where the width quotient uses Q16 before its Q14 clamp.

```sh
python3 tools/reference/generate_silk_stereo_vectors.py
cargo test --lib width_and_predictor_transitions_match_c
```

## QEXT 96 kHz speech modes

`compare_qext_speech.py` compares 92 complete SILK and hybrid codec cases at
96 kHz, including 10–120 ms packets, mono/stereo, VBR/CBR, four edge signals,
and high-rate hybrid encoding. Packet bytes, entropy ranges, and integer and
floating-point decoded samples must match exactly.

```sh
python3 tools/reference/compare_qext_speech.py \
  --candidate target/release/opus-rs-oracle \
  --record target/reference/qext-speech.json
```
