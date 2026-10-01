# opus-rs

A safe, pure Rust implementation of Opus, ported against C revision
`503d81b138d76621aae4b12786e90de48aa8db3a`. Numerical tests require identical
encoded packets, decoded integer samples, floating-point bits, and entropy final
ranges under the documented scalar arithmetic profiles.

The codec and its software math library use `no_std` with `alloc` and enforce
`#![forbid(unsafe_code)]`. Cargo does not compile or link the C codec. The original
C source remains an independent test oracle with its original build system.

See [numerical compatibility](docs/NUMERICAL_COMPATIBILITY.md) for the feature
matrix, reference conditions, verification coverage, and defined input boundaries.
[The port goal](docs/RUST_PORT.md) records the acceptance criteria and source
provenance. Rust API compatibility with the C library is not required.

## Rust API

Rust 1.88 or newer is required.

```rust
use opus_rs::{Application, Bitrate, Channels, Decoder, Encoder};

let mut encoder = Encoder::builder(48_000, Channels::Stereo, Application::Audio)
    .bitrate(Bitrate::Bits(96_000))
    .build()?;
let mut decoder = Decoder::new(48_000, Channels::Stereo)?;
let input = [0i16; 960 * 2];
let mut packet = [0u8; 1500];
let length = encoder.encode(&input, &mut packet)?;
let mut output = [0i16; 960 * 2];
let samples_per_channel = decoder.decode(&packet[..length], &mut output, false)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The typed API also supports float and signed 24-bit PCM. Packet parsing,
repacketization, padding, multistream, projection, and model-loading APIs are
available under `c_style_api`. Enable `custom_modes` for the owned
[Opus Custom API](docs/CUSTOM_MODES.md).

## Verification

```sh
cargo test --locked
cargo test --locked --manifest-path rust/math/Cargo.toml
bash tools/reference/verify.sh baseline float
bash tools/reference/verify.sh baseline fixed
bash tools/reference/verify.sh baseline fixed-res24
```

Ordinary tests consume committed, independently generated C fixtures and require
no C compiler. `verify.sh` builds separate C and Rust executables and compares
complete results, including packet loss, FEC, transitions, controls, and official
RFC 8251 streams. Its `pfa`, `custom`, and `custom-pfa` suites select corresponding
reference configurations. See [oracle instructions](tools/reference/README.md)
for QEXT and neural verification and fixture regeneration.

For WebAssembly, install both targets and Node.js 24 or newer:

```sh
rustup target add wasm32-unknown-unknown wasm32-wasip1
./tools/wasm/check.sh
```

This checks the browser-compatible library, then compares native and WASI
execution. [Wasm verification](docs/WASM.md) describes feature selection and
stateful and neural comparisons. WASI is a test adapter, not a codec dependency.

## Optional configurations

| Feature | Behavior |
| --- | --- |
| `fixed_point` | Fixed-point arithmetic with 16-bit internal PCM |
| `enable_res24` | Retain 24-bit PCM precision; implies `fixed_point` |
| `pfa` | Prime-factor FFT |
| `custom_modes` | Owned Opus Custom modes and codec states |
| `enable_qext` | Quality extensions, 96 kHz, and Q31 fixed coefficients |
| `osce` | LACE, NoLACE, and BBWENet with supplied model blobs |
| `dred` | DRED encoding, parsing, neural processing, and recovery |
| `deep_plc` | Neural concealment; implies `dred` |
| `deep_plc_weights` | Bundle verified deep-PLC parameters |
| `dnn_debug_float` | Diagnostic float neural weights instead of standard int8 weights |

The default codec uses scalar float arithmetic. Neural configurations use the
standard C int8 model profile unless `dnn_debug_float` is selected. Fixed point
with OSCE, DRED, or deep PLC is rejected, matching the C configuration rules.

DRED and bundled deep-PLC parameters use separately downloaded Xiph models.
Set `DRED_WEIGHTS_PATH` and `DNN_WEIGHTS_PATH` to a local model directory or
archive before building the corresponding features. The pinned archive is
`opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz`
from [Xiph's model archive](https://media.xiph.org/opus/models/); the filename
contains its SHA-256 digest. The weight packages convert parameter exports into
Rust data and do not link libopus. Explicit fetch features permit build-time
downloads. The runtime codec never downloads models.

## License and provenance

The C reference is covered by [COPYING](COPYING). The Rust starting point was
[mousiki 0.2.1](https://crates.io/crates/mousiki/0.2.1); its MIT license is retained
in [LICENSES/mousiki-MIT.txt](LICENSES/mousiki-MIT.txt). Pion-derived range-decoder
tests retain their [MIT notice](LICENSES/pion-MIT.txt). Software math retains its
[MIT license](rust/math/LICENSE.txt). Source attribution and upstream notices are
preserved. The port plan records the pinned source revisions.
