# Opus Custom

Enable `custom_modes` to use the public `opus_rs::custom` API. Opus Custom is
CELT with an explicitly agreed sample rate and frame size. Its packets require
the same mode configuration at the sender and receiver; they are not ordinary
Opus packets with an arbitrary replacement sample rate.

```rust
use opus_rs::custom::Mode;

let mode = Mode::new(48_000, 512)?;
let view = mode.view();
let mut encoder = view.encoder(2)?;
let mut decoder = view.decoder(2)?;
encoder.set_complexity(10)?;

let input = [0i16; 512 * 2];
let mut packet = [0u8; 160];
let length = encoder.encode(&input, &mut packet)?;
let mut output = [0i16; 512 * 2];
assert_eq!(decoder.decode(Some(&packet[..length]), &mut output)?, 512);

// Conceal a lost frame while retaining decoder history.
decoder.decode(None, &mut output)?;
# Ok::<(), opus_rs::custom::Error>(())
```

`Mode` owns immutable mode tables. `ModeView` borrows those tables, and codec
states borrow the view. Keep both values alive until the encoders and decoders
are dropped. Rust enforces this lifetime relationship; construction does not
leak mode tables, use raw pointers, or require self-referential allocations.

Frame sizes and return values count samples per channel. PCM is interleaved.
`encode`/`decode` use signed 16-bit PCM, `encode_float`/`decode_float` use float
PCM normalized around ±1, and `encode_24`/`decode_24` use signed 24-bit values
in `i32` containers. The encoder requires exactly one frame of PCM. Decoder
output buffers must hold at least one frame. `reset` clears stream history;
`final_range` exposes the entropy-coder range for compatibility checks.

Encoder controls include complexity, bitrate, VBR and VBR constraint,
prediction, expected packet-loss percentage, input bit depth and stereo phase
inversion. Decoder controls include complexity and phase inversion.

Construction accepts the C implementation's 8–96 kHz rate range and 40–1024
sample size range (up to 2048 with `enable_qext`), subject to CELT's duration,
short-block and band-width constraints. The scalar FFT requires factors of
2, 3 and 5. Unsupported sizes return `Error` before constructing a transform.
The canonical 48 kHz modes and QEXT 96 kHz modes retain their prescribed
static tables; other modes generate their own tables.

The API supports `no_std` plus `alloc`, including WebAssembly. `fixed_point`
selects integer codec arithmetic, `enable_res24` additionally retains 24-bit
PCM precision, and `pfa` selects the optional prime-factor FFT where supported.
`enable_qext` additionally selects Q31 fixed coefficients, the QEXT transform
normalization, and the 96 kHz canonical mode. The encoder exposes
`set_qext(bool)` and `qext()` when that feature is enabled. Each combination
has separate C fixtures because its arithmetic can differ.

Run the independent normal-packet and packet-loss comparisons with:

```sh
cargo test --features custom_modes --test custom_reference
cargo test --features custom_modes,fixed_point --test custom_reference
cargo test --features custom_modes,enable_res24 --test custom_reference
cargo test --features custom_modes,pfa --test custom_reference
cargo test --features custom_modes,fixed_point,pfa --test custom_reference
cargo test --features custom_modes,enable_res24,pfa --test custom_reference
cargo test --features custom_modes --lib dynamic_mode_tables_match_pinned_c
```

See [fixture provenance](../rust/tests/fixtures/reference/CUSTOM.md) for the
pinned C revision, generation commands, input signals and coverage.

The same six commands can include `enable_qext` to select the corresponding
QEXT oracle profiles, including explicit high-rate extension payloads,
the 96 kHz/1920-sample mode and 90-sample short-block extension modes.
The constructor and transform checks also cover invalid dimensions and
unsupported FFT factorizations.

C accepts some dynamic modes whose frame consumes the decoder's prediction
history (for example 96 kHz with a 2048-sample frame), then reads before its
allocation during postfiltering. Rust preserves mode construction but returns
`Error::InvalidMode` when creating such a decoder. The canonical 96 kHz
1920-sample mode has doubled history and supports normal decoding and loss
concealment.
