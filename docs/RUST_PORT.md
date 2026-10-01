# Safe Rust Opus port

## Goal and acceptance criteria

Port the Opus implementation at C reference commit
`503d81b138d76621aae4b12786e90de48aa8db3a` to Rust. Rust API compatibility with
libopus is not required. Numerical compatibility is required; perceptual
similarity, successful round trips, and matching packet lengths are insufficient.

Completion requires all of the following:

- A pure Rust encoder and decoder for SILK, CELT, and hybrid modes, including
  mono/stereo, supported sample rates and frame durations, rate control,
  mode transitions, DTX, FEC, packet loss concealment, and reset/control behavior.
- Packet parsing, repacketization, padding, multistream, and projection support.
- Exact C/Rust encoded packets, decoded integer samples, floating-point sample
  bits, and entropy final ranges under a recorded reference build profile.
- No unsafe Rust implementation or C/FFI dependency in the codec. Enforce
  `#![forbid(unsafe_code)]`; C is allowed only as an independent test oracle.
- A `no_std` + `alloc` core that builds for `wasm32-unknown-unknown` and runs
  equivalent vectors on a WebAssembly runtime, with native/Wasm agreement.
- A documented feature matrix for every non-default reference configuration,
  including fixed point, 24-bit PCM, custom modes, PFA, QEXT, DRED, deep PLC,
  and OSCE. Unimplemented or unverified configurations must remain explicit
  completion blockers.
- Reproducible tests and truthful reporting: no skipped failing parity checks,
  relaxed numeric tolerances, or self-generated golden results presented as
  C-reference evidence.

## Numerical reference

The first verification profile uses the pinned C source with scalar floating
point, no intrinsics, no fast math, no floating-point approximation, and
`-ffp-contract=off`. Pinning the profile is necessary because libopus supports
multiple arithmetic implementations whose floating-point results can differ.
A passing profile does not establish compatibility with other configurations.

## Source provenance

The repository root retains the Xiph C reference and its existing copyright and
license notices (`COPYING`). The initial Rust source was imported from
[`mousiki` 0.2.1](https://crates.io/crates/mousiki/0.2.1), published from commit
`89a2f1d032d9147bac483425f1aa3512e3db1bbf`. Its MIT license is preserved in
`LICENSES/mousiki-MIT.txt`. This is a starting point for correction against the
pinned reference, not evidence of numerical equivalence. The imported source
includes algorithms originally ported from Xiph Opus and Pion Opus; existing
source attribution is retained. Range-decoder tests in `rust/src/range.rs` derive
from Pion Opus commit `e8536fe9e4ca2181db7d808e35d50b2c0400ceb1`. Its exact
[upstream MIT notice](https://github.com/pion/opus/blob/e8536fe9e4ca2181db7d808e35d50b2c0400ceb1/LICENSE)
is retained in `LICENSES/pion-MIT.txt`, including the original copyright notice.

## Build

```sh
cargo build --lib
rustup target add wasm32-unknown-unknown wasm32-wasip1
cargo check --lib --target wasm32-unknown-unknown
cargo test --lib
```

The Rust library is rooted at `rust/src/lib.rs`. The reference C build is
independent of Cargo; no `build.rs` compiles or links the C codec into Rust.

## Completion record

The acceptance criteria are complete for the pinned portable scalar reference.
The Rust implementation covers SILK, CELT, hybrid, packet operations,
multistream, projection, fixed point, 24-bit PCM, Custom, PFA, QEXT, OSCE, DRED,
and deep PLC. Both the standard int8 and diagnostic float neural model profiles
have independent C comparisons. Configuration combinations rejected by the
C reference are rejected explicitly by Rust.

Exact comparison records include 302 encoder configurations and 906 decoder
configurations for each standard arithmetic profile, 384 QEXT codec
configurations per profile, and 120 complete RFC 8251 stream comparisons per
profile. The official streams contain 200,750 decoded packets per profile.
Native/WebAssembly execution agrees in 6,044 codec and neural comparisons;
40 additional fixed stereo-to-mono C regression cases also pass under WASI.
The codec and software math crates enforce `#![forbid(unsafe_code)]`, use
`no_std` with `alloc`, and have no C/FFI runtime dependency.

See [numerical compatibility](NUMERICAL_COMPATIBILITY.md) for the complete
feature and regression matrices, [Wasm verification](WASM.md) for target checks,
and [reference instructions](../tools/reference/README.md) for reproducible
oracle commands. Committed C fixtures and verification manifests are stored in
`rust/tests/fixtures/reference` and `tests/fixtures/reference`.

These finite regression results establish the documented compatibility target;
they are not a proof for every possible input or a claim of matching the
independent arithmetic of optional C SIMD kernels. The
[defined Custom input boundaries](CUSTOM_MODES.md) exclude C undefined behavior.
