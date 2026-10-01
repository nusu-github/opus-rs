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
  including fixed point, custom modes, DRED, deep PLC, and OSCE. Unimplemented
  or unverified configurations must remain explicit completion blockers.
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
source attribution is retained.

## Build

```sh
cargo build --lib
rustup target add wasm32-unknown-unknown wasm32-wasip1
cargo check --lib --target wasm32-unknown-unknown
cargo test --lib
```

The Rust library is rooted at `rust/src/lib.rs`. The reference C build is
independent of Cargo; no `build.rs` compiles or links the C codec into Rust.

## Current verification status

Implementation and differential validation are in progress. Do not treat this
branch as a completed or numerically compatible port until the acceptance
criteria above are all supported by recorded test results.
