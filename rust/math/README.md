# Opus portable math

A safe, scalar adaptation of Rust `libm` 0.2.16, used by the Opus algorithms.
The upstream MIT license and per-file notices are preserved. This crate is
`no_std`, forbids unsafe Rust, and does not depend on a C math library.

Changes from upstream: remove architecture assembly and runtime dispatch; use
checked indexing in every profile; use stable `to_bits`/`from_bits` rather than
transmute; use ordinary division with validated preconditions; use `black_box`
for required evaluation rather than volatile pointer reads. Generic software
floating-point algorithms and their coefficient tables are retained.

Floating-point status flags and dynamically selected rounding modes are outside
this crate's contract. Opus uses the standard round-to-nearest environment.

Run the upstream-derived unit tests with:

```sh
cargo test --manifest-path rust/math/Cargo.toml
```
