#![allow(dead_code)]

//! Floating-point to integer conversion helpers from `celt/float_cast.h`.
//!
//! The original header provides a family of macros that round floating-point
//! samples to integral types using the rounding behaviour guaranteed by C99's
//! `lrintf()`/`lrint()` functions.  CELT relies on these helpers when bridging
//! between the float API and the fixed-point internals.  The Rust port exposes
//! equivalent functions so that other translated modules can depend on the same
//! rounding semantics without reimplementing the details.

use libm::rintf;

/// Scaling factor used by CELT to map floating-point samples to its internal
/// fixed-point representation.
pub(crate) const CELT_SIG_SCALE: f32 = 32_768.0;

/// Rounds a `f32` to the nearest `i32`, matching the behaviour of the
/// `float2int()` helper from the C implementation.
///
/// The reference code delegates to `lrintf()` when it is available, which
/// rounds to the nearest integer using the current floating-point rounding
/// mode (round-to-nearest-even in practice).  Rust's `as` conversion from
/// `f32` to `i32` already saturates on overflow, so the implementation simply
/// applies `rintf()` before casting.
#[must_use]
pub(crate) fn float2int(value: f32) -> i32 {
    rintf(value) as i32
}

/// Converts a floating-point sample to a signed 16-bit integer using CELT's
/// canonical scaling and rounding behaviour.
///
/// Mirrors the `FLOAT2INT16()` macro in `float_cast.h` by scaling the input,
/// clamping it to the representable range, and rounding ties to even.
#[must_use]
#[inline]
pub(crate) fn float2int16(value: f32) -> i16 {
    let scaled = (value * CELT_SIG_SCALE).clamp(-32_768.0, 32_767.0);
    // Throughout this bounded range, adding 1.5 * 2^23 places the result in
    // the binade whose f32 spacing is exactly one. The addition therefore
    // rounds to the nearest even integer; subtracting the bias is exact.
    // This also allows LLVM to vectorize PCM conversion without calling the
    // general software rintf implementation for every sample. NaN still casts
    // to zero, and infinities are clamped before the adjustment.
    const BIAS: f32 = 12_582_912.0;
    ((scaled + BIAS) - BIAS) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float2int_rounds_to_nearest_even() {
        // Half-way cases should round to the nearest even integer, matching the
        // default IEEE 754 rounding mode used by the C implementation.
        assert_eq!(float2int(1.5), 2);
        assert_eq!(float2int(2.5), 2);
        assert_eq!(float2int(-1.5), -2);
        assert_eq!(float2int(-2.5), -2);
    }

    #[test]
    fn float2int16_clamps_to_i16_range() {
        // Values outside the 16-bit range are clamped before rounding.
        assert_eq!(float2int16(2.0), 32_767);
        assert_eq!(float2int16(-2.0), -32_768);
        // In-range values follow the same rounding mode as float2int().
        assert_eq!(float2int16(0.500_1 / CELT_SIG_SCALE), 1);
        assert_eq!(float2int16(-0.500_1 / CELT_SIG_SCALE), -1);
    }

    #[test]
    fn bounded_pcm_rounding_matches_general_rintf() {
        for integer in -32768..=32767 {
            for fraction in [-0.501f32, -0.5, -0.499, 0.0, 0.499, 0.5, 0.501] {
                let value = (integer as f32 + fraction) / CELT_SIG_SCALE;
                let expected = rintf((value * CELT_SIG_SCALE).clamp(-32768.0, 32767.0)) as i16;
                assert_eq!(float2int16(value), expected, "{value:?}");
            }
        }
        let mut bits = 1u32;
        for _ in 0..100_000 {
            bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = f32::from_bits(bits);
            let expected = rintf((value * CELT_SIG_SCALE).clamp(-32768.0, 32767.0)) as i16;
            assert_eq!(float2int16(value), expected, "bits={bits:08x}");
        }
    }
}
