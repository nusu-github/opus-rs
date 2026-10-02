// Copyright (c) 2001-2011 Timothy B. Terriberry
// Copyright (c) 2008-2009 Xiph.Org Foundation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//! Public, allocation-free access to the codec's canonical entropy coder.
//!
//! The underlying implementation follows `celt/entcode.c`, `celt/entenc.c`, and
//! `celt/entdec.c`; the audio codec and this API share the same arithmetic.
//! Probability models must satisfy the original Opus range-coder contracts.
//! Exhausted input is zero padded, exactly as in the reference implementation.

use crate::celt::{EcDec, EcEnc, ec_tell, ec_tell_frac};

/// Fractional bit counts are measured in eighths of a bit.
pub const BITRES: u32 = 3;

/// A range encoder borrowing a caller-provided packet buffer.
#[derive(Debug)]
pub struct RangeEncoder<'a> {
    inner: EcEnc<'a>,
    finished: bool,
}

impl<'a> RangeEncoder<'a> {
    /// Starts a packet without allocating.
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Self {
            inner: EcEnc::new(buffer),
            finished: false,
        }
    }

    /// Returns the active packet buffer, including unused space until finished.
    pub fn buffer(&self) -> &[u8] {
        &self.inner.ctx().buffer()[..self.inner.ctx().storage as usize]
    }

    /// Returns the number of finalized range-coded bytes at the packet front.
    pub fn range_bytes(&self) -> usize {
        self.inner.ctx().offs as usize
    }

    /// Returns zero on success or -1 after overflow or an invalid patch.
    pub fn error(&self) -> i32 {
        self.inner.ctx().error
    }

    /// Returns the current range for Opus final-range checks.
    pub fn range(&self) -> u32 {
        self.inner.ctx().rng
    }

    /// Returns whole-bit usage, rounded upwards identically to libopus.
    pub fn tell(&self) -> i32 {
        ec_tell(self.inner.ctx())
    }

    /// Returns bit usage in eighths of a bit, rounded upwards.
    pub fn tell_frac(&self) -> u32 {
        ec_tell_frac(self.inner.ctx())
    }

    fn active(&self) {
        assert!(!self.finished, "the packet has already been finished");
    }

    /// Encodes cumulative frequency interval `[low, high)` out of `total`.
    pub fn encode(&mut self, low: u32, high: u32, total: u32) {
        self.active();
        assert!(low < high && high <= total && total <= self.range());
        self.inner.encode(low, high, total);
    }

    /// Encodes an interval with total frequency `1 << bits`.
    pub fn encode_bin(&mut self, low: u32, high: u32, bits: u32) {
        self.active();
        assert!(bits < 32 && low < high && high <= 1 << bits && self.range() >> bits > 0);
        self.inner.encode_bin(low, high, bits);
    }

    /// Encodes a bit whose probability of being true is `1 / (1 << logp)`.
    pub fn encode_bit_logp(&mut self, value: bool, logp: u32) {
        self.active();
        assert!(logp > 0 && logp < 32 && self.range() >> logp > 0);
        self.inner.enc_bit_logp(i32::from(value), logp);
    }

    /// Encodes an index in a non-increasing inverse CDF ending in zero.
    pub fn encode_icdf(&mut self, symbol: usize, icdf: &[u8], bits: u32) {
        self.active();
        self.inner.enc_icdf(symbol, icdf, bits);
    }

    /// Encodes an index using the 16-bit inverse-CDF variant.
    pub fn encode_icdf16(&mut self, symbol: usize, icdf: &[u16], bits: u32) {
        self.active();
        self.inner.enc_icdf16(symbol, icdf, bits);
    }

    /// Encodes an integer in `0..total`, where `total` is at least two.
    pub fn encode_uint(&mut self, value: u32, total: u32) {
        self.active();
        assert!(total > 1 && value < total);
        self.inner.enc_uint(value, total);
    }

    /// Appends one to 25 raw bits to the backwards bit stream.
    pub fn encode_bits(&mut self, value: u32, bits: u32) {
        self.active();
        assert!((1..=25).contains(&bits) && value < 1 << bits);
        self.inner.enc_bits(value, bits);
    }

    /// Replaces up to eight leading bits previously encoded with exact
    /// power-of-two probabilities. Insufficient encoded bits set `error()`.
    pub fn patch_initial_bits(&mut self, value: u32, bits: u32) {
        self.active();
        assert!(bits <= 8 && value < 1 << bits);
        self.inner.enc_patch_initial_bits(value, bits);
    }

    /// Moves the raw-byte tail to a smaller packet. Already written bytes must
    /// fit, and `size` must not exceed the current buffer length.
    pub fn shrink(&mut self, size: usize) {
        self.active();
        assert!(size <= self.inner.ctx().storage as usize);
        self.inner.enc_shrink(size as u32);
    }

    /// Terminates the packet and zeroes unused space. Check [`Self::error`]
    /// before using the returned bytes. Repeated calls leave the packet intact.
    pub fn finish(&mut self) -> &[u8] {
        if !self.finished {
            self.inner.enc_done();
            self.finished = true;
        }
        self.buffer()
    }
}

/// A range decoder borrowing a packet and padding exhausted input with zeros.
#[derive(Clone, Debug)]
pub struct RangeDecoder<'a> {
    pub(crate) inner: EcDec<'a>,
}

impl<'a> RangeDecoder<'a> {
    /// Starts decoding a packet without allocating.
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            inner: EcDec::new(buffer),
        }
    }

    /// Returns the original packet buffer.
    pub fn buffer(&self) -> &[u8] {
        self.inner.ctx().buffer()
    }

    /// Returns the number of range-coded bytes read, excluding zero padding.
    pub fn range_bytes(&self) -> usize {
        self.inner.ctx().offs as usize
    }

    /// Returns one after an out-of-range unsigned integer, otherwise zero.
    /// Input exhaustion alone does not signal a range-coder error.
    pub fn error(&self) -> i32 {
        self.inner.ctx().error
    }

    /// Returns the current range for Opus final-range checks.
    pub fn range(&self) -> u32 {
        self.inner.ctx().rng
    }

    /// Returns whole-bit usage, rounded upwards identically to libopus.
    pub fn tell(&self) -> i32 {
        ec_tell(self.inner.ctx())
    }

    /// Returns bit usage in eighths of a bit, rounded upwards.
    pub fn tell_frac(&self) -> u32 {
        ec_tell_frac(self.inner.ctx())
    }

    /// Returns a frequency in `0..total`. Follow with exactly one [`Self::update`].
    pub fn decode(&mut self, total: u32) -> u32 {
        assert!(total > 0 && total <= self.range());
        self.inner.decode(total)
    }

    /// Returns a cumulative frequency with total frequency `1 << bits`.
    /// Follow with exactly one [`Self::update`].
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        assert!(bits < 32 && self.range() >> bits > 0);
        self.inner.decode_bin(bits)
    }

    /// Advances past the interval selected using [`Self::decode`] or
    /// [`Self::decode_bin`], with the same total frequency.
    pub fn update(&mut self, low: u32, high: u32, total: u32) {
        assert!(low < high && high <= total && total <= self.range());
        self.inner.update(low, high, total);
    }

    /// Decodes a bit whose probability of being true is `1 / (1 << logp)`.
    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        assert!(logp > 0 && logp < 32 && self.range() >> logp > 0);
        self.inner.dec_bit_logp(logp) != 0
    }

    /// Decodes an index in a non-increasing inverse CDF ending in zero.
    pub fn decode_icdf(&mut self, icdf: &[u8], bits: u32) -> usize {
        self.inner.dec_icdf(icdf, bits) as usize
    }

    /// Decodes an index using the 16-bit inverse-CDF variant.
    pub fn decode_icdf16(&mut self, icdf: &[u16], bits: u32) -> usize {
        self.inner.dec_icdf16(icdf, bits) as usize
    }

    /// Decodes an integer in `0..total`, where `total` is at least two.
    /// Invalid packet values return `total - 1` and set [`Self::error`] to one.
    pub fn decode_uint(&mut self, total: u32) -> u32 {
        self.inner.dec_uint(total)
    }

    /// Reads zero to 25 raw bits from the backwards bit stream.
    pub fn decode_bits(&mut self, bits: u32) -> u32 {
        assert!(bits <= 25);
        self.inner.dec_bits(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::{RangeDecoder, RangeEncoder};

    #[test]
    fn mixed_symbols_preserve_values_ranges_and_bit_accounting() {
        let mut packet = [0xa5; 512];
        let mut encoder = RangeEncoder::new(&mut packet);
        let icdf = [201, 123, 20, 0];
        let icdf16 = [32001, 16384, 511, 0];
        let mut states = [(0, 0, 0); 160];
        for (i, state) in states.iter_mut().enumerate() {
            let value = i as u32;
            match i % 8 {
                0 => encoder.encode_uint(value * 11, 65537),
                1 => encoder.encode_bits(value & 31, 5),
                2 => encoder.encode_bit_logp(i % 3 == 0, 4),
                3 => encoder.encode_icdf(i % 4, &icdf, 8),
                4 => encoder.encode_icdf16(i % 4, &icdf16, 15),
                5 => encoder.encode(value % 13, value % 13 + 1, 13),
                6 => encoder.encode_bin(value % 8, value % 8 + 1, 3),
                _ => encoder.encode_uint(u32::MAX - value, u32::MAX),
            }
            *state = (encoder.tell(), encoder.tell_frac(), encoder.range());
        }
        let size = (encoder.tell() as usize + 7) / 8;
        encoder.shrink(size);
        encoder.finish();
        assert_eq!(encoder.error(), 0);
        let mut decoder = RangeDecoder::new(encoder.buffer());
        for (i, state) in states.iter().enumerate() {
            let value = i as u32;
            match i % 8 {
                0 => assert_eq!(decoder.decode_uint(65537), value * 11),
                1 => assert_eq!(decoder.decode_bits(5), value & 31),
                2 => assert_eq!(decoder.decode_bit_logp(4), i % 3 == 0),
                3 => assert_eq!(decoder.decode_icdf(&icdf, 8), i % 4),
                4 => assert_eq!(decoder.decode_icdf16(&icdf16, 15), i % 4),
                5 => {
                    let s = decoder.decode(13);
                    assert_eq!(s, value % 13);
                    decoder.update(s, s + 1, 13);
                }
                6 => {
                    let s = decoder.decode_bin(3);
                    assert_eq!(s, value % 8);
                    decoder.update(s, s + 1, 8);
                }
                _ => assert_eq!(decoder.decode_uint(u32::MAX), u32::MAX - value),
            }
            assert_eq!(
                (decoder.tell(), decoder.tell_frac(), decoder.range()),
                *state
            );
        }
        assert_eq!(decoder.error(), 0);
    }

    #[test]
    fn leading_bits_can_be_patched_before_and_after_normalization() {
        for additional_bits in [0, 5, 16, 32, 64] {
            let mut packet = [0; 20];
            let mut encoder = RangeEncoder::new(&mut packet);
            encoder.encode_bin(0, 1, 3);
            for _ in 0..additional_bits {
                encoder.encode_bit_logp(false, 1);
            }
            encoder.patch_initial_bits(5, 3);
            encoder.finish();
            assert_eq!(encoder.error(), 0);
            let mut decoder = RangeDecoder::new(encoder.buffer());
            assert_eq!(decoder.decode_bin(3), 5);
            decoder.update(5, 6, 8);
            for _ in 0..additional_bits {
                assert!(!decoder.decode_bit_logp(1));
            }
        }
    }

    #[test]
    fn truncated_input_and_output_overflow_are_bounded() {
        let mut decoder = RangeDecoder::new(&[]);
        assert_eq!(decoder.tell(), 1);
        assert_eq!(decoder.tell_frac(), 8);
        for _ in 0..200 {
            assert_eq!(decoder.decode_uint(256), 0);
            assert_eq!(decoder.decode_bits(25), 0);
        }
        assert_eq!(decoder.error(), 0);
        let mut buffer = [];
        let mut encoder = RangeEncoder::new(&mut buffer);
        encoder.encode_uint(123, 256);
        encoder.encode_bits(0xff, 8);
        encoder.finish();
        assert_eq!(encoder.error(), -1);
    }

    #[test]
    fn insufficient_initial_bits_set_a_sticky_error() {
        let mut packet = [0; 16];
        let mut encoder = RangeEncoder::new(&mut packet);
        encoder.patch_initial_bits(1, 1);
        assert_eq!(encoder.error(), -1);
        encoder.encode_uint(1, 8);
        encoder.finish();
        assert_eq!(encoder.error(), -1);
    }

    #[test]
    fn empty_stream_and_finish_are_deterministic() {
        let mut packet = [0xff; 4];
        let mut encoder = RangeEncoder::new(&mut packet);
        assert_eq!(encoder.tell(), 1);
        assert_eq!(encoder.tell_frac(), 8);
        assert_eq!(encoder.finish(), [0, 0, 0, 0]);
        assert_eq!(encoder.finish(), [0, 0, 0, 0]);
        assert_eq!(encoder.error(), 0);
    }
}
