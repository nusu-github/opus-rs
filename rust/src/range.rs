use alloc::vec::Vec;

use crate::celt::{EcEnc, EcEncSnapshot, ec_tell};
use crate::silk::icdf::ICDFContext;

/// The shared Opus entropy decoder used by both SILK and CELT.
pub use crate::entropy::RangeDecoder;

impl<'a> RangeDecoder<'a> {
    /// Starts decoding a packet using the canonical Opus range coder.
    pub fn init(buf: &'a [u8]) -> Self {
        Self::new(buf)
    }

    /// Decodes a bit with a probability of `1 / (1 << logp)` of being one.
    pub fn decode_symbol_logp(&mut self, logp: usize) -> u32 {
        self.inner.dec_bit_logp(logp as u32) as u32
    }

    /// Decodes a symbol from an ascending cumulative distribution.
    pub fn decode_symbol_with_icdf(&mut self, icdf_ctx: ICDFContext) -> u32 {
        let ICDFContext { total, dist_table } = icdf_ctx;
        let symbol = self.inner.decode(total);
        let index = dist_table
            .iter()
            .position(|&v| v as u32 > symbol)
            .expect("cumulative distribution does not cover the symbol");
        let high = dist_table[index] as u32;
        let low = if index == 0 {
            0
        } else {
            dist_table[index - 1] as u32
        };
        self.inner.update(low, high, total);
        index as u32
    }

    /// Returns the final arithmetic range used for Opus diagnostics.
    pub fn range_final(&self) -> u32 {
        self.range()
    }

    #[cfg(test)]
    pub(crate) fn from_test_state(
        buf: &'a [u8],
        bits_read: usize,
        total_bits: i32,
        range: u32,
        value: u32,
    ) -> Self {
        let mut decoder = Self::new(buf);
        let ctx = decoder.inner.ctx_mut();
        let offs = bits_read.div_ceil(8).min(buf.len());
        ctx.offs = offs as u32;
        ctx.rem = if offs == 0 {
            0
        } else {
            i32::from(buf[offs - 1])
        };
        ctx.nbits_total = total_bits;
        ctx.rng = range;
        ctx.val = value;
        decoder
    }
}

const RANGE_ENCODER_STORAGE_BYTES: usize = 1275;

#[derive(Debug)]
pub struct RangeEncoder {
    encoder: EcEnc<'static>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self::with_capacity(RANGE_ENCODER_STORAGE_BYTES)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let encoder = EcEnc::with_capacity(capacity);
        Self { encoder }
    }

    fn from_snapshot(snapshot: &EcEncSnapshot) -> Self {
        let mut encoder = EcEnc::with_capacity(snapshot.buffer_len());
        snapshot.restore(&mut encoder);
        Self { encoder }
    }

    /// Returns the number of whole bits emitted so far.
    #[must_use]
    pub fn tell(&self) -> i32 {
        ec_tell(self.encoder.ctx()) as i32
    }

    /// Returns the final range-coder state (`rng`) used by Opus for diagnostics.
    #[must_use]
    pub fn range_final(&self) -> u32 {
        self.encoder.ctx().rng
    }

    pub fn encode_bin(&mut self, low: u32, high: u32, bits: u32) {
        self.encoder.encode_bin(low, high, bits);
    }

    pub fn encode_symbol_with_icdf(&mut self, symbol: usize, icdf_ctx: ICDFContext) {
        let ICDFContext { total, dist_table } = icdf_ctx;
        debug_assert!(symbol < dist_table.len(), "symbol index out of bounds");
        let high = dist_table[symbol] as u32;
        let low = if symbol > 0 {
            dist_table[symbol - 1] as u32
        } else {
            0
        };
        self.encoder.encode(low, high, total);
    }

    pub fn encode_icdf16(&mut self, symbol: usize, icdf: &[u16], ftb: u32) {
        debug_assert!(symbol < icdf.len(), "symbol index out of bounds");
        self.encoder.enc_icdf16(symbol, icdf, ftb);
    }

    /// Encodes a symbol using the compact 8-bit cumulative distribution format
    /// used by SILK's shell coder and related tables.
    pub fn encode_icdf(&mut self, symbol: usize, icdf: &[u8], ftb: u32) {
        debug_assert!(symbol < icdf.len(), "symbol index out of bounds");
        self.encoder.enc_icdf(symbol, icdf, ftb);
    }

    pub(crate) fn encode_bit_logp(&mut self, value: i32, logp: u32) {
        self.encoder.enc_bit_logp(value, logp);
    }

    pub(crate) fn encode_uint(&mut self, value: u32, total: u32) {
        self.encoder.enc_uint(value, total);
    }

    pub(crate) fn shrink(&mut self, size: usize) {
        self.encoder.enc_shrink(size as u32);
    }

    pub(crate) fn encoder_mut(&mut self) -> &mut EcEnc<'static> {
        &mut self.encoder
    }

    /// Patches bits at the start of the encoded stream.
    ///
    /// Mirrors `ec_enc_patch_initial_bits`, allowing callers to reserve space
    /// for header bits and fill them in once the final values are known.
    pub fn patch_initial_bits(&mut self, value: u32, nbits: u32) {
        if nbits == 0 {
            return;
        }
        self.encoder.enc_patch_initial_bits(value, nbits);
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Raw bits are written at the packet's back. Compact before flushing
        // their partial byte so that a later shrink cannot discard that byte.
        let has_raw_bits = self.encoder.ctx().end_offs > 0 || self.encoder.ctx().nend_bits > 0;
        if has_raw_bits {
            let bytes = (ec_tell(self.encoder.ctx()) as u32).div_ceil(8);
            self.encoder
                .enc_shrink(bytes.min(self.encoder.ctx().storage));
        }
        self.encoder.enc_done();
        let size = if has_raw_bits {
            self.encoder.ctx().storage
        } else {
            self.encoder.ctx().offs + self.encoder.ctx().end_offs
        };
        if size < self.encoder.ctx().storage {
            self.encoder.enc_shrink(size);
        }
        let size = size as usize;
        let buffer = self.encoder.ctx().buffer();
        #[cfg(test)]
        range_done_trace::maybe_dump(&buffer[..size]);
        buffer[..size].to_vec()
    }

    pub(crate) fn finish_without_done(self) -> Vec<u8> {
        let size = self.encoder.ctx().storage as usize;
        let buffer = self.encoder.ctx().buffer();
        buffer[..size].to_vec()
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RangeEncoder {
    fn clone(&self) -> Self {
        let snapshot = EcEncSnapshot::capture(&self.encoder);
        Self::from_snapshot(&snapshot)
    }
}

#[cfg(test)]
mod range_done_trace {
    extern crate std;

    use core::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
    use std::env;
    use std::sync::OnceLock;

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);
    static CURRENT_FRAME: AtomicIsize = AtomicIsize::new(-1);

    pub(crate) fn begin_frame() -> Option<usize> {
        if config().is_some() {
            Some(FRAME_INDEX.fetch_add(1, Ordering::Relaxed))
        } else {
            None
        }
    }

    pub(crate) fn set_frame(frame_idx: usize) {
        CURRENT_FRAME.store(frame_idx as isize, Ordering::Relaxed);
    }

    fn current_frame() -> Option<usize> {
        let value = CURRENT_FRAME.load(Ordering::Relaxed);
        if value >= 0 {
            Some(value as usize)
        } else {
            None
        }
    }

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = match env::var("OPUS_TRACE_RANGE_DONE") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                if !enabled {
                    return None;
                }
                let frame = env::var("OPUS_TRACE_RANGE_DONE_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                Some(TraceConfig { frame })
            })
            .as_ref()
    }

    pub(crate) fn maybe_dump(buffer: &[u8]) {
        let cfg = match config() {
            Some(cfg) => cfg,
            None => return,
        };
        let frame_idx = match current_frame() {
            Some(frame_idx) => frame_idx,
            None => return,
        };
        if cfg.frame.map_or(true, |frame| frame == frame_idx) {
            crate::test_trace::trace_println!("opus_range_done[{frame_idx}].len={}", buffer.len());
            for (idx, value) in buffer.iter().enumerate() {
                crate::test_trace::trace_println!(
                    "opus_range_done[{frame_idx}].byte[{idx}]=0x{value:02x}"
                );
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn begin_range_done_trace_frame() -> Option<usize> {
    range_done_trace::begin_frame()
}

#[cfg(test)]
pub(crate) fn set_range_done_trace_frame(frame_idx: usize) {
    range_done_trace::set_frame(frame_idx);
    crate::celt::set_enc_done_trace_frame(frame_idx);
}

// taken from <https://github.com/pion/opus/blob/e8536fe9e4ca2181db7d808e35d50b2c0400ceb1/internal/rangecoding/decoder_test.go>
#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt::EcDec;
    use crate::icdf;
    use crate::silk::SilkRangeDecoder;
    use crate::silk::tables_other::SILK_UNIFORM4_ICDF;
    use crate::silk::tables_pulses_per_block::{
        SILK_SHELL_CODE_TABLE_OFFSETS, SILK_SHELL_CODE_TABLE0,
    };

    #[test]
    fn compact_packet_preserves_partial_raw_tail() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_uint(65536, 65537);
        encoder.encode_uint(u32::MAX - 1, u32::MAX);
        let packet = encoder.finish();
        // Pinned C oracle: size 7; uint 65536 65537; uint 4294967294 4294967295.
        assert_eq!(packet, [0xff, 0xff, 0x01, 0xff, 0xff, 0xfc, 0x00]);
        let mut decoder = RangeDecoder::new(&packet);
        assert_eq!(decoder.decode_uint(65537), 65536);
        assert_eq!(decoder.decode_uint(u32::MAX), u32::MAX - 1);
        assert_eq!(decoder.error(), 0);
    }

    const SILK_FRAME_TYPE_INACTIVE: ICDFContext = icdf!(256; 26, 256);

    const SILK_GAIN_HIGH_BITS: [ICDFContext; 3] = [
        icdf!(256; 32, 144, 212, 241, 253, 254, 255, 256),
        icdf!(256; 2, 19, 64, 124, 186, 233, 252, 256),
        icdf!(256; 1, 4, 30, 101, 195, 245, 254, 256),
    ];

    const SILK_GAIN_LOW_BITS: ICDFContext = icdf!(256; 32, 64, 96, 128, 160, 192, 224, 256);

    const SILK_GAIN_DELTA: ICDFContext = icdf!(
        256; 6, 11, 22, 53, 185, 206, 214, 218, 221, 223, 225, 227, 228, 229, 230, 231, 232, 233,
        234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
        252, 253, 254, 255, 256
    );

    const SILK_LSF_S1: [[ICDFContext; 2]; 2] = [
        [
            icdf!(
                256; 44, 78, 108, 127, 148, 160, 171, 174, 177, 179, 195, 197, 199, 200, 205, 207,
                208, 211, 214, 215, 216, 218, 220, 222, 225, 226, 235, 244, 246, 253, 255, 256
            ),
            icdf!(
                256; 1, 11, 12, 20, 23, 31, 39, 53, 66, 80, 81, 95, 107, 120, 131, 142, 154, 165,
                175, 185, 196, 204, 213, 221, 228, 236, 237, 238, 244, 245, 251, 256
            ),
        ],
        [
            icdf!(
                256; 31, 52, 55, 72, 73, 81, 98, 102, 103, 121, 137, 141, 143, 146, 147, 157, 158,
                161, 177, 188, 204, 206, 208, 211, 213, 224, 225, 229, 238, 246, 253, 256
            ),
            icdf!(
                256; 1, 5, 21, 26, 44, 55, 60, 74, 89, 90, 93, 105, 118, 132, 146, 152, 166, 178,
                180, 186, 187, 199, 211, 222, 232, 235, 245, 250, 251, 252, 253, 256
            ),
        ],
    ];

    const SILK_LSF_S2: [ICDFContext; 16] = [
        icdf!(256; 1, 2, 3, 18, 242, 253, 254, 255, 256),
        icdf!(256; 1, 2, 4, 38, 221, 253, 254, 255, 256),
        icdf!(256; 1, 2, 6, 48, 197, 252, 254, 255, 256),
        icdf!(256; 1, 2, 10, 62, 185, 246, 254, 255, 256),
        icdf!(256; 1, 4, 20, 73, 174, 248, 254, 255, 256),
        icdf!(256; 1, 4, 21, 76, 166, 239, 254, 255, 256),
        icdf!(256; 1, 8, 32, 85, 159, 226, 252, 255, 256),
        icdf!(256; 1, 2, 20, 83, 161, 219, 249, 255, 256),
        icdf!(256; 1, 2, 3, 12, 244, 253, 254, 255, 256),
        icdf!(256; 1, 2, 4, 32, 218, 253, 254, 255, 256),
        icdf!(256; 1, 2, 5, 47, 199, 252, 254, 255, 256),
        icdf!(256; 1, 2, 12, 61, 187, 252, 254, 255, 256),
        icdf!(256; 1, 5, 24, 72, 172, 249, 254, 255, 256),
        icdf!(256; 1, 2, 16, 70, 170, 242, 254, 255, 256),
        icdf!(256; 1, 2, 17, 78, 165, 226, 251, 255, 256),
        icdf!(256; 1, 8, 29, 79, 156, 237, 254, 255, 256),
    ];

    const SILK_LSF_INTERPOLATION_OFFSET: ICDFContext = icdf!(256; 13, 35, 64, 75, 256);

    const SILK_LCG_SEED: ICDFContext = icdf!(256; 64, 128, 192, 256);

    const SILK_EXC_RATE: [ICDFContext; 2] = [
        icdf!(256; 15, 66, 78, 124, 169, 182, 215, 242, 256),
        icdf!(256; 33, 63, 99, 116, 150, 199, 217, 238, 256),
    ];

    const SILK_PULSE_COUNT: [ICDFContext; 11] = [
        icdf!(
            256; 131, 205, 230, 238, 241, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
            255, 256
        ),
        icdf!(
            256; 58, 151, 211, 234, 241, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
            255, 256
        ),
        icdf!(
            256; 43, 94, 140, 173, 197, 213, 224, 232, 238, 241, 244, 247, 249, 250, 251, 253, 254,
            256
        ),
        icdf!(
            256; 17, 69, 140, 197, 228, 240, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
            256
        ),
        icdf!(
            256; 6, 27, 68, 121, 170, 205, 226, 237, 243, 246, 248, 250, 251, 252, 253, 254, 255,
            256
        ),
        icdf!(
            256; 7, 21, 43, 71, 100, 128, 153, 173, 190, 203, 214, 223, 230, 235, 239, 243, 246,
            256
        ),
        icdf!(
            256; 2, 7, 21, 50, 92, 138, 179, 210, 229, 240, 246, 249, 251, 252, 253, 254, 255, 256
        ),
        icdf!(256; 1, 3, 7, 17, 36, 65, 100, 137, 171, 199, 219, 233, 241, 246, 250, 252, 254, 256),
        icdf!(256; 1, 3, 5, 10, 19, 33, 53, 77, 104, 132, 158, 181, 201, 216, 227, 235, 241, 256),
        icdf!(256; 1, 2, 3, 9, 36, 94, 150, 189, 214, 228, 238, 244, 247, 250, 252, 253, 254, 256),
        icdf!(
            256; 2, 3, 9, 36, 94, 150, 189, 214, 228, 238, 244, 247, 250, 252, 253, 254, 256, 256
        ),
    ];

    #[test]
    fn decoder() {
        let mut decoder = RangeDecoder::init(&[0x0b, 0xe4, 0xc1, 0x36, 0xec, 0xc5, 0x80]);

        assert_eq!(decoder.decode_symbol_logp(0x1), 0);
        assert_eq!(decoder.decode_symbol_logp(0x1), 0);

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_FRAME_TYPE_INACTIVE), 1);

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_HIGH_BITS[0]), 0);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_LOW_BITS), 6);

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_DELTA), 0);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_DELTA), 3);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_DELTA), 4);

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_LSF_S1[1][0]), 9);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_LSF_S2[10]), 5);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_LSF_S2[9]), 4);

        for _i in 0..14 {
            assert_eq!(decoder.decode_symbol_with_icdf(SILK_LSF_S2[8]), 4);
        }

        assert_eq!(
            decoder.decode_symbol_with_icdf(SILK_LSF_INTERPOLATION_OFFSET),
            4
        );

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_LCG_SEED), 2);

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_EXC_RATE[0]), 0);

        for _i in 0..20 {
            assert_eq!(decoder.decode_symbol_with_icdf(SILK_PULSE_COUNT[0]), 0);
        }
    }

    #[test]
    fn encodes_symbols_with_icdf_context() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_symbol_with_icdf(1, SILK_FRAME_TYPE_INACTIVE);
        encoder.encode_symbol_with_icdf(2, SILK_GAIN_HIGH_BITS[1]);
        encoder.encode_symbol_with_icdf(6, SILK_GAIN_LOW_BITS);
        encoder.encode_symbol_with_icdf(0, SILK_GAIN_DELTA);

        let mut storage = encoder.finish();
        let mut decoder = EcDec::new(storage.as_mut_slice());

        assert_eq!(decoder.decode_symbol_with_icdf(SILK_FRAME_TYPE_INACTIVE), 1);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_HIGH_BITS[1]), 2);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_LOW_BITS), 6);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_GAIN_DELTA), 0);
    }

    #[test]
    fn encode_decode_with_u8_icdf_roundtrip() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_icdf(2, &SILK_UNIFORM4_ICDF, 8);
        encoder.encode_icdf(0, &SILK_UNIFORM4_ICDF, 8);

        let mut storage = encoder.finish();
        let mut decoder = EcDec::new(storage.as_mut_slice());

        assert_eq!(decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8), 2);
        assert_eq!(decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8), 0);
    }

    #[test]
    fn encode_decode_with_shell_table_slice() {
        let start = usize::from(SILK_SHELL_CODE_TABLE_OFFSETS[4]);
        let end = usize::from(SILK_SHELL_CODE_TABLE_OFFSETS[5]);
        let icdf = &SILK_SHELL_CODE_TABLE0[start..end];

        let mut encoder = RangeEncoder::new();
        encoder.encode_icdf(2, icdf, 8);
        encoder.encode_icdf(1, icdf, 8);

        let mut storage = encoder.finish();
        let mut decoder = EcDec::new(storage.as_mut_slice());

        assert_eq!(decoder.decode_icdf(icdf, 8), 2);
        assert_eq!(decoder.decode_icdf(icdf, 8), 1);
    }

    #[test]
    fn patch_initial_bits_round_trips_logp_flags() {
        for header_bits in 1u32..=8 {
            let value = (1u32 << header_bits).wrapping_sub(1) ^ (header_bits % 3);
            let mut encoder = RangeEncoder::new();
            let mut icdf = [0u8; 2];
            icdf[0] = (256u16 - (256u16 >> header_bits)) as u8;
            encoder.encode_icdf(0, &icdf, 8);
            encoder.patch_initial_bits(value, header_bits);

            let mut storage = encoder.finish();
            let mut decoder = EcDec::new(storage.as_mut_slice());
            let mut decoded = 0u32;
            for _ in 0..header_bits {
                decoded = (decoded << 1) | decoder.decode_symbol_logp(1);
            }
            let mask = (1u32 << header_bits) - 1;
            assert_eq!(decoded, value & mask);
        }
    }

    #[test]
    fn range_final_matches_after_roundtrip_decode() {
        let mut encoder = RangeEncoder::new();
        encoder.encode_icdf(2, &SILK_UNIFORM4_ICDF, 8);
        encoder.encode_symbol_with_icdf(1, SILK_FRAME_TYPE_INACTIVE);
        encoder.encode_icdf(0, &SILK_UNIFORM4_ICDF, 8);
        let range_final = encoder.range_final();

        let mut storage = encoder.finish();
        let mut decoder = EcDec::new(storage.as_mut_slice());
        assert_eq!(decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8), 2);
        assert_eq!(decoder.decode_symbol_with_icdf(SILK_FRAME_TYPE_INACTIVE), 1);
        assert_eq!(decoder.decode_icdf(&SILK_UNIFORM4_ICDF, 8), 0);

        assert_eq!(decoder.range_final(), range_final);
    }
}
