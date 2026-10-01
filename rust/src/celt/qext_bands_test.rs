//! Independent scalar-C fixtures for complete scalable band coding.
use super::*;
use crate::celt::entcode::{ec_tell, ec_tell_frac};
use crate::celt::qext_vq::Scalar;

struct Input<'a>(&'a [u8]);
impl<'a> Input<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    fn word(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn words(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.word()).collect()
    }
}
fn scalar(word: u32) -> Scalar {
    #[cfg(feature = "fixed_point")]
    {
        word as i32
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        f32::from_bits(word)
    }
}
fn bits(value: Scalar) -> u32 {
    #[cfg(feature = "fixed_point")]
    {
        value as u32
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        value.to_bits()
    }
}
fn coder_state(
    base: &crate::celt::entcode::EcCtx<'_>,
    ext: &crate::celt::entcode::EcCtx<'_>,
    seed: u32,
) -> [u32; 7] {
    [
        base.rng,
        ext.rng,
        ec_tell(base) as u32,
        ec_tell(ext) as u32,
        ec_tell_frac(base),
        ec_tell_frac(ext),
        seed,
    ]
}
#[test]
fn qext_bands_match_pinned_scalar_c() {
    #[cfg(not(feature = "fixed_point"))]
    let bytes = include_bytes!("../../tests/fixtures/reference/qext-bands.bin");
    #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
    let bytes = include_bytes!("../../tests/fixtures/reference/fixed-qext-bands.bin");
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    let bytes = include_bytes!("../../tests/fixtures/reference/fixed-res24-qext-bands.bin");
    let mut input = Input(bytes);
    assert_eq!(input.word(), 0x51424e44);
    let cases = input.word();
    assert_eq!(cases, 288);
    let mut packet_failures = Vec::new();
    for _ in 0..cases {
        let h = input.words(17);
        let case = h[0];
        let rate = h[1];
        let lm = h[2] as i32;
        let channels = h[3] as usize;
        let short = h[4] != 0;
        let extra_bands = h[6] != 0;
        let spread = h[7] as i32;
        let dual = h[8] != 0;
        let intensity = h[9] as usize;
        let complexity = h[10] as i32;
        let start = h[11] as usize;
        let end = h[12] as usize;
        let bands = h[13] as usize;
        let count = h[14] as usize;
        let base_cap = h[15] as usize;
        let ext_cap = h[16] as usize;
        let mut pulses = vec![0; bands];
        let mut extra = vec![0; bands];
        let mut caps = vec![0; bands];
        let mut tf = vec![0; bands];
        for i in 0..bands {
            pulses[i] = input.word() as i32;
            extra[i] = input.word() as i32;
            caps[i] = input.word() as i32;
            tf[i] = input.word() as i32;
        }
        let source = input.words(count * channels);
        let encoded = input.words(count * channels);
        let enc_state = input.words(7);
        let enc_masks = input.words(bands * channels);
        let decoded = input.words(count * channels);
        let dec_state = input.words(7);
        let dec_masks = input.words(bands * channels);
        let expected_base = input.bytes(base_cap);
        let expected_ext = input.bytes(ext_cap);
        let base_mode = if rate == 96000 {
            crate::celt::modes::canonical_mode_96k().unwrap()
        } else {
            crate::celt::canonical_mode().unwrap()
        };
        let extra_mode = crate::celt::modes::compute_qext_mode(base_mode);
        let mode = if extra_bands { &extra_mode } else { base_mode };
        #[cfg(feature = "fixed_point")]
        let energies = vec![1 << 24; bands * channels];
        #[cfg(not(feature = "fixed_point"))]
        let energies = vec![1.0; bands * channels];
        let cap = if extra_bands {
            &[][..]
        } else {
            caps.as_slice()
        };
        let mut output: Vec<_> = source.iter().map(|&word| scalar(word)).collect();
        let mut packet = vec![0; base_cap];
        let mut extension = vec![0; ext_cap];
        let mut encoder = EcEnc::new(&mut packet);
        let mut ext_encoder = EcEnc::new(&mut extension);
        let mut seed = 0x12345678;
        let mut masks = vec![0; bands * channels];
        let (x, y) = output.split_at_mut(count);
        quant_all_bands(
            true,
            mode,
            start,
            end,
            x,
            if channels == 2 { Some(y) } else { None },
            &mut masks,
            &energies,
            &pulses,
            short,
            spread,
            dual,
            intensity,
            &tf,
            (base_cap as i32) * 64,
            0,
            &mut BandCodingState::Encoder(&mut encoder),
            lm,
            end,
            &mut seed,
            complexity,
            0,
            false,
            &mut BandCodingState::Encoder(&mut ext_encoder),
            &extra,
            extra_bands,
            cap,
        );
        assert_eq!(
            coder_state(encoder.ctx(), ext_encoder.ctx(), seed).as_slice(),
            enc_state,
            "case {case} {h:?} encoder state"
        );
        assert_eq!(
            output.iter().map(|&x| bits(x)).collect::<Vec<_>>(),
            encoded,
            "case {case} encoder coefficients"
        );
        assert_eq!(
            masks.iter().map(|&x| u32::from(x)).collect::<Vec<_>>(),
            enc_masks,
            "case {case} encoder masks"
        );
        encoder.enc_done();
        ext_encoder.enc_done();
        drop(encoder);
        drop(ext_encoder);

        let mut decoder = EcDec::new(expected_base);
        let mut ext_decoder = EcDec::new(expected_ext);
        let mut output = vec![0 as Scalar; count * channels];
        let mut masks = vec![0; bands * channels];
        let mut seed = 0x12345678;
        let (x, y) = output.split_at_mut(count);
        quant_all_bands(
            false,
            mode,
            start,
            end,
            x,
            if channels == 2 { Some(y) } else { None },
            &mut masks,
            &energies,
            &pulses,
            short,
            spread,
            dual,
            intensity,
            &tf,
            (base_cap as i32) * 64,
            0,
            &mut BandCodingState::Decoder(&mut decoder),
            lm,
            end,
            &mut seed,
            complexity,
            0,
            false,
            &mut BandCodingState::Decoder(&mut ext_decoder),
            &extra,
            extra_bands,
            cap,
        );
        assert_eq!(
            coder_state(decoder.ctx(), ext_decoder.ctx(), seed).as_slice(),
            dec_state,
            "case {case} decoder state"
        );
        assert_eq!(
            output.iter().map(|&x| bits(x)).collect::<Vec<_>>(),
            decoded,
            "case {case} decoder coefficients"
        );
        assert_eq!(
            masks.iter().map(|&x| u32::from(x)).collect::<Vec<_>>(),
            dec_masks,
            "case {case} decoder masks"
        );
        if packet != expected_base || extension != expected_ext {
            packet_failures.push(case);
        }
    }
    assert!(input.0.is_empty());
    assert!(
        packet_failures.is_empty(),
        "packet mismatches in cases {packet_failures:?}"
    );
}
