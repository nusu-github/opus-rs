//! Exact C regression for fixed CELT stereo packets decoded to mono.
#![cfg(feature = "fixed_point")]

use opus_rs::c_style_api::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decode_float, opus_decoder_create, opus_decoder_ctl,
};
use sha2::{Digest, Sha256};

const PACKETS: &str = include_str!("fixtures/reference/rfc8251-downmix-packets.tsv");

fn golden() -> &'static str {
    match (
        cfg!(feature = "enable_res24"),
        cfg!(feature = "enable_qext"),
        cfg!(feature = "pfa"),
    ) {
        (false, false, false) => include_str!("fixtures/reference/fixed-downmix.tsv"),
        (false, false, true) => include_str!("fixtures/reference/fixed-pfa-downmix.tsv"),
        (false, true, false) => include_str!("fixtures/reference/fixed-qext-downmix.tsv"),
        (false, true, true) => include_str!("fixtures/reference/fixed-qext-pfa-downmix.tsv"),
        (true, false, false) => include_str!("fixtures/reference/fixed-res24-downmix.tsv"),
        (true, false, true) => include_str!("fixtures/reference/fixed-res24-pfa-downmix.tsv"),
        (true, true, false) => include_str!("fixtures/reference/fixed-res24-qext-downmix.tsv"),
        (true, true, true) => include_str!("fixtures/reference/fixed-res24-qext-pfa-downmix.tsv"),
    }
}

fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn stereo_packets_decode_to_mono_like_pinned_c() {
    let mut cases = 0;
    for line in golden().lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let packet = fields
            .get(7)
            .map(|packet| unhex(packet))
            .unwrap_or_else(|| {
                PACKETS
                    .lines()
                    .find_map(|line| {
                        let (name, packet) = line.split_once('\t').unwrap();
                        (name == fields[0]).then(|| unhex(packet))
                    })
                    .unwrap()
            });
        let rate: i32 = fields[1].parse().unwrap();
        let maximum = rate as usize * 120 / 1000;
        let expected_samples: usize = fields[2].parse().unwrap();
        let mut decoder = opus_decoder_create(rate, 1).unwrap();
        let mut float_decoder = opus_decoder_create(rate, 1).unwrap();
        let mut pcm = vec![0i16; maximum];
        let mut float_pcm = vec![0.0f32; maximum];
        let samples = opus_decode(
            &mut decoder,
            Some(&packet),
            packet.len(),
            &mut pcm,
            maximum,
            false,
        )
        .unwrap();
        let float_samples = opus_decode_float(
            &mut float_decoder,
            Some(&packet),
            packet.len(),
            &mut float_pcm,
            maximum,
            false,
        )
        .unwrap();
        assert_eq!(samples, expected_samples);
        assert_eq!(float_samples, expected_samples);
        let mut integer_hash = Sha256::new();
        let mut float_hash = Sha256::new();
        for value in &pcm[..samples] {
            integer_hash.update(value.to_le_bytes());
        }
        for value in &float_pcm[..samples] {
            float_hash.update(value.to_bits().to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", integer_hash.finalize()),
            fields[5],
            "{} at {rate}: integer PCM",
            fields[0]
        );
        assert_eq!(
            format!("{:x}", float_hash.finalize()),
            fields[6],
            "{} at {rate}: float PCM",
            fields[0]
        );
        let (mut range, mut float_range) = (0, 0);
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        opus_decoder_ctl(
            &mut float_decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut float_range),
        )
        .unwrap();
        assert_eq!(range, fields[3].parse::<u32>().unwrap());
        assert_eq!(float_range, fields[4].parse::<u32>().unwrap());
        cases += 1;
    }
    assert_eq!(cases, if cfg!(feature = "enable_qext") { 6 } else { 4 });
}
