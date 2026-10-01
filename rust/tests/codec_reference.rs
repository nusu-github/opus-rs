//! Exact independently generated C vectors; no C library or FFI at test runtime.
#![cfg(not(feature = "fixed_point"))]

use opus_rs::c_style_api::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decode_float, opus_decoder_create, opus_decoder_ctl,
};
use opus_rs::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};

const GOLDEN: &str = include_str!("fixtures/reference/celt-stereo-20ms.tsv");
const PCM: &[u8] = include_bytes!("fixtures/reference/celt-stereo-20ms.pcm");

fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn equal<T: PartialEq + std::fmt::Debug>(frame: usize, field: &str, actual: &[T], expected: &[T]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "frame {frame}: {field} length"
    );
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "frame {frame}: {field}[{index}], Rust {:?}, C {:?}",
            actual[index], expected[index]
        );
    }
}

#[test]
fn c_reference_celt_encode_packets_and_ranges() {
    let mut encoder = opus_encoder_create(48_000, 2, 2049).unwrap();
    for request in [
        OpusEncoderCtlRequest::SetBitrate(96_000),
        OpusEncoderCtlRequest::SetVbr(false),
        OpusEncoderCtlRequest::SetComplexity(10),
        OpusEncoderCtlRequest::SetForceMode(1002),
    ] {
        opus_encoder_ctl(&mut encoder, request).unwrap();
    }
    let mut output = vec![0; 1276 * 6];
    for (frame, (input, line)) in PCM
        .chunks_exact(960 * 2 * 2)
        .zip(GOLDEN.lines())
        .enumerate()
    {
        let expected: Vec<_> = line.split('\t').collect();
        let input: Vec<_> = input
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let len = opus_encode(&mut encoder, &input, 960, &mut output).unwrap();
        equal(frame, "packet", &output[..len], &unhex(expected[7]));
        let mut range = 0;
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        assert_eq!(
            range,
            expected[4].parse::<u32>().unwrap(),
            "frame {frame}: encoder range"
        );
    }
}

#[test]
fn c_reference_celt_decode_integer_pcm_and_ranges() {
    let mut decoder = opus_decoder_create(48_000, 2).unwrap();
    let mut output = vec![0; 5760 * 2];
    for (frame, line) in GOLDEN.lines().enumerate() {
        let expected: Vec<_> = line.split('\t').collect();
        let packet = unhex(expected[7]);
        let samples = opus_decode(
            &mut decoder,
            Some(&packet),
            packet.len(),
            &mut output,
            5760,
            false,
        )
        .unwrap();
        assert_eq!(
            samples,
            expected[3].parse::<usize>().unwrap(),
            "frame {frame}: sample count"
        );
        let pcm: Vec<_> = unhex(expected[8])
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        equal(frame, "integer PCM", &output[..samples * 2], &pcm);
        let mut range = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        assert_eq!(
            range,
            expected[5].parse::<u32>().unwrap(),
            "frame {frame}: decoder range"
        );
    }
}

#[test]
fn c_reference_celt_decode_float_bits_and_ranges() {
    let mut decoder = opus_decoder_create(48_000, 2).unwrap();
    let mut output = vec![0.0; 5760 * 2];
    for (frame, line) in GOLDEN.lines().enumerate() {
        let expected: Vec<_> = line.split('\t').collect();
        let packet = unhex(expected[7]);
        let samples = opus_decode_float(
            &mut decoder,
            Some(&packet),
            packet.len(),
            &mut output,
            5760,
            false,
        )
        .unwrap();
        assert_eq!(
            samples,
            expected[3].parse::<usize>().unwrap(),
            "frame {frame}: sample count"
        );
        let bits: Vec<_> = expected[9]
            .as_bytes()
            .chunks_exact(8)
            .map(|b| u32::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let actual: Vec<_> = output[..samples * 2]
            .iter()
            .map(|value| value.to_bits())
            .collect();
        equal(frame, "float PCM bits", &actual, &bits);
        let mut range = 0;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        assert_eq!(
            range,
            expected[6].parse::<u32>().unwrap(),
            "frame {frame}: float decoder range"
        );
    }
}

fn check_silk_decode(rate: i32, channels: i32, commands: &str, golden: &str) {
    let mut decoder = opus_decoder_create(rate, channels).unwrap();
    let mut float_decoder = opus_decoder_create(rate, channels).unwrap();
    let mut pcm = vec![0; rate as usize * 120 / 1000 * channels as usize];
    let mut float_pcm = vec![0.0; pcm.len()];
    for (frame, (command, line)) in commands.lines().zip(golden.lines()).enumerate() {
        let command: Vec<_> = command.split_whitespace().collect();
        let size = command[0].parse().unwrap();
        let packet = (command[2] != "-").then(|| unhex(command[2]));
        let packet = packet.as_deref();
        let len = packet.map_or(0, <[u8]>::len);
        let samples = opus_decode(&mut decoder, packet, len, &mut pcm, size, false).unwrap();
        let float_samples =
            opus_decode_float(&mut float_decoder, packet, len, &mut float_pcm, size, false)
                .unwrap();
        let expected: Vec<_> = line.split('\t').collect();
        assert_eq!(samples, expected[2].parse::<usize>().unwrap());
        assert_eq!(float_samples, samples);
        let expected_pcm: Vec<_> = unhex(expected[5])
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        equal(
            frame,
            "SILK integer PCM",
            &pcm[..samples * channels as usize],
            &expected_pcm,
        );
        let expected_bits: Vec<_> = expected[6]
            .as_bytes()
            .chunks_exact(8)
            .map(|b| u32::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let actual: Vec<_> = float_pcm[..samples * channels as usize]
            .iter()
            .map(|value| value.to_bits())
            .collect();
        equal(frame, "SILK float PCM bits", &actual, &expected_bits);
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
        assert_eq!(range, expected[3].parse::<u32>().unwrap());
        assert_eq!(float_range, expected[4].parse::<u32>().unwrap());
    }
}

#[test]
fn c_reference_silk_narrowband_normal() {
    check_silk_decode(
        8000,
        1,
        include_str!("fixtures/reference/silk-8000-1-normal.packets"),
        include_str!("fixtures/reference/silk-8000-1-normal.tsv"),
    );
}

#[test]
fn c_reference_silk_narrowband_loss_and_recovery() {
    check_silk_decode(
        8000,
        1,
        include_str!("fixtures/reference/silk-8000-1-loss.packets"),
        include_str!("fixtures/reference/silk-8000-1-loss.tsv"),
    );
}

#[test]
fn c_reference_silk_wideband_normal() {
    check_silk_decode(
        16000,
        2,
        include_str!("fixtures/reference/silk-16000-2-normal.packets"),
        include_str!("fixtures/reference/silk-16000-2-normal.tsv"),
    );
}

#[test]
fn c_reference_silk_wideband_loss_and_recovery() {
    check_silk_decode(
        16000,
        2,
        include_str!("fixtures/reference/silk-16000-2-loss.packets"),
        include_str!("fixtures/reference/silk-16000-2-loss.tsv"),
    );
}

#[test]
fn c_reference_silk_channel_transitions_mono_output() {
    check_silk_decode(
        48000,
        1,
        include_str!("fixtures/reference/silk-channel-transitions.packets"),
        include_str!("fixtures/reference/silk-channel-transitions-1.tsv"),
    );
}

#[test]
fn c_reference_silk_channel_transitions_stereo_output() {
    check_silk_decode(
        48000,
        2,
        include_str!("fixtures/reference/silk-channel-transitions.packets"),
        include_str!("fixtures/reference/silk-channel-transitions-2.tsv"),
    );
}
