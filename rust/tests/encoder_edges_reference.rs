//! Exact C-generated controls, DTX transitions, packets, and selected PCM frames.
#![cfg(not(any(feature = "fixed_point", feature = "osce")))]

use opus_rs::c_style_api::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decode_float, opus_decoder_create, opus_decoder_ctl,
};
use opus_rs::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};

fn unhex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn input(frames: usize, channels: usize, pattern: &str) -> Vec<i16> {
    let mut state = 0x1234_5678_u32;
    let mut output = Vec::with_capacity(frames * 960 * channels);
    for index in 0..frames * 960 {
        for channel in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 16) as i32 - 32768;
            let triangle = ((index * (197 + channel * 37)) % 32768) as i32 - 16384;
            output.push(
                if pattern == "silence"
                    || (pattern == "activity" && (5 * 960..45 * 960).contains(&index))
                {
                    0
                } else {
                    (triangle + (noise >> 3)).clamp(-32768, 32767) as i16
                },
            );
        }
    }
    output
}

#[test]
fn c_reference_encoder_edge_controls_and_dtx_pcm() {
    let mut records = include_str!("fixtures/reference/encoder-edges.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'));
    let mut cases = 0;
    while let Some(configuration) = records.next() {
        let fields: Vec<_> = configuration.split_whitespace().collect();
        assert_eq!(fields.len(), 9);
        let number = |index: usize| fields[index].parse::<i32>().unwrap();
        let channels = number(2);
        let frames = number(8) as usize;
        let mut encoder = opus_encoder_create(48000, channels, number(3)).unwrap();
        for request in [
            OpusEncoderCtlRequest::SetBitrate(
                if number(1) == 1000 { 24000 } else { 48000 } * channels,
            ),
            OpusEncoderCtlRequest::SetComplexity(10),
            OpusEncoderCtlRequest::SetVbr(true),
            OpusEncoderCtlRequest::SetSignal(number(4)),
            OpusEncoderCtlRequest::SetVbrConstraint(number(5) != 0),
            OpusEncoderCtlRequest::SetDtx(number(6) != 0),
            OpusEncoderCtlRequest::SetForceMode(number(1)),
        ] {
            opus_encoder_ctl(&mut encoder, request).unwrap();
        }
        if matches!(number(1), 1000 | 1001) {
            opus_encoder_ctl(
                &mut encoder,
                OpusEncoderCtlRequest::SetBandwidth(if number(1) == 1000 { 1103 } else { 1105 }),
            )
            .unwrap();
        }
        let mut decoder = opus_decoder_create(48000, channels).unwrap();
        let mut float_decoder = opus_decoder_create(48000, channels).unwrap();
        let mut packet = vec![0; 1276 * 6];
        let mut decoded = vec![0; 960 * channels as usize];
        let mut float_decoded = vec![0.0_f32; 960 * channels as usize];
        for (frame, pcm) in input(frames, channels as usize, fields[7])
            .chunks_exact(960 * channels as usize)
            .enumerate()
        {
            let expected: Vec<_> = records.next().unwrap().split_whitespace().collect();
            assert_eq!(expected.len(), 7);
            let bytes = opus_encode(&mut encoder, pcm, 960, &mut packet).unwrap();
            assert_eq!(
                &packet[..bytes],
                unhex(expected[4]),
                "{configuration}, frame {frame}: packet"
            );
            let mut range = 0;
            opus_encoder_ctl(
                &mut encoder,
                OpusEncoderCtlRequest::GetFinalRange(&mut range),
            )
            .unwrap();
            assert_eq!(
                range,
                expected[1].parse::<u32>().unwrap(),
                "{configuration}, frame {frame}: encoder range"
            );
            assert_eq!(
                opus_decode(
                    &mut decoder,
                    Some(&packet[..bytes]),
                    bytes,
                    &mut decoded,
                    960,
                    false
                )
                .unwrap(),
                960
            );
            assert_eq!(
                opus_decode_float(
                    &mut float_decoder,
                    Some(&packet[..bytes]),
                    bytes,
                    &mut float_decoded,
                    960,
                    false
                )
                .unwrap(),
                960
            );
            opus_decoder_ctl(
                &mut decoder,
                OpusDecoderCtlRequest::GetFinalRange(&mut range),
            )
            .unwrap();
            assert_eq!(
                range,
                expected[2].parse::<u32>().unwrap(),
                "{configuration}, frame {frame}: decoder range"
            );
            opus_decoder_ctl(
                &mut float_decoder,
                OpusDecoderCtlRequest::GetFinalRange(&mut range),
            )
            .unwrap();
            assert_eq!(
                range,
                expected[3].parse::<u32>().unwrap(),
                "{configuration}, frame {frame}: float decoder range"
            );
            if expected[5] != "-" {
                let expected_pcm: Vec<_> = unhex(expected[5])
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    decoded, expected_pcm,
                    "{configuration}, frame {frame}: integer PCM"
                );
                let expected_bits: Vec<_> = expected[6]
                    .as_bytes()
                    .chunks_exact(8)
                    .map(|bytes| {
                        u32::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap()
                    })
                    .collect();
                assert_eq!(
                    float_decoded
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits,
                    "{configuration}, frame {frame}: float PCM bits"
                );
            }
        }
        cases += 1;
    }
    assert_eq!(cases, 112);
}
