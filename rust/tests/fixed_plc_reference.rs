//! Complete fixed-point PCM and float-bit checks for C-generated packet loss.
#![cfg(feature = "fixed_point")]

use opus_rs::c_style_api::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decode_float, opus_decoder_create, opus_decoder_ctl,
};

macro_rules! fixture {
    ($file:literal) => {{
        match (
            cfg!(feature = "enable_res24"),
            cfg!(feature = "enable_qext"),
            cfg!(feature = "pfa"),
        ) {
            (false, false, false) => include_str!(concat!("fixtures/reference/fixed-plc-", $file)),
            (false, false, true) => {
                include_str!(concat!("fixtures/reference/fixed-pfa-plc-", $file))
            }
            (false, true, false) => {
                include_str!(concat!("fixtures/reference/fixed-qext-plc-", $file))
            }
            (false, true, true) => {
                include_str!(concat!("fixtures/reference/fixed-qext-pfa-plc-", $file))
            }
            (true, false, false) => {
                include_str!(concat!("fixtures/reference/fixed-res24-plc-", $file))
            }
            (true, false, true) => {
                include_str!(concat!("fixtures/reference/fixed-res24-pfa-plc-", $file))
            }
            (true, true, false) => {
                include_str!(concat!("fixtures/reference/fixed-res24-qext-plc-", $file))
            }
            (true, true, true) => include_str!(concat!(
                "fixtures/reference/fixed-res24-qext-pfa-plc-",
                $file
            )),
        }
    }};
}

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn fixed_plc_matches_pinned_c_samples_and_ranges() {
    let cases = [
        (
            8000,
            1,
            fixture!("8000-1-40.packets"),
            fixture!("8000-1-40.tsv"),
        ),
        (
            24000,
            1,
            fixture!("24000-1-120.packets"),
            fixture!("24000-1-120.tsv"),
        ),
        (
            24000,
            2,
            fixture!("24000-2-240.packets"),
            fixture!("24000-2-240.tsv"),
        ),
        (
            48000,
            1,
            fixture!("48000-1-960.packets"),
            fixture!("48000-1-960.tsv"),
        ),
    ];
    for (rate, channels, script, golden) in cases {
        let mut integer = opus_decoder_create(rate, channels).unwrap();
        let mut float = opus_decoder_create(rate, channels).unwrap();
        assert_eq!(script.lines().count(), golden.lines().count());
        for (frame, (line, expected)) in script.lines().zip(golden.lines()).enumerate() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let size: usize = fields[0].parse().unwrap();
            let packet = if fields[2] == "-" {
                Vec::new()
            } else {
                unhex(fields[2])
            };
            let data = if packet.is_empty() {
                None
            } else {
                Some(packet.as_slice())
            };
            let expected: Vec<_> = expected.split_whitespace().collect();
            let samples: usize = expected[2].parse().unwrap();
            let mut pcm = vec![0i16; size * channels as usize];
            let mut floating = vec![0f32; pcm.len()];
            assert_eq!(
                opus_decode(&mut integer, data, packet.len(), &mut pcm, size, false).unwrap(),
                samples
            );
            assert_eq!(
                opus_decode_float(&mut float, data, packet.len(), &mut floating, size, false)
                    .unwrap(),
                samples
            );
            let expected_pcm = unhex(expected[5]);
            for (index, (sample, bytes)) in pcm.iter().zip(expected_pcm.chunks_exact(2)).enumerate()
            {
                assert_eq!(
                    *sample,
                    i16::from_le_bytes([bytes[0], bytes[1]]),
                    "{rate}/{channels} frame {frame} sample {index}"
                );
            }
            for (index, (sample, bits)) in floating
                .iter()
                .zip(expected[6].as_bytes().chunks_exact(8))
                .enumerate()
            {
                assert_eq!(
                    sample.to_bits(),
                    u32::from_str_radix(core::str::from_utf8(bits).unwrap(), 16).unwrap(),
                    "{rate}/{channels} frame {frame} float {index}"
                );
            }
            for (decoder, range) in [(&mut integer, expected[3]), (&mut float, expected[4])] {
                let mut actual = 0;
                opus_decoder_ctl(decoder, OpusDecoderCtlRequest::GetFinalRange(&mut actual))
                    .unwrap();
                assert_eq!(actual, range.parse::<u32>().unwrap());
            }
        }
    }
}
