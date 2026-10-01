//! Exact C-generated SILK packets across durations, complexity, CBR/VBR, and FEC.

use opus_rs::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};

fn input(samples: usize, channels: usize, pattern: &str) -> Vec<i16> {
    let mut output = Vec::with_capacity(samples * channels);
    let mut state = 0x1234_5678_u32;
    for index in 0..samples {
        for channel in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 16) as i32 - 32768;
            let triangle = ((index * (197 + channel * 37)) % 32768) as i32 - 16384;
            let sample = match pattern {
                "silence" => 0,
                "impulse" => {
                    if index % 257 == 0 {
                        32767
                    } else {
                        0
                    }
                }
                "alternating" => {
                    if (index + channel) % 2 == 0 {
                        32767
                    } else {
                        -32768
                    }
                }
                "noise" => noise,
                "mixed" => (triangle + (noise >> 3)).clamp(-32768, 32767),
                _ => panic!("Unknown fixture waveform"),
            };
            output.push(sample as i16);
        }
    }
    output
}

#[test]
fn c_reference_silk_packets_and_ranges() {
    #[cfg(not(feature = "fixed_point"))]
    let fixture = include_str!("fixtures/reference/silk-encoder.tsv");
    #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
    let fixture = include_str!("fixtures/reference/silk-encoder-fixed.tsv");
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    let fixture = include_str!("fixtures/reference/silk-encoder-fixed-res24.tsv");
    let mut records = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut configurations = 0;
    while let Some(configuration) = records.next() {
        let fields: Vec<_> = configuration.split_whitespace().collect();
        assert_eq!(fields.len(), 10);
        assert_eq!(fields[0], "S");
        let number = |index: usize| fields[index].parse::<i32>().unwrap();
        let rate = number(1);
        let channels = number(2);
        let frame_size = rate as usize * number(3) as usize / 1_000_000;
        let frames = number(9) as usize;
        let pcm = input(frame_size * frames, channels as usize, fields[8]);
        let mut encoder = opus_encoder_create(rate, channels, 2049).unwrap();
        for request in [
            OpusEncoderCtlRequest::SetBitrate(24000 * channels),
            OpusEncoderCtlRequest::SetComplexity(number(4)),
            OpusEncoderCtlRequest::SetVbr(number(5) != 0),
            OpusEncoderCtlRequest::SetInbandFec(number(6) != 0),
            OpusEncoderCtlRequest::SetPacketLossPerc(number(7)),
            OpusEncoderCtlRequest::SetForceMode(1000),
            OpusEncoderCtlRequest::SetBandwidth(if rate >= 16000 {
                1103
            } else if rate >= 12000 {
                1102
            } else {
                1101
            }),
        ] {
            opus_encoder_ctl(&mut encoder, request).unwrap();
        }
        let mut packet = vec![0; 1276 * 6];
        for (frame, pcm) in pcm.chunks_exact(frame_size * channels as usize).enumerate() {
            let expected: Vec<_> = records.next().unwrap().split_whitespace().collect();
            assert_eq!(expected.len(), 3);
            assert_eq!(expected[0], "P");
            let bytes = opus_encode(&mut encoder, pcm, frame_size, &mut packet)
                .unwrap_or_else(|error| panic!("{configuration}, frame {frame}: {error:?}"));
            let expected_packet: Vec<_> = expected[2]
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            assert_eq!(
                &packet[..bytes],
                expected_packet,
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
                "{configuration}, frame {frame}: range"
            );
        }
        configurations += 1;
    }
    assert_eq!(configurations, 420);
}
