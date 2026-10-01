//! Opus Custom packet and sample compatibility with independently built C.
#![cfg(feature = "custom_modes")]

use opus_rs::custom::Mode;

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn custom_packets_and_pcm_match_pinned_c() {
    let profile = if cfg!(feature = "enable_res24") {
        "fixed24"
    } else if cfg!(feature = "fixed_point") {
        "fixed"
    } else {
        "float"
    };
    let mut suffix = if profile != "float" || cfg!(feature = "pfa") || cfg!(feature = "enable_qext")
    {
        format!("-{profile}")
    } else {
        String::new()
    };
    if cfg!(feature = "enable_qext") {
        suffix.push_str("-qext");
    }
    if cfg!(feature = "pfa") {
        suffix.push_str("-pfa");
    }
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("rust/tests/fixtures/reference/custom{suffix}"));
    let cases = std::fs::read_to_string(directory.join("cases.txt")).unwrap();
    let mut configurations = Vec::new();
    for case in cases.lines() {
        let (rate, size) = case.split_once(' ').unwrap();
        for channels in [1, 2] {
            configurations.push((
                rate.parse::<u32>().unwrap(),
                size.parse::<usize>().unwrap(),
                channels,
                false,
            ));
        }
    }
    configurations.extend([
        (48000, 90, 1, true),
        (32000, 320, 2, true),
        (96000, 960, 1, true),
    ]);
    #[allow(unused_mut)]
    let mut configurations: Vec<_> = configurations
        .into_iter()
        .map(|(r, s, c, f)| (r, s, c, f, false))
        .collect();
    #[cfg(feature = "enable_qext")]
    for (rate, size) in [(48000, 960), (96000, 1920), (48000, 720), (96000, 1440)] {
        for channels in [1, 2] {
            configurations.push((rate, size, channels, false, true));
        }
    }
    let selected = std::env::var("OPUS_CUSTOM_TEST_CASE").ok();
    let mut matched = 0;
    let mut failures = Vec::new();
    for (rate, size, channels, fractional, extension) in configurations {
        let suffix = if extension {
            "-ext"
        } else if fractional {
            "-24"
        } else {
            ""
        };
        let name = format!("{rate}-{size}-{channels}{suffix}");
        if selected.as_ref().is_some_and(|value| value != &name) {
            continue;
        }
        matched += 1;
        let result = std::panic::catch_unwind(|| {
            let input_suffix = if extension {
                "-ext"
            } else if fractional {
                "-24"
            } else {
                ""
            };
            let capacity = if extension { 800 } else { 80 };
            let golden = std::fs::read_to_string(
                directory.join(format!("{rate}-{size}-{channels}{input_suffix}.tsv")),
            )
            .unwrap();
            let mode = Mode::new(rate, size).unwrap();
            let view = mode.view();
            let mut encoder = view.encoder(channels).unwrap();
            encoder.set_complexity(10).unwrap();
            let mut encoder_float = view.encoder(channels).unwrap();
            let mut encoder_24 = view.encoder(channels).unwrap();
            encoder_float.set_complexity(10).unwrap();
            encoder_24.set_complexity(10).unwrap();
            #[cfg(feature = "enable_qext")]
            for enc in [&mut encoder, &mut encoder_float, &mut encoder_24] {
                assert!(!enc.qext().unwrap());
                enc.set_qext(extension).unwrap();
                assert_eq!(enc.qext().unwrap(), extension);
            }
            let mut decoder = view.decoder(channels).unwrap();
            let mut decoder_float = view.decoder(channels).unwrap();
            let mut decoder_24 = view.decoder(channels).unwrap();
            for (frame, line) in golden.lines().enumerate() {
                let fields: Vec<_> = line.split_whitespace().collect();
                let input: Vec<i16> = (0..size * channels)
                    .map(|i| (((i + frame * size * channels) * 127 + 811) % 24001) as i16 - 12000)
                    .collect();
                let mut packet = vec![0u8; capacity];
                let reference = if fields[5] == "-" {
                    Vec::new()
                } else {
                    unhex(fields[5])
                };
                let received = if reference.is_empty() {
                    None
                } else {
                    Some(reference.as_slice())
                };
                if received.is_some() {
                    if !fractional {
                        let count = encoder.encode(&input, &mut packet).unwrap();
                        assert_eq!(
                            &packet[..count],
                            reference,
                            "{rate}/{size}/{channels} frame{frame} packet"
                        );
                        assert_eq!(encoder.final_range(), fields[3].parse::<u32>().unwrap());
                    }
                    let input24: Vec<i32> = input
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            i32::from(x) * 256
                                + if fractional {
                                    (((i + frame * size * channels) * 23) % 255) as i32 - 127
                                } else {
                                    0
                                }
                        })
                        .collect();
                    let float_input: Vec<f32> =
                        input24.iter().map(|&x| x as f32 / 8388608.0).collect();
                    let count = encoder_float
                        .encode_float(&float_input, &mut packet)
                        .unwrap();
                    assert_eq!(
                        &packet[..count],
                        reference,
                        "{rate}/{size}/{channels} frame{frame} float encode"
                    );
                    let count = encoder_24.encode_24(&input24, &mut packet).unwrap();
                    assert_eq!(
                        &packet[..count],
                        reference,
                        "{rate}/{size}/{channels} frame{frame} 24-bit encode"
                    );
                }
                let mut pcm = vec![0i16; size * channels];
                assert_eq!(decoder.decode(received, &mut pcm).unwrap(), size);
                assert_eq!(decoder.final_range(), fields[4].parse::<u32>().unwrap());
                let expected: Vec<i16> = unhex(fields[6])
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]))
                    .collect();
                if let Some(index) = pcm.iter().zip(&expected).position(|(a, b)| a != b) {
                    panic!(
                        "{rate}/{size}/{channels} frame{frame} PCM sample{index}: Rust {} C {}",
                        pcm[index], expected[index]
                    );
                }
                assert_eq!(decoder.final_range(), fields[4].parse::<u32>().unwrap());
                let mut floats = vec![0.0; size * channels];
                decoder_float.decode_float(received, &mut floats).unwrap();
                for (index, word) in fields[7].as_bytes().chunks_exact(8).enumerate() {
                    let expected =
                        u32::from_str_radix(core::str::from_utf8(word).unwrap(), 16).unwrap();
                    assert_eq!(
                        floats[index].to_bits(),
                        expected,
                        "{rate}/{size}/{channels} frame{frame} float sample{index}"
                    );
                }
                let mut pcm24 = vec![0i32; size * channels];
                decoder_24.decode_24(received, &mut pcm24).unwrap();
                for (index, word) in fields[8].as_bytes().chunks_exact(8).enumerate() {
                    let expected = u32::from_str_radix(core::str::from_utf8(word).unwrap(), 16)
                        .unwrap() as i32;
                    assert_eq!(
                        pcm24[index], expected,
                        "{rate}/{size}/{channels} frame{frame} 24-bit sample{index}"
                    );
                }
            }
            encoder.reset();
            decoder.reset();
            let first: Vec<_> = golden.lines().next().unwrap().split_whitespace().collect();
            let reference = unhex(first[5]);
            let input: Vec<i16> = (0..size * channels)
                .map(|i| ((i * 127 + 811) % 24001) as i16 - 12000)
                .collect();
            let mut packet = vec![0; capacity];
            if !fractional {
                let count = encoder.encode(&input, &mut packet).unwrap();
                assert_eq!(
                    &packet[..count],
                    reference,
                    "reset encoder {rate}/{size}/{channels}"
                );
            }
            let mut pcm = vec![0; size * channels];
            decoder.decode(Some(&reference), &mut pcm).unwrap();
            let expected: Vec<_> = unhex(first[6])
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            assert_eq!(pcm, expected, "reset decoder {rate}/{size}/{channels}");
            assert!(encoder.encode(&[], &mut packet).is_err());
            assert!(decoder.decode(Some(&reference), &mut []).is_err());
        });
        if result.is_err() {
            failures.push((rate, size, channels, fractional, extension));
        }
    }
    assert!(matched > 0, "no matching custom reference case");
    assert!(failures.is_empty(), "C reference mismatch in {failures:?}");
}

#[test]
fn custom_mode_rejects_unsupported_transforms_without_panicking() {
    for size in (40usize..=if cfg!(feature = "enable_qext") {
        2048
    } else {
        1024
    })
        .step_by(2)
    {
        let mut factor = size / 2;
        for radix in [2, 3, 5] {
            while factor.is_multiple_of(radix) {
                factor /= radix;
            }
        }
        if factor != 1 {
            assert!(
                Mode::new(48000, size).is_err(),
                "unsupported FFT size {size}"
            );
        }
    }
    assert!(Mode::new(48000, 510).is_err());
    assert!(Mode::new(48000, 1).is_err());
    assert!(Mode::new(u32::MAX, 960).is_err());
    assert!(Mode::new(48000, 960).unwrap().view().encoder(0).is_err());
}

#[test]
fn mode_dimensions_match_c_without_panics() {
    #[cfg(not(feature = "enable_qext"))]
    let cases = include_str!("fixtures/reference/custom/constructors.txt");
    #[cfg(feature = "enable_qext")]
    let cases = include_str!("fixtures/reference/custom-float-qext/constructors.txt");
    for line in cases.lines() {
        let values: Vec<usize> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let result = Mode::new(values[0] as u32, values[1]);
        assert_eq!(
            result.is_ok(),
            values[2] == 1,
            "rate {} frame {}",
            values[0],
            values[1]
        );
    }
    assert!(Mode::new(48000, usize::MAX).is_err());
}

#[cfg(feature = "enable_qext")]
#[test]
fn custom_decoder_rejects_modes_without_prediction_history() {
    let mode = Mode::new(96000, 2048).unwrap();
    assert!(mode.view().decoder(1).is_err());
    assert!(mode.view().decoder(2).is_err());
    assert!(Mode::new(96000, 1920).unwrap().view().decoder(2).is_ok());
}
