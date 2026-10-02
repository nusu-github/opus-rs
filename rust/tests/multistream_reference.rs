//! Native process comparisons for multistream and projection encoding.
use opus_rs::c_style_api::opus_multistream::*;
use opus_rs::c_style_api::projection::*;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn hash_bytes(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(14695981039346656037u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
    })
}

#[cfg(not(feature = "fixed_point"))]
#[test]
fn low_frequency_tone_transient_matches_pinned_c() {
    let golden = include_str!("fixtures/reference/multistream-tone.tsv");
    for vbr in [false, true] {
        let mut encoder = opus_multistream_encoder_create(48000, 2, 2, 0, &[0, 1], 2051).unwrap();
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetBitrate(128000),
        )
        .unwrap();
        opus_multistream_encoder_ctl(&mut encoder, OpusMultistreamEncoderCtlRequest::SetVbr(vbr))
            .unwrap();
        for (frame, line) in golden
            .lines()
            .filter(|line| line.starts_with(if vbr { '1' } else { '0' }))
            .enumerate()
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            let pcm: Vec<f32> = (frame * 240..(frame + 1) * 240)
                .flat_map(|time| {
                    (0..2).map(move |channel| {
                        ((((time * (127 + channel * 73) + channel * 199) % 65536) as i32 - 32768)
                            * 211) as f32
                            / 8388608.0
                    })
                })
                .collect();
            let mut packet = vec![0; 65536];
            let length =
                opus_multistream_encode_float(&mut encoder, &pcm, 240, &mut packet).unwrap();
            let expected: Vec<u8> = fields[3]
                .as_bytes()
                .chunks_exact(2)
                .map(|bytes| u8::from_str_radix(core::str::from_utf8(bytes).unwrap(), 16).unwrap())
                .collect();
            assert_eq!(&packet[..length], expected, "VBR {vbr}, frame {frame}");
            let mut range = 0;
            opus_multistream_encoder_ctl(
                &mut encoder,
                OpusMultistreamEncoderCtlRequest::GetFinalRange(&mut range),
            )
            .unwrap();
            assert_eq!(
                range,
                fields[2].parse::<u32>().unwrap(),
                "VBR {vbr}, frame {frame} range"
            );
        }
    }
}

enum TestDecoder {
    Multistream(OpusMultistreamDecoder<'static>),
    Projection(OpusProjectionDecoder<'static>),
}

impl TestDecoder {
    fn hashes(decoders: &mut [Self], packet: &[u8], channels: usize, frame_size: usize) -> String {
        let mut short = vec![0i16; channels * frame_size];
        let mut int24 = vec![0i32; channels * frame_size];
        let mut floats = vec![0f32; channels * frame_size];
        match &mut decoders[0] {
            Self::Multistream(decoder) => opus_multistream_decode(
                decoder,
                packet,
                packet.len(),
                &mut short,
                frame_size,
                false,
            )
            .unwrap(),
            Self::Projection(decoder) => {
                opus_projection_decode(decoder, packet, packet.len(), &mut short, frame_size, false)
                    .unwrap()
            }
        };
        match &mut decoders[1] {
            Self::Multistream(decoder) => opus_multistream_decode24(
                decoder,
                packet,
                packet.len(),
                &mut int24,
                frame_size,
                false,
            )
            .unwrap(),
            Self::Projection(decoder) => opus_projection_decode24(
                decoder,
                packet,
                packet.len(),
                &mut int24,
                frame_size,
                false,
            )
            .unwrap(),
        };
        match &mut decoders[2] {
            Self::Multistream(decoder) => opus_multistream_decode_float(
                decoder,
                packet,
                packet.len(),
                &mut floats,
                frame_size,
                false,
            )
            .unwrap(),
            Self::Projection(decoder) => opus_projection_decode_float(
                decoder,
                packet,
                packet.len(),
                &mut floats,
                frame_size,
                false,
            )
            .unwrap(),
        };
        format!(
            "{:016x} {:016x} {:016x}",
            hash_bytes(short.iter().flat_map(|s| s.to_ne_bytes())),
            hash_bytes(int24.iter().flat_map(|s| s.to_ne_bytes())),
            hash_bytes(floats.iter().flat_map(|s| s.to_ne_bytes()))
        )
    }
}

fn reference_directory() -> PathBuf {
    let mut directory = String::from("target/reference");
    if cfg!(feature = "fixed_point") {
        directory.push_str("-fixed");
        if cfg!(feature = "enable_res24") {
            directory.push_str("-res24");
        }
    }
    if cfg!(feature = "enable_qext") {
        directory.push_str("-qext");
    }
    if cfg!(feature = "pfa") {
        directory.push_str("-pfa");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(directory)
}

fn build_reference() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = reference_directory().join("source");
    let executable =
        reference_directory().join(format!("multistream-reference-{}", std::process::id()));
    assert!(
        reference.join("libopus.a").exists(),
        "run tools/reference/build.sh first"
    );
    let mut command = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()));
    command.args(["-O2", "-std=c99", "-DOPUS_BUILD", "-DHAVE_LRINTF"]);
    if cfg!(feature = "enable_qext") {
        command.arg("-DENABLE_QEXT");
    }
    if cfg!(feature = "pfa") {
        command.arg("-DENABLE_PFA");
    }
    if cfg!(feature = "fixed_point") {
        command.arg("-DFIXED_POINT");
        if cfg!(feature = "enable_res24") {
            command.arg("-DENABLE_RES24");
        }
    }
    for path in ["include", "celt", "src", "silk"] {
        command.arg("-I").arg(reference.join(path));
    }
    assert!(
        command
            .arg(root.join("rust/tests/multistream_reference.c"))
            .arg(reference.join("libopus.a"))
            .arg("-lm")
            .arg("-o")
            .arg(&executable)
            .status()
            .unwrap()
            .success()
    );
    executable
}

#[test]
#[ignore = "requires pinned C reference; tests multistream, surround, ambisonics, projection"]
fn multistream_and_projection_match_reference() {
    compare_matrix(
        &[
            ("streams", 2),
            ("surround", 6),
            ("ambisonics", 4),
            ("projection", 4),
        ],
        &[8_000, 12_000, 16_000, 24_000, 48_000],
        &[1, 2, 4, 8, 16, 24, 32, 40, 48],
    );
}

#[cfg(feature = "enable_qext")]
#[test]
#[ignore = "requires pinned QEXT C reference; tests 96 kHz multistream and projection"]
fn quality_extension_96khz_matches_reference() {
    compare_matrix(
        &[
            ("streams", 2),
            ("surround", 6),
            ("ambisonics", 4),
            ("projection", 9),
        ],
        &[96_000],
        &[1, 2, 4, 8, 16, 24, 32, 40, 48],
    );
}

#[test]
#[ignore = "requires pinned C reference; tests every supported projection order and surround layout"]
fn channel_layouts_match_reference() {
    compare_matrix(
        &[
            ("surround", 3),
            ("surround", 4),
            ("surround", 5),
            ("surround", 6),
            ("surround", 7),
            ("surround", 8),
            ("ambisonics", 6),
            ("ambisonics", 11),
            ("projection", 4),
            ("projection", 6),
            ("projection", 9),
            ("projection", 11),
            ("projection", 16),
            ("projection", 18),
            ("projection", 25),
            ("projection", 27),
            ("projection", 36),
            ("projection", 38),
        ],
        &[48_000],
        &[8],
    );
}

#[cfg(feature = "enable_qext")]
#[test]
#[ignore = "requires pinned QEXT C reference; tests packets exceeding the standard size cap"]
fn quality_extension_maximum_packet_budget_matches_reference() {
    compare_matrix_at_bitrate(&[("streams", 2)], &[48_000], &[48], Some(1_536_000));
}

fn compare_matrix(layouts: &[(&str, usize)], rates: &[i32], durations: &[usize]) {
    compare_matrix_at_bitrate(layouts, rates, durations, None);
}

fn compare_matrix_at_bitrate(
    layouts: &[(&str, usize)],
    rates: &[i32],
    durations: &[usize],
    bitrate_override: Option<usize>,
) {
    static REFERENCE_PROCESS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = REFERENCE_PROCESS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let executable = build_reference();
    let mut cases = 0;
    let mut mismatches = Vec::new();
    let filter = std::env::var("OPUS_MULTISTREAM_TEST_KIND").ok();
    let selected_channels = std::env::var("OPUS_MULTISTREAM_TEST_CHANNELS")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid channel count"));
    for &(kind, channels) in layouts {
        if selected_channels.is_some_and(|selected| selected != channels) {
            continue;
        }
        if filter.as_ref().is_some_and(|selected| selected != kind) {
            continue;
        }
        let selected_rate = std::env::var("OPUS_MULTISTREAM_TEST_RATE")
            .ok()
            .map(|value| value.parse::<i32>().expect("valid sample rate"));
        for &rate in rates {
            if selected_rate.is_some_and(|selected| selected != rate) {
                continue;
            }
            for format in ["s16", "f32", "s24"] {
                for &duration_units in durations {
                    if std::env::var("OPUS_MULTISTREAM_TEST_DURATION_UNITS")
                        .ok()
                        .is_some_and(|selected| {
                            selected.parse::<usize>().unwrap() != duration_units
                        })
                    {
                        continue;
                    }
                    let frame_size = duration_units * rate as usize / 400;
                    for vbr in [false, true] {
                        let application = 2051;
                        let bitrate_per_channel =
                            std::env::var("OPUS_MULTISTREAM_TEST_BITRATE_PER_CHANNEL")
                                .ok()
                                .map(|value| value.parse::<usize>().unwrap())
                                .unwrap_or(bitrate_override.unwrap_or(
                                    if cfg!(feature = "enable_qext") {
                                        192_000
                                    } else {
                                        64_000
                                    },
                                ));
                        let bitrate = bitrate_per_channel * channels;
                        let frames = 3;
                        let signal: Vec<i32> = (0..frame_size * channels * frames)
                            .map(|i| {
                                let channel = i % channels;
                                let time = i / channels;
                                (((time * (127 + channel * 73) + channel * 199) % 65536) as i32
                                    - 32768)
                                    * 211
                            })
                            .collect();
                        let input_path = reference_directory().join(format!(
                            "multistream-{kind}-{channels}-{}-pcm.bin",
                            std::process::id()
                        ));
                        let bytes: Vec<u8> = signal
                            .iter()
                            .flat_map(|&sample| match format {
                                "s16" => ((sample >> 8) as i16).to_le_bytes().to_vec(),
                                "s24" => sample.to_le_bytes().to_vec(),
                                _ => (sample as f32 / 8388608.0).to_le_bytes().to_vec(),
                            })
                            .collect();
                        fs::write(&input_path, bytes).unwrap();
                        let args = [
                            kind.to_string(),
                            rate.to_string(),
                            channels.to_string(),
                            frame_size.to_string(),
                            frames.to_string(),
                            format.to_string(),
                            application.to_string(),
                            i32::from(vbr).to_string(),
                            bitrate.to_string(),
                        ];
                        let output = Command::new(&executable)
                            .args(&args)
                            .arg(&input_path)
                            .output()
                            .unwrap();
                        assert!(
                            output.status.success(),
                            "oracle {args:?}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                        let expected = String::from_utf8(output.stdout).unwrap();
                        let mut projection = if kind == "projection" {
                            Some(
                                opus_projection_ambisonics_encoder_create(
                                    rate,
                                    channels,
                                    3,
                                    application,
                                )
                                .unwrap()
                                .0,
                            )
                        } else {
                            None
                        };
                        let mut encoder = if kind == "projection" {
                            None
                        } else if kind == "streams" {
                            let mapping: Vec<u8> = (0..channels).map(|i| i as u8).collect();
                            Some(
                                opus_multistream_encoder_create(
                                    rate,
                                    channels,
                                    channels,
                                    0,
                                    &mapping,
                                    application,
                                )
                                .unwrap(),
                            )
                        } else {
                            Some(
                                opus_multistream_surround_encoder_create(
                                    rate,
                                    channels,
                                    if kind == "surround" { 1 } else { 2 },
                                    application,
                                )
                                .unwrap()
                                .0,
                            )
                        };
                        let ms = if let Some(projection) = &mut projection {
                            projection.multistream_encoder()
                        } else {
                            encoder.as_mut().unwrap()
                        };
                        opus_multistream_encoder_ctl(
                            ms,
                            OpusMultistreamEncoderCtlRequest::SetBitrate(bitrate as i32),
                        )
                        .unwrap();
                        opus_multistream_encoder_ctl(
                            ms,
                            OpusMultistreamEncoderCtlRequest::SetVbr(vbr),
                        )
                        .unwrap();
                        #[cfg(feature = "enable_qext")]
                        opus_multistream_encoder_ctl(
                            ms,
                            OpusMultistreamEncoderCtlRequest::SetQext(true),
                        )
                        .unwrap();
                        let mut decoders: Vec<TestDecoder> = (0..3)
                            .map(|_| {
                                if let Some(projection) = &projection {
                                    let layout = projection.projection_layout();
                                    let mut matrix = vec![0; demixing_matrix_size(layout).unwrap()];
                                    write_demixing_matrix_subset(layout, &mut matrix).unwrap();
                                    TestDecoder::Projection(
                                        opus_projection_decoder_create(
                                            rate,
                                            channels,
                                            layout.streams,
                                            layout.coupled_streams,
                                            &matrix,
                                        )
                                        .unwrap(),
                                    )
                                } else {
                                    let layout = encoder.as_ref().unwrap().layout();
                                    TestDecoder::Multistream(
                                        opus_multistream_decoder_create(
                                            rate,
                                            channels,
                                            layout.nb_streams,
                                            layout.nb_coupled_streams,
                                            &layout.mapping[..channels],
                                        )
                                        .unwrap(),
                                    )
                                }
                            })
                            .collect();
                        for (frame, reference) in expected.lines().enumerate() {
                            let fields: Vec<&str> = reference.split_whitespace().collect();
                            assert_eq!(fields.len(), 6);
                            let signal = &signal[frame * frame_size * channels
                                ..(frame + 1) * frame_size * channels];
                            let integers: Vec<i16> =
                                signal.iter().map(|&sample| (sample >> 8) as i16).collect();
                            let floats: Vec<f32> = signal
                                .iter()
                                .map(|&sample| sample as f32 / 8388608.0)
                                .collect();
                            let mut packet = vec![0; 65536];
                            let length = if let Some(projection) = &mut projection {
                                match format {
                                    "s16" => opus_projection_encode(
                                        projection,
                                        &integers,
                                        frame_size,
                                        &mut packet,
                                    ),
                                    "s24" => opus_projection_encode24(
                                        projection,
                                        signal,
                                        frame_size,
                                        &mut packet,
                                    ),
                                    _ => opus_projection_encode_float(
                                        projection,
                                        &floats,
                                        frame_size,
                                        &mut packet,
                                    ),
                                }
                                .unwrap()
                            } else {
                                let encoder = encoder.as_mut().unwrap();
                                match format {
                                    "s16" => opus_multistream_encode(
                                        encoder,
                                        &integers,
                                        frame_size,
                                        &mut packet,
                                    ),
                                    "s24" => opus_multistream_encode24(
                                        encoder,
                                        signal,
                                        frame_size,
                                        &mut packet,
                                    ),
                                    _ => opus_multistream_encode_float(
                                        encoder,
                                        &floats,
                                        frame_size,
                                        &mut packet,
                                    ),
                                }
                                .unwrap()
                            };
                            let ms = if let Some(projection) = &mut projection {
                                projection.multistream_encoder()
                            } else {
                                encoder.as_mut().unwrap()
                            };
                            let mut range = 0;
                            opus_multistream_encoder_ctl(
                                ms,
                                OpusMultistreamEncoderCtlRequest::GetFinalRange(&mut range),
                            )
                            .unwrap();
                            let mut actual = format!("{length} {range} ");
                            for byte in &packet[..length] {
                                write!(actual, "{byte:02x}").unwrap();
                            }
                            let reference_packet: Vec<u8> = fields[2]
                                .as_bytes()
                                .chunks_exact(2)
                                .map(|digits| {
                                    u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16)
                                        .unwrap()
                                })
                                .collect();
                            actual.push(' ');
                            actual.push_str(&TestDecoder::hashes(
                                &mut decoders,
                                &reference_packet,
                                channels,
                                frame_size,
                            ));
                            let artifact = format!(
                                "multistream-{kind}-{channels}-{rate}-{format}-{frame_size}-{}-{frame}",
                                i32::from(vbr)
                            );
                            if actual != reference {
                                let first_difference = actual
                                    .bytes()
                                    .zip(reference.bytes())
                                    .position(|(a, b)| a != b)
                                    .unwrap_or(actual.len().min(reference.len()));
                                mismatches.push(format!(
                                "{args:?}, frame {frame}, first text difference {first_difference}"
                            ));
                                fs::write(
                                    reference_directory().join(format!("{artifact}-rust.txt")),
                                    &actual,
                                )
                                .unwrap();
                                fs::write(
                                    reference_directory().join(format!("{artifact}-c.txt")),
                                    reference,
                                )
                                .unwrap();
                            } else {
                                for implementation in ["rust", "c"] {
                                    let _ = fs::remove_file(
                                        reference_directory()
                                            .join(format!("{artifact}-{implementation}.txt")),
                                    );
                                }
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "{} of {cases} multistream/projection frames matched the reference",
        cases - mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} of {cases} frames differed: {:#?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(30)]
    );
}
