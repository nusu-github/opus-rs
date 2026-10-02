//! Optional differential tests against the pinned C process oracle.
//! Run `tools/reference/build.sh`, then
//! `cargo test --test packet_reference -- --ignored --nocapture`.

use std::fmt::Write;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use opus_rs::c_style_api::extensions::{OpusExtensionData, opus_packet_extensions_parse};
use opus_rs::c_style_api::packet::opus_packet_parse_impl;
use opus_rs::c_style_api::repacketizer::{
    OpusRepacketizer, opus_multistream_packet_pad, opus_multistream_packet_unpad, opus_packet_pad,
    opus_packet_unpad,
};

struct Case {
    operation: char,
    parameter: usize,
    streams: usize,
    packet: Vec<u8>,
}

fn hex(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);
    for byte in data {
        write!(result, "{byte:02x}").unwrap();
    }
    result
}

fn evaluate(case: &Case) -> String {
    let packet = &case.packet;
    if case.operation == 'p' {
        return match opus_packet_parse_impl(packet, packet.len(), case.parameter != 0) {
            Err(error) => error.code().to_string(),
            Ok(parsed) => {
                let mut result = format!(
                    "{} {} {} {} {}",
                    parsed.frame_count,
                    parsed.toc,
                    parsed.payload_offset,
                    parsed.packet_offset,
                    parsed.padding.len()
                );
                let mut offset = parsed.payload_offset;
                for frame in &parsed.frames[..parsed.frame_count] {
                    write!(result, " {offset}:{}", frame.len()).unwrap();
                    offset += frame.len();
                }
                result
            }
        };
    }
    if case.operation == 'e' {
        let mut extensions = [OpusExtensionData::default(); 4096];
        return match opus_packet_extensions_parse(
            packet,
            packet.len(),
            case.parameter,
            &mut extensions,
        ) {
            Err(error) => error.code().to_string(),
            Ok(count) => {
                let mut result = format!("0 {count}");
                for ext in &extensions[..count] {
                    let offset = ext.data.as_ptr() as usize - packet.as_ptr() as usize;
                    write!(result, " {}:{}:{offset}:{}", ext.id, ext.frame, ext.len).unwrap();
                }
                result
            }
        };
    }
    let mut output = vec![0u8; packet.len().max(case.parameter).max(1)];
    output[..packet.len()].copy_from_slice(packet);
    let result = match case.operation {
        'r' => {
            let mut rp = OpusRepacketizer::new();
            rp.opus_repacketizer_cat(packet, packet.len())
                .and_then(|()| rp.opus_repacketizer_out(&mut output, case.parameter))
        }
        'a' => opus_packet_pad(&mut output, packet.len(), case.parameter).map(|()| case.parameter),
        'u' => opus_packet_unpad(&mut output, packet.len()),
        'm' => opus_multistream_packet_pad(&mut output, packet.len(), case.parameter, case.streams)
            .map(|()| case.parameter),
        'n' => opus_multistream_packet_unpad(&mut output, packet.len(), case.streams),
        _ => unreachable!(),
    };
    match result {
        Ok(len) => format!("{len}:{}", hex(&output[..len])),
        Err(error) => error.code().to_string(),
    }
}

fn random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
#[ignore = "requires the pinned C reference built by tools/reference/build.sh"]
fn packet_operations_match_pinned_c_reference() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = root.join("target/reference/source");
    let executable = root.join("target/reference/packet-reference");
    assert!(
        reference.join("libopus.a").exists(),
        "run tools/reference/build.sh first"
    );
    let status = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args(["-O2", "-std=c99", "-DOPUS_BUILD"])
        .arg("-I")
        .arg(reference.join("include"))
        .arg("-I")
        .arg(reference.join("celt"))
        .arg("-I")
        .arg(reference.join("src"))
        .arg("-I")
        .arg(reference.join("silk"))
        .arg(root.join("rust/tests/packet_reference.c"))
        .arg(reference.join("libopus.a"))
        .arg("-lm")
        .arg("-o")
        .arg(&executable)
        .status()
        .unwrap();
    assert!(status.success());

    let mut cases = Vec::new();
    for value in 0u32..=65535 {
        let packet = vec![(value >> 8) as u8, value as u8];
        for parameter in 0..=1 {
            cases.push(Case {
                operation: 'p',
                parameter,
                streams: 1,
                packet: packet.clone(),
            });
        }
    }
    let mut rng = 0xbabef00du32;
    for trial in 0..12000 {
        let len = if trial < 4000 {
            trial % 32
        } else {
            random(&mut rng) as usize % 2048
        };
        let packet: Vec<u8> = (0..len).map(|_| random(&mut rng) as u8).collect();
        for parameter in 0..=1 {
            cases.push(Case {
                operation: 'p',
                parameter,
                streams: 1,
                packet: packet.clone(),
            });
        }
        for operation in ['r', 'a', 'u'] {
            cases.push(Case {
                operation,
                parameter: len + (trial % 513),
                streams: 1,
                packet: packet.clone(),
            });
        }
        cases.push(Case {
            operation: 'e',
            parameter: trial % 48 + 1,
            streams: 1,
            packet,
        });
    }
    // Long extensions and lacing transitions require deliberately valid data;
    // random packets overwhelmingly exercise rejection paths instead.
    for id in [32u8, 124] {
        for extension_len in [
            0, 1, 2, 252, 253, 254, 255, 256, 507, 508, 509, 1275, 2550, 3825, 22950,
        ] {
            let extension_size = extension_len + 1;
            let pad_header = extension_size / 254;
            let mut packet = vec![0x83, 0x41];
            packet.extend(std::iter::repeat_n(255, pad_header));
            packet.push((extension_size - 254 * pad_header) as u8);
            packet.push(id << 1);
            packet.extend(std::iter::repeat_n(0x7a, extension_len));
            for operation in ['r', 'a', 'u'] {
                for extra in [0, 1, 2, 254, 255, 256, 800] {
                    cases.push(Case {
                        operation,
                        parameter: packet.len() + extra,
                        streams: 1,
                        packet: packet.clone(),
                    });
                }
            }
        }
    }
    for count in 1..=48 {
        for frame_len in [0, 1, 2, 251, 252, 255, 1275] {
            let mut packet = vec![0x83, count as u8];
            packet.extend(std::iter::repeat_n(0x55, count * frame_len));
            for operation in ['r', 'a', 'u'] {
                cases.push(Case {
                    operation,
                    parameter: packet.len() + 257,
                    streams: 1,
                    packet: packet.clone(),
                });
            }
        }
    }
    for streams in 1..=8 {
        for tail in [vec![0x80], vec![0x80, 0x55], vec![0x83, 0x41, 0]] {
            let mut packet = Vec::new();
            for _ in 1..streams {
                packet.extend_from_slice(&[0x80, 1, 0x77]);
            }
            packet.extend_from_slice(&tail);
            for operation in ['m', 'n'] {
                cases.push(Case {
                    operation,
                    parameter: packet.len() + 257,
                    streams,
                    packet: packet.clone(),
                });
            }
        }
    }
    let mut input = String::new();
    for case in &cases {
        writeln!(
            input,
            "{} {} {} {}",
            case.operation,
            case.parameter,
            case.streams,
            hex(&case.packet)
        )
        .unwrap();
    }
    let input_path = root.join("target/reference/packet-cases.txt");
    fs::write(&input_path, input).unwrap();
    let output = Command::new(&executable).arg(input_path).output().unwrap();
    assert!(
        output.status.success(),
        "C oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference_results = String::from_utf8(output.stdout).unwrap();
    assert_eq!(reference_results.lines().count(), cases.len());
    for (index, (case, expected)) in cases.iter().zip(reference_results.lines()).enumerate() {
        assert_eq!(
            evaluate(case),
            expected,
            "case {index}: {} {} {} {}",
            case.operation,
            case.parameter,
            case.streams,
            hex(&case.packet)
        );
    }
    eprintln!(
        "{} packet operation cases matched the pinned C reference",
        cases.len()
    );
}
