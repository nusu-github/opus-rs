//! In-process codec benchmark. Setup, I/O, and corpus generation are not timed.
#![forbid(unsafe_code)]

use opus_rs::c_style_api::opus_decoder::{
    OpusDecoderCtlRequest, opus_decode, opus_decoder_create, opus_decoder_ctl,
};
use opus_rs::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

const CORPUS: usize = 128;
const WARMUP: usize = 64;
const CAPACITY: usize = 24576;

fn cpu_time() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        10,
        "Usage: opus-rs-bench encode|decode|verify RATE CHANNELS FRAME MODE BITRATE COMPLEXITY ITERATIONS PCM_I16LE"
    );
    let number = |i: usize| args[i].parse::<i32>().unwrap();
    let (rate, channels, frame, mode, bitrate, complexity) = (
        number(2),
        number(3),
        number(4) as usize,
        number(5),
        number(6),
        number(7),
    );
    let iterations = args[8].parse::<usize>().unwrap();
    let input: Vec<_> = std::fs::read(&args[9])
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    let stride = frame * channels as usize;
    assert_eq!(input.len(), CORPUS * stride);
    let loss_every: usize = std::env::var("OPUS_BENCH_LOSS_EVERY")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let mut encoder = opus_encoder_create(rate, channels, 2049).unwrap();
    for request in [
        OpusEncoderCtlRequest::SetBitrate(bitrate),
        OpusEncoderCtlRequest::SetComplexity(complexity),
        OpusEncoderCtlRequest::SetForceMode(mode),
        OpusEncoderCtlRequest::SetVbr(true),
        OpusEncoderCtlRequest::SetVbrConstraint(true),
        #[cfg(feature = "enable_qext")]
        OpusEncoderCtlRequest::SetQext(true),
    ] {
        opus_encoder_ctl(&mut encoder, request).unwrap();
    }
    let bandwidth = match mode {
        1000 => Some(if rate >= 16000 {
            1103
        } else if rate >= 12000 {
            1102
        } else {
            1101
        }),
        1001 => Some(if rate >= 48000 { 1105 } else { 1104 }),
        _ => None,
    };
    if let Some(bandwidth) = bandwidth {
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBandwidth(bandwidth)).unwrap();
    }
    let mut packet = vec![0; CAPACITY];
    for i in 0..WARMUP {
        opus_encode(
            &mut encoder,
            &input[i * stride..(i + 1) * stride],
            frame,
            &mut packet,
        )
        .unwrap();
    }
    if args[1] == "encode" {
        let mut checksum = 0u64;
        let cpu_start = cpu_time();
        let start = Instant::now();
        for i in 0..iterations {
            let offset = (i % CORPUS) * stride;
            let size = opus_encode(
                &mut encoder,
                black_box(&input[offset..offset + stride]),
                frame,
                black_box(&mut packet),
            )
            .unwrap();
            checksum = checksum.wrapping_add(size as u64 + u64::from(packet[0]));
        }
        let elapsed = start.elapsed().as_nanos();
        let cpu_elapsed = cpu_time() - cpu_start;
        let mut range = 0;
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        println!(
            "{{\"elapsed_ns\":{elapsed},\"cpu_ns\":{cpu_elapsed},\"iterations\":{iterations},\"checksum\":{checksum},\"final_range\":{range}}}"
        );
        return;
    }
    let mut packets = Vec::with_capacity(CORPUS);
    let mut ranges = Vec::with_capacity(CORPUS);
    for i in 0..CORPUS {
        let size = opus_encode(
            &mut encoder,
            &input[i * stride..(i + 1) * stride],
            frame,
            &mut packet,
        )
        .unwrap();
        packets.push(packet[..size].to_vec());
        let mut range = 0;
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::GetFinalRange(&mut range),
        )
        .unwrap();
        ranges.push(range);
    }
    let mut decoder = opus_decoder_create(rate, channels).unwrap();
    let mut output = vec![0i16; stride];
    let decode = |decoder: &mut _, i: usize, output: &mut [i16]| {
        let lost = loss_every != 0 && (i + 1) % loss_every == 0;
        let bytes = &packets[i % CORPUS];
        opus_decode(
            decoder,
            (!lost).then_some(bytes.as_slice()),
            if lost { 0 } else { bytes.len() },
            output,
            frame,
            false,
        )
        .unwrap()
    };
    for i in 0..WARMUP {
        decode(&mut decoder, i, &mut output);
    }
    if args[1] == "verify" {
        let mut stdout = io::BufWriter::new(io::stdout().lock());
        for i in 0..CORPUS {
            let samples = decode(&mut decoder, i, &mut output);
            let mut range = 0;
            opus_decoder_ctl(
                &mut decoder,
                OpusDecoderCtlRequest::GetFinalRange(&mut range),
            )
            .unwrap();
            stdout
                .write_all(&(packets[i].len() as u32).to_le_bytes())
                .unwrap();
            stdout.write_all(&packets[i]).unwrap();
            stdout.write_all(&ranges[i].to_le_bytes()).unwrap();
            stdout.write_all(&(samples as u32).to_le_bytes()).unwrap();
            for value in &output[..samples * channels as usize] {
                stdout.write_all(&value.to_le_bytes()).unwrap();
            }
            stdout.write_all(&range.to_le_bytes()).unwrap();
        }
        return;
    }
    assert_eq!(args[1], "decode");
    let mut checksum = 0u64;
    let cpu_start = cpu_time();
    let start = Instant::now();
    for i in 0..iterations {
        let samples = decode(&mut decoder, i, black_box(&mut output));
        checksum = checksum.wrapping_add(samples as u64 + u64::from(output[0] as u16));
    }
    let elapsed = start.elapsed().as_nanos();
    let cpu_elapsed = cpu_time() - cpu_start;
    let mut range = 0;
    opus_decoder_ctl(
        &mut decoder,
        OpusDecoderCtlRequest::GetFinalRange(&mut range),
    )
    .unwrap();
    println!(
        "{{\"elapsed_ns\":{elapsed},\"cpu_ns\":{cpu_elapsed},\"iterations\":{iterations},\"checksum\":{checksum},\"final_range\":{range}}}"
    );
}
