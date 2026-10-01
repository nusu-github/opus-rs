//! Test-only process oracle for comparison with the scalar C reference.
//!
//! This executable is deliberately separate from the portable codec library.

#![forbid(unsafe_code)]

use opus_rs::c_style_api::dred::{
    opus_decoder_dred_decode, opus_decoder_dred_decode_float, opus_decoder_dred_decode24,
    opus_dred_alloc, opus_dred_decoder_create, opus_dred_parse, opus_dred_process,
};
use opus_rs::c_style_api::opus_decoder::{
    OpusDecoder, OpusDecoderCtlRequest, opus_decode, opus_decode_float, opus_decode24,
    opus_decoder_create, opus_decoder_ctl,
};
use opus_rs::c_style_api::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encode_float, opus_encode24, opus_encoder_create,
    opus_encoder_ctl,
};
use std::fmt::Write as _;
use std::io::{self, Write as _};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn codec_env_int(
    name: &str,
    default_value: i32,
    minimum: i32,
    maximum: i32,
) -> Result<i32, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<i32>()
            .ok()
            .filter(|value| (minimum..=maximum).contains(value))
            .ok_or_else(|| "Invalid codec environment setting.".into()),
        Err(std::env::VarError::NotPresent) => Ok(default_value),
        Err(_) => Err("Invalid codec environment setting.".into()),
    }
}

fn configure_decoder(decoder: &mut OpusDecoder) -> Result<(), String> {
    opus_decoder_ctl(
        decoder,
        OpusDecoderCtlRequest::SetComplexity(codec_env_int(
            "OPUS_ORACLE_DECODER_COMPLEXITY",
            0,
            0,
            10,
        )?),
    )
    .map_err(debug)?;
    #[cfg(feature = "osce")]
    opus_decoder_ctl(
        decoder,
        OpusDecoderCtlRequest::SetOsceBwe(codec_env_int("OPUS_ORACLE_OSCE_BWE", 0, 0, 1)? != 0),
    )
    .map_err(debug)?;
    if let Ok(path) = std::env::var("OPUS_ORACLE_DNN_BLOB") {
        let bytes = std::fs::read(path).map_err(debug)?;
        opus_decoder_ctl(decoder, OpusDecoderCtlRequest::SetDnnBlob(&bytes)).map_err(debug)?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 5 && matches!(args[1].as_str(), "decode" | "dred_decode") {
        return decode(&args);
    }
    if args.len() != 9
        || !matches!(
            args[1].as_str(),
            "codec" | "codec_float" | "codec24" | "codec_transition"
        )
    {
        return Err("Usage: opus-rs-oracle codec RATE CHANNELS FRAME_SIZE FRAME_COUNT MODE BITRATE INPUT_I16LE".into());
    }
    let number = |index: usize| {
        args[index]
            .parse::<i32>()
            .map_err(|_| format!("Invalid integer: {}", args[index]))
    };
    let rate = number(2)?;
    let channels = number(3)?;
    let frame_size = number(4)?;
    let frame_count = number(5)?;
    let mode = number(6)?;
    let bitrate = number(7)?;
    if !(1..=rate * 120 / 1000).contains(&frame_size)
        || !(1..=1000).contains(&frame_count)
        || !(1..=2).contains(&channels)
    {
        return Err("Invalid codec dimensions.".into());
    }
    let mut encoder = opus_encoder_create(
        rate,
        channels,
        codec_env_int("OPUS_ORACLE_APPLICATION", 2049, 2048, 2051)?,
    )
    .map_err(debug)?;
    let mut decoder = opus_decoder_create(rate, channels).map_err(debug)?;
    let mut float_decoder = opus_decoder_create(rate, channels).map_err(debug)?;
    configure_decoder(&mut decoder)?;
    configure_decoder(&mut float_decoder)?;
    for request in [
        OpusEncoderCtlRequest::SetBitrate(bitrate),
        #[cfg(feature = "enable_qext")]
        OpusEncoderCtlRequest::SetQext(codec_env_int("OPUS_ORACLE_QEXT", 0, 0, 1)? != 0),
        OpusEncoderCtlRequest::SetVbr(codec_env_int("OPUS_ORACLE_VBR", 0, 0, 1)? != 0),
        OpusEncoderCtlRequest::SetComplexity(codec_env_int("OPUS_ORACLE_COMPLEXITY", 10, 0, 10)?),
        OpusEncoderCtlRequest::SetInbandFec(codec_env_int("OPUS_ORACLE_FEC", 0, 0, 1)? != 0),
        OpusEncoderCtlRequest::SetPacketLossPerc(codec_env_int("OPUS_ORACLE_LOSS", 0, 0, 100)?),
        OpusEncoderCtlRequest::SetDtx(codec_env_int("OPUS_ORACLE_DTX", 0, 0, 1)? != 0),
        #[cfg(feature = "dred")]
        OpusEncoderCtlRequest::SetDredDuration(codec_env_int(
            "OPUS_ORACLE_DRED_DURATION",
            0,
            0,
            104,
        )?),
        OpusEncoderCtlRequest::SetVbrConstraint(codec_env_int("OPUS_ORACLE_CVBR", 1, 0, 1)? != 0),
        OpusEncoderCtlRequest::SetSignal(codec_env_int("OPUS_ORACLE_SIGNAL", -1000, -1000, 3002)?),
        OpusEncoderCtlRequest::SetForceMode(mode),
    ] {
        opus_encoder_ctl(&mut encoder, request).map_err(debug)?;
    }
    let bandwidth = match mode {
        1000 => Some(if rate >= 16_000 {
            1103
        } else if rate >= 12_000 {
            1102
        } else {
            1101
        }),
        1001 => Some(if rate >= 48_000 { 1105 } else { 1104 }),
        _ => None,
    };
    if let Some(bandwidth) = bandwidth {
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBandwidth(bandwidth))
            .map_err(debug)?;
    }
    if args[1] == "codec_transition" {
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetInbandFec(true)).map_err(debug)?;
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetPacketLossPerc(15))
            .map_err(debug)?;
    }
    let input = std::fs::read(&args[8]).map_err(debug)?;
    let frame_size = frame_size as usize;
    let channels = channels as usize;
    let frame_count = frame_count as usize;
    let sample_bytes = if matches!(args[1].as_str(), "codec" | "codec_transition") {
        2
    } else {
        4
    };
    let input_frame_bytes = frame_size * channels * sample_bytes;
    if input.len() != input_frame_bytes * frame_count {
        return Err("PCM input length differs from requested dimensions.".into());
    }
    let max_samples = rate as usize * 120 / 1000;
    let mut encoded = vec![0; 3826 * 6];
    let mut decoded = vec![0i16; max_samples * channels];
    let mut decoded_float = vec![0f32; max_samples * channels];
    let mut output = io::BufWriter::new(io::stdout().lock());
    for (index, bytes) in input.chunks_exact(input_frame_bytes).enumerate() {
        if index as i32 == codec_env_int("OPUS_ORACLE_RESET_AT", -1, -1, i32::MAX)? {
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::ResetState).map_err(debug)?;
        }
        if args[1] == "codec_transition" {
            let mode = [1000, 1001, 1002, 1001, 1000][(index / 4) % 5];
            let rate_per_channel = if mode == 1000 { 24_000 } else { 48_000 };
            let bandwidth = if mode == 1000 {
                1103
            } else if rate >= 48_000 {
                1105
            } else {
                1104
            };
            for request in [
                OpusEncoderCtlRequest::SetInbandFec(true),
                OpusEncoderCtlRequest::SetPacketLossPerc(15),
                OpusEncoderCtlRequest::SetForceMode(mode),
                OpusEncoderCtlRequest::SetBitrate(rate_per_channel * channels as i32),
                OpusEncoderCtlRequest::SetBandwidth(bandwidth),
            ] {
                opus_encoder_ctl(&mut encoder, request).map_err(debug)?;
            }
        }
        let size = match args[1].as_str() {
            "codec_float" => {
                let pcm: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|sample| f32::from_le_bytes(sample.try_into().expect("four-byte sample")))
                    .collect();
                opus_encode_float(&mut encoder, &pcm, frame_size, &mut encoded)
            }
            "codec24" => {
                let pcm: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|sample| i32::from_le_bytes(sample.try_into().expect("four-byte sample")))
                    .collect();
                opus_encode24(&mut encoder, &pcm, frame_size, &mut encoded)
            }
            _ => {
                let pcm: Vec<i16> = bytes
                    .chunks_exact(2)
                    .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                    .collect();
                opus_encode(&mut encoder, &pcm, frame_size, &mut encoded)
            }
        }
        .map_err(debug)?;
        let packet = &encoded[..size];
        let samples = opus_decode(
            &mut decoder,
            Some(packet),
            size,
            &mut decoded,
            max_samples,
            false,
        )
        .map_err(debug)?;
        let float_samples = opus_decode_float(
            &mut float_decoder,
            Some(packet),
            size,
            &mut decoded_float,
            max_samples,
            false,
        )
        .map_err(debug)?;
        if samples != float_samples {
            return Err("Decoder sample counts differ.".into());
        }
        let (mut enc_range, mut dec_range, mut float_range) = (0, 0, 0);
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::GetFinalRange(&mut enc_range),
        )
        .map_err(debug)?;
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut dec_range),
        )
        .map_err(debug)?;
        opus_decoder_ctl(
            &mut float_decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut float_range),
        )
        .map_err(debug)?;
        let mut line =
            format!("C\t{index}\t{size}\t{samples}\t{enc_range}\t{dec_range}\t{float_range}\t");
        for byte in packet {
            write!(line, "{byte:02x}").map_err(debug)?;
        }
        line.push('\t');
        for sample in &decoded[..samples * channels] {
            for byte in sample.to_le_bytes() {
                write!(line, "{byte:02x}").map_err(debug)?;
            }
        }
        line.push('\t');
        for sample in &decoded_float[..samples * channels] {
            write!(line, "{:08x}", sample.to_bits()).map_err(debug)?;
        }
        writeln!(output, "{line}").map_err(debug)?;
    }
    output.flush().map_err(debug)
}

fn debug(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

fn decode(args: &[String]) -> Result<(), String> {
    let rate = args[2].parse::<i32>().map_err(debug)?;
    let channels = args[3].parse::<i32>().map_err(debug)?;
    let mut decoder = opus_decoder_create(rate, channels).map_err(debug)?;
    let mut float_decoder = opus_decoder_create(rate, channels).map_err(debug)?;
    configure_decoder(&mut decoder)?;
    configure_decoder(&mut float_decoder)?;
    let input = std::fs::read_to_string(&args[4]).map_err(debug)?;
    let mut output = io::BufWriter::new(io::stdout().lock());
    let dred_mode = args[1] == "dred_decode";
    let dred_decoder = if dred_mode {
        Some(opus_dred_decoder_create().map_err(debug)?)
    } else {
        None
    };
    let mut dred = if dred_mode {
        Some(opus_dred_alloc().map_err(debug)?)
    } else {
        None
    };
    let mut decoder24 = if dred_mode {
        let mut state = opus_decoder_create(rate, channels).map_err(debug)?;
        configure_decoder(&mut state)?;
        Some(state)
    } else {
        None
    };
    let mut index = 0;
    for command in input.lines() {
        let fields: Vec<_> = command.split_whitespace().collect();
        if fields.as_slice() == ["reset"] {
            if let Some(state) = decoder24.as_mut() {
                opus_decoder_ctl(state, OpusDecoderCtlRequest::ResetState).map_err(debug)?;
            }
            opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::ResetState).map_err(debug)?;
            opus_decoder_ctl(&mut float_decoder, OpusDecoderCtlRequest::ResetState)
                .map_err(debug)?;
            continue;
        }
        if fields.len() == 2 && fields[0] == "gain" {
            let gain = fields[1].parse::<i32>().map_err(debug)?;
            if let Some(state) = decoder24.as_mut() {
                opus_decoder_ctl(state, OpusDecoderCtlRequest::SetGain(gain)).map_err(debug)?;
            }
            opus_decoder_ctl(&mut decoder, OpusDecoderCtlRequest::SetGain(gain)).map_err(debug)?;
            opus_decoder_ctl(&mut float_decoder, OpusDecoderCtlRequest::SetGain(gain))
                .map_err(debug)?;
            continue;
        }
        if fields.len() != 3 {
            return Err(format!("Invalid decode command on line {}", index + 1));
        }
        let frame_size = fields[0].parse::<usize>().map_err(debug)?;
        if frame_size == 0 || frame_size > rate as usize * 120 / 1000 {
            return Err("Invalid decode frame size.".into());
        }
        let dred_offset = if dred_mode {
            fields[1].parse::<i32>().map_err(debug)?
        } else {
            -1
        };
        let fec = if dred_mode {
            false
        } else {
            match fields[1] {
                "0" => false,
                "1" => true,
                _ => return Err("Invalid FEC flag.".into()),
            }
        };
        let packet = if fields[2] == "-" {
            None
        } else {
            let hex = fields[2].as_bytes();
            if !hex.len().is_multiple_of(2) {
                return Err("Odd hexadecimal input length.".into());
            }
            Some(
                hex.chunks_exact(2)
                    .map(|pair| {
                        let text = std::str::from_utf8(pair).map_err(debug)?;
                        u8::from_str_radix(text, 16).map_err(debug)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let packet = packet.as_deref();
        let len = packet.map_or(0, <[u8]>::len);
        let mut pcm = vec![0i16; frame_size * channels as usize];
        let mut float_pcm = vec![0f32; frame_size * channels as usize];
        let (samples, float_samples) = if dred_mode && dred_offset >= 0 {
            let state = dred.as_mut().unwrap();
            let dred_decoder = dred_decoder.as_ref().unwrap();
            let mut dred_end = 0;
            let defer = codec_env_int("OPUS_ORACLE_DRED_DEFER", 0, 0, 1)? != 0;
            let available = opus_dred_parse(
                dred_decoder,
                state,
                packet.unwrap_or(&[]),
                rate * 104 / 100,
                rate,
                Some(&mut dred_end),
                defer,
            )
            .map_err(debug)?;
            if available <= 0 {
                return Err("Recovery packet contains no decodable DRED redundancy.".into());
            }
            if defer {
                let source = state.clone();
                opus_dred_process(dred_decoder, &source, state).map_err(debug)?;
            }
            writeln!(output, "R\t{index}\t{available}\t{dred_end}").map_err(debug)?;
            let samples =
                opus_decoder_dred_decode(&mut decoder, state, dred_offset, &mut pcm, frame_size)
                    .map_err(debug)? as i32;
            let float_samples = opus_decoder_dred_decode_float(
                &mut float_decoder,
                state,
                dred_offset,
                &mut float_pcm,
                frame_size,
            )
            .map_err(debug)? as i32;
            (samples, float_samples)
        } else {
            let samples = opus_decode(&mut decoder, packet, len, &mut pcm, frame_size, fec)
                .map_or_else(|error| error.code(), |samples| samples as i32);
            let float_samples = opus_decode_float(
                &mut float_decoder,
                packet,
                len,
                &mut float_pcm,
                frame_size,
                fec,
            )
            .map_or_else(|error| error.code(), |samples| samples as i32);
            (samples, float_samples)
        };
        if samples != float_samples {
            return Err("Decoder sample counts differ.".into());
        }
        let (mut range, mut float_range) = (0, 0);
        opus_decoder_ctl(
            &mut decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut range),
        )
        .map_err(debug)?;
        opus_decoder_ctl(
            &mut float_decoder,
            OpusDecoderCtlRequest::GetFinalRange(&mut float_range),
        )
        .map_err(debug)?;
        let mut line = format!("D\t{index}\t{samples}\t{range}\t{float_range}\t");
        for sample in &pcm[..samples.max(0) as usize * channels as usize] {
            for byte in sample.to_le_bytes() {
                write!(line, "{byte:02x}").map_err(debug)?;
            }
        }
        line.push('\t');
        for sample in &float_pcm[..samples.max(0) as usize * channels as usize] {
            write!(line, "{:08x}", sample.to_bits()).map_err(debug)?;
        }
        writeln!(output, "{line}").map_err(debug)?;
        if let Some(state) = decoder24.as_mut() {
            let mut pcm24 = vec![0i32; frame_size * channels as usize];
            let samples24 = if dred_offset >= 0 {
                opus_decoder_dred_decode24(
                    state,
                    dred.as_ref().unwrap(),
                    dred_offset,
                    &mut pcm24,
                    frame_size,
                )
                .map_err(debug)?
            } else {
                opus_decode24(state, packet, len, &mut pcm24, frame_size, false).map_err(debug)?
            };
            if samples24 as i32 != samples {
                return Err("24-bit decoder sample count differs.".into());
            }
            let mut range24 = 0;
            opus_decoder_ctl(state, OpusDecoderCtlRequest::GetFinalRange(&mut range24))
                .map_err(debug)?;
            let mut line = format!("Q\t{index}\t{samples24}\t{range24}\t");
            for &sample in &pcm24[..samples24 * channels as usize] {
                write!(line, "{:08x}", sample as u32).map_err(debug)?;
            }
            writeln!(output, "{line}").map_err(debug)?;
        }
        index += 1;
    }
    output.flush().map_err(debug)
}
