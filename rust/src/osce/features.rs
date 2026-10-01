/* Copyright (c) 2023 Amazon
Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//! Feature extraction and transition windows for the Opus speech enhancer.
use super::tables::*;
use crate::celt::KissFftCpx;
use crate::silk::{FrameSignalType, decoder_control::DecoderControl, decoder_state::DecoderState};

#[derive(Clone, Debug)]
pub(crate) struct OsceFeatures {
    history: [f32; 350],
    numbits_smooth: f32,
    pub(crate) reset: i32,
}
impl Default for OsceFeatures {
    fn default() -> Self {
        Self {
            history: [0.0; 350],
            numbits_smooth: 0.0,
            reset: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct BweFeatures {
    history: [f32; 160],
    last_spec: [f32; 82],
}
impl Default for BweFeatures {
    fn default() -> Self {
        Self {
            history: [0.0; 160],
            last_spec: core::array::from_fn(|k| if k % 2 == 0 { 1e-9 } else { 0.0 }),
        }
    }
}
fn transform(input: &[f32]) -> [KissFftCpx; 320] {
    let x = core::array::from_fn::<_, 320, _>(|k| KissFftCpx {
        r: input[k],
        i: 0.0,
    });
    let mut y = [KissFftCpx::default(); 320];
    fft().fft(&x, &mut y);
    y
}
fn magnitude(input: &[KissFftCpx; 320]) -> [f32; 161] {
    core::array::from_fn(|k| {
        (320.0 * libm::sqrt(f64::from(input[k].r * input[k].r + input[k].i * input[k].i))) as f32
    })
}
fn filterbank(output: &mut [f32], input: &[f32], centers: &[usize], weights: &[f32]) {
    output[0] = 0.0;
    for b in 0..output.len() - 1 {
        output[b + 1] = 0.0;
        for i in centers[b]..centers[b + 1] {
            let frac = (centers[b + 1] - i) as f32 / (centers[b + 1] - centers[b]) as f32;
            output[b] += weights[b] * frac * input[i];
            output[b + 1] += weights[b + 1] * (1.0 - frac) * input[i];
        }
    }
    let last = output.len() - 1;
    output[last] += weights[last] * input[centers[last]];
}
fn lpc_spectrum(output: &mut [f32], coeffs: &[i16]) {
    let mut buffer = [0.0; 320];
    buffer[0] = 1.0;
    for (i, &a) in coeffs.iter().enumerate() {
        buffer[i + 1] = -f32::from(a) / 4096.0;
    }
    let mut mag = magnitude(&transform(&buffer));
    for value in &mut mag {
        *value = 1.0 / (*value + 1e-9f32);
    }
    filterbank(output, &mag, &CENTER_BINS_CLEAN, &BAND_WEIGHTS_CLEAN);
    for value in output {
        *value = (f64::from(0.3f32) * libm::log(f64::from(*value + 1e-9f32))) as f32;
    }
}
fn cepstrum(output: &mut [f32], input: &[f32]) {
    let windowed = core::array::from_fn::<_, 320, _>(|k| OSCE_WINDOW[k] * input[k]);
    let mag = magnitude(&transform(&windowed));
    let mut bands = [0.0; 18];
    filterbank(&mut bands, &mag, &CENTER_BINS_NOISY, &BAND_WEIGHTS_NOISY);
    for value in &mut bands {
        *value = libm::log(f64::from(*value + 1e-9f32)) as f32;
    }
    for i in 0..18 {
        let mut sum = 0.0f32;
        for j in 0..18 {
            sum += bands[j] * DCT_TABLE[j * 18 + i];
        }
        output[i] = (f64::from(sum) * libm::sqrt(2.0 / 18.0)) as f32;
    }
}
pub(crate) fn calculate(
    state: &mut OsceFeatures,
    decoder: &DecoderState,
    control: &DecoderControl,
    pcm: &[i16],
    num_bits: i32,
) -> ([f32; 372], [f32; 2], [usize; 4]) {
    let frames = decoder.sample_rate.nb_subfr;
    let samples = frames * 80;
    let mut buffer = [0.0; 670];
    buffer[..350].copy_from_slice(&state.history);
    for n in 0..samples {
        buffer[350 + n] = f32::from(pcm[n]) / 32768.0;
    }
    state.numbits_smooth = 0.9f32 * state.numbits_smooth + 0.1f32 * num_bits as f32;
    let bits = [num_bits as f32, state.numbits_smooth];
    let mut features = [0.0; 372];
    let mut periods = [0; 4];
    for k in 0..frames {
        let offset = 350 + k * 80;
        if k % 2 == 0 {
            lpc_spectrum(
                &mut features[k * 93..k * 93 + 64],
                &control.pred_coef_q12[k >> 1][..decoder.sample_rate.lpc_order],
            );
            cepstrum(
                &mut features[k * 93 + 64..k * 93 + 82],
                &buffer[offset - 160..offset + 160],
            );
        } else {
            features.copy_within((k - 1) * 93..(k - 1) * 93 + 82, k * 93);
        }
        periods[k] = if decoder.indices.signal_type == FrameSignalType::Voiced {
            control.pitch_l[k] as usize
        } else {
            7
        };
        for shift in 0..5 {
            let mut xx = 0.0f32;
            let mut yy = 0.0f32;
            let mut xy = 0.0f32;
            for n in 0..80 {
                let a = buffer[offset + n];
                let b = buffer[offset + n - periods[k] + shift - 2];
                xx += a * a;
                yy += b * b;
                xy += a * b;
            }
            features[k * 93 + 82 + shift] =
                (f64::from(xy) / libm::sqrt(f64::from(xx * yy + 1e-9f32))) as f32;
        }
        for i in 0..5 {
            features[k * 93 + 87 + i] = f32::from(control.ltp_coef_q14[k * 5 + i]) / 16384.0;
        }
        features[k * 93 + 92] =
            libm::log(f64::from(control.gains_q16[k] as f32 / 65536.0 + 1e-9f32)) as f32;
    }
    state
        .history
        .copy_from_slice(&buffer[samples..samples + 350]);
    (features, bits, periods)
}
pub(crate) fn calculate_bwe(state: &mut BweFeatures, pcm: &[i16]) -> [f32; 228] {
    assert!(pcm.len() <= 320 && pcm.len() % 160 == 0);
    let mut features = [0.0; 228];
    for (frame, input) in pcm.chunks_exact(160).enumerate() {
        let mut buffer = [0.0; 320];
        buffer[..160].copy_from_slice(&state.history);
        for n in 0..160 {
            buffer[160 + n] = f32::from(input[n]) / 32768.0;
        }
        state.history.copy_from_slice(&buffer[160..]);
        for n in 0..320 {
            buffer[n] *= OSCE_WINDOW[n];
        }
        let spectrum = transform(&buffer);
        let mut spec = [0.0; 82];
        for k in 0..41 {
            spec[2 * k] = (f64::from(320.0 * spectrum[k].r) + 1e-9) as f32;
            spec[2 * k + 1] = 320.0 * spectrum[k].i;
            let re =
                spec[2 * k] * state.last_spec[2 * k] + spec[2 * k + 1] * state.last_spec[2 * k + 1];
            let im =
                spec[2 * k + 1] * state.last_spec[2 * k] - spec[2 * k] * state.last_spec[2 * k + 1];
            let abs = libm::sqrt(f64::from(re * re + im * im)) as f32;
            features[frame * 114 + 32 + k] = (f64::from(re) / (f64::from(abs) + 1e-9)) as f32;
            features[frame * 114 + 73 + k] = (f64::from(im) / (f64::from(abs) + 1e-9)) as f32;
        }
        let bands = &mut features[frame * 114..frame * 114 + 32];
        filterbank(
            bands,
            &magnitude(&spectrum),
            &CENTER_BINS_BWE,
            &BAND_WEIGHTS_BWE,
        );
        for value in bands {
            *value = libm::log(f64::from(*value) + 1e-9) as f32;
        }
        state.last_spec = spec;
    }
    features
}
pub(crate) fn crossfade(output: &mut [f32], input: &[f32]) {
    for i in 0..160 {
        output[i] = OSCE_WINDOW[i] * output[i] + (1.0 - OSCE_WINDOW[i]) * input[i];
    }
}
pub(crate) fn crossfade_bwe(output: &mut [i16], input: &[i16]) {
    for i in 0..160 {
        let diff = if i == 159 {
            0.0
        } else {
            OSCE_WINDOW[i + 1] - OSCE_WINDOW[i]
        };
        let mut weight = OSCE_WINDOW[i];
        for j in 0..3 {
            let n = 3 * i + j;
            let sum = weight * f32::from(output[n]) + (1.0 - weight) * f32::from(input[n]);
            output[n] = (f64::from(sum) + 0.5) as i32 as i16;
            weight += diff * (1.0f32 / 3.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    fn check(line: &str, output: &[f32]) {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len() - 2, output.len());
        for (k, &value) in output.iter().enumerate() {
            let expected = u32::from_str_radix(fields[k + 2], 16).unwrap();
            assert_eq!(
                value.to_bits(),
                expected,
                "{} frame {} sample {}: Rust {}, C {}",
                fields[0],
                fields[1],
                k,
                value,
                f32::from_bits(expected)
            );
        }
    }
    #[test]
    fn features_match_pinned_scalar_c() {
        let mut decoder = DecoderState::default();
        let mut control = DecoderControl::default();
        let mut state = OsceFeatures::default();
        let mut bwe = BweFeatures::default();
        let mut lines = include_str!("../../tests/fixtures/reference/osce-features.txt").lines();
        for frame in 0..8usize {
            decoder.sample_rate.nb_subfr = if frame % 3 == 1 { 2 } else { 4 };
            decoder.sample_rate.lpc_order = if frame % 2 == 1 { 10 } else { 16 };
            decoder.indices.signal_type = match frame % 3 {
                0 => FrameSignalType::Inactive,
                1 => FrameSignalType::Unvoiced,
                _ => FrameSignalType::Voiced,
            };
            let pcm = core::array::from_fn::<_, 320, _>(|i| {
                (((i * 359 + frame * 311) % 60001) as i32 - 30000) as i16
            });
            for k in 0..2 {
                for i in 0..16 {
                    control.pred_coef_q12[k][i] =
                        ((i * 73 + k * 97 + frame * 31) % 701) as i16 - 350;
                }
            }
            for k in 0..4 {
                control.pitch_l[k] = (40 + (k * 31 + frame * 7) % 160) as i32;
                control.gains_q16[k] = (17371 + k * 13791 + frame * 578) as i32;
            }
            for i in 0..20 {
                control.ltp_coef_q14[i] = ((i * 719 + frame * 991) % 16001) as i16 - 8000;
            }
            let (features, bits, periods) = calculate(
                &mut state,
                &decoder,
                &control,
                &pcm,
                (90 + frame * 57) as i32,
            );
            check(lines.next().unwrap(), &features);
            check(lines.next().unwrap(), &bits);
            let expected: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .skip(2)
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(periods.as_slice(), expected);
            check(
                lines.next().unwrap(),
                &calculate_bwe(&mut bwe, &pcm[..decoder.sample_rate.nb_subfr * 80]),
            );
        }
        let mut fade =
            core::array::from_fn::<_, 160, _>(|i| ((i * 13 % 251) as i32 - 125) as f32 / 128.0);
        let original =
            core::array::from_fn::<_, 160, _>(|i| ((i * 17 % 127) as i32 - 63) as f32 / 64.0);
        crossfade(&mut fade, &original);
        check(lines.next().unwrap(), &fade);
        let mut fade =
            core::array::from_fn::<_, 480, _>(|i| ((i * 137 % 60001) as i32 - 30000) as i16);
        let original =
            core::array::from_fn::<_, 480, _>(|i| ((i * 177 % 55001) as i32 - 27500) as i16);
        crossfade_bwe(&mut fade, &original);
        let expected: Vec<i16> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .skip(2)
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(fade.as_slice(), expected);
        assert!(lines.next().is_none());
    }
}
