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

//! Scalar adaptive filtering for the Opus speech coding enhancer.

use crate::celt::{celt_log2, celt_pitch_xcorr};
use crate::nnet::{
    ACTIVATION_EXP, ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_TANH, LinearLayer,
    compute_activation, compute_generic_conv1d, compute_generic_dense,
};

pub(crate) const ADACONV_MAX_KERNEL_SIZE: usize = 32;
pub(crate) const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
pub(crate) const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
pub(crate) const ADACONV_MAX_FRAME_SIZE: usize = 240;
pub(crate) const ADACONV_MAX_OVERLAP_SIZE: usize = 120;
pub(crate) const ADACOMB_MAX_LAG: usize = 300;
pub(crate) const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
pub(crate) const ADACOMB_MAX_FRAME_SIZE: usize = 80;
pub(crate) const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;
pub(crate) const ADASHAPE_MAX_INPUT_DIM: usize = 512;
pub(crate) const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

#[derive(Clone, Debug)]
pub(crate) struct AdaConvState {
    pub history: [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
    pub last_kernel:
        [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS],
    pub last_gain: f32,
}
impl Default for AdaConvState {
    fn default() -> Self {
        Self {
            history: [0.0; 96],
            last_kernel: [0.0; 288],
            last_gain: 0.0,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct AdaCombState {
    pub history: [f32; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
    pub last_kernel: [f32; ADACOMB_MAX_KERNEL_SIZE],
    pub last_global_gain: f32,
    pub last_pitch_lag: usize,
}
impl Default for AdaCombState {
    fn default() -> Self {
        Self {
            history: [0.0; 316],
            last_kernel: [0.0; 16],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct AdaShapeState {
    pub conv_alpha1f_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha1t_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha2_state: [f32; ADASHAPE_MAX_FRAME_SIZE],
    pub interpolate_state: [f32; 1],
}
impl Default for AdaShapeState {
    fn default() -> Self {
        Self {
            conv_alpha1f_state: [0.0; 512],
            conv_alpha1t_state: [0.0; 512],
            conv_alpha2_state: [0.0; 240],
            interpolate_state: [0.0],
        }
    }
}

pub(crate) fn compute_overlap_window(window: &mut [f32], overlap_size: usize) {
    assert!(window.len() >= overlap_size);
    for (index, value) in window[..overlap_size].iter_mut().enumerate() {
        // nndsp.c's strict C99 profile uses its float M_PI fallback before cos(double).
        let angle = core::f32::consts::PI * (index as f32 + 0.5) / overlap_size as f32;
        *value = (0.5 + 0.5 * libm::cos(f64::from(angle))) as f32;
    }
}

fn scale_kernel(
    kernel: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    gain: &[f32],
) {
    let channel_size = in_channels * kernel_size;
    for channel in 0..out_channels {
        let channel_kernel = &mut kernel[channel * channel_size..(channel + 1) * channel_size];
        let mut norm = 0.0f32;
        for &value in channel_kernel.iter() {
            norm += value * value;
        }
        let scale = (1.0 / (f64::from(1e-6f32) + libm::sqrt(f64::from(norm)))) as f32;
        for value in channel_kernel {
            *value *= scale * gain[channel];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adaconv_process_frame(
    state: &mut AdaConvState,
    output: &mut [f32],
    input: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer<'_>,
    gain_layer: &LinearLayer<'_>,
    feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    shape_gain: f32,
    window: &[f32],
) {
    assert_eq!(shape_gain, 1.0);
    assert!((1..=ADACONV_MAX_KERNEL_SIZE).contains(&kernel_size));
    assert_eq!(left_padding, kernel_size - 1);
    assert!(kernel_size < frame_size && frame_size <= ADACONV_MAX_FRAME_SIZE);
    assert!(overlap_size <= frame_size && overlap_size <= ADACONV_MAX_OVERLAP_SIZE);
    assert!((1..=3).contains(&in_channels) && (1..=3).contains(&out_channels));
    assert!(input.len() >= in_channels * frame_size && output.len() >= out_channels * frame_size);
    assert!(window.len() >= overlap_size && features.len() >= feature_dim);
    let mut output_buffer = [0.0; 720];
    let mut kernel_buffer = [0.0; 288];
    let mut input_buffer = [0.0; 816];
    let mut gains = [0.0; 3];
    for channel in 0..in_channels {
        let start = channel * (kernel_size + frame_size);
        input_buffer[start..start + kernel_size]
            .copy_from_slice(&state.history[channel * kernel_size..(channel + 1) * kernel_size]);
        input_buffer[start + kernel_size..start + kernel_size + frame_size]
            .copy_from_slice(&input[channel * frame_size..(channel + 1) * frame_size]);
    }
    let kernel_count = in_channels * out_channels * kernel_size;
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer[..kernel_count],
        &features[..feature_dim],
        ACTIVATION_LINEAR,
        0,
    );
    compute_generic_dense(
        gain_layer,
        &mut gains[..out_channels],
        &features[..feature_dim],
        ACTIVATION_TANH,
        0,
    );
    for gain in &mut gains[..out_channels] {
        *gain = libm::exp(f64::from(filter_gain_a * *gain + filter_gain_b)) as f32;
    }
    scale_kernel(
        &mut kernel_buffer,
        in_channels,
        out_channels,
        kernel_size,
        &gains,
    );
    for out_channel in 0..out_channels {
        for in_channel in 0..in_channels {
            let index = (out_channel * in_channels + in_channel) * kernel_size;
            let mut previous_kernel = [0.0; ADACONV_MAX_KERNEL_SIZE];
            let mut current_kernel = [0.0; ADACONV_MAX_KERNEL_SIZE];
            previous_kernel[..kernel_size]
                .copy_from_slice(&state.last_kernel[index..index + kernel_size]);
            current_kernel[..kernel_size]
                .copy_from_slice(&kernel_buffer[index..index + kernel_size]);
            let input_start = kernel_size + in_channel * (frame_size + kernel_size) - left_padding;
            let mut previous = [0.0; ADACONV_MAX_OVERLAP_SIZE];
            let mut current = [0.0; ADACONV_MAX_FRAME_SIZE];
            celt_pitch_xcorr(
                &previous_kernel,
                &input_buffer[input_start..],
                ADACONV_MAX_KERNEL_SIZE,
                overlap_size,
                &mut previous,
            );
            celt_pitch_xcorr(
                &current_kernel,
                &input_buffer[input_start..],
                ADACONV_MAX_KERNEL_SIZE,
                frame_size,
                &mut current,
            );
            for sample in 0..overlap_size {
                output_buffer[sample + out_channel * frame_size] +=
                    window[sample] * previous[sample];
                output_buffer[sample + out_channel * frame_size] +=
                    (1.0 - window[sample]) * current[sample];
            }
            for sample in overlap_size..frame_size {
                output_buffer[sample + out_channel * frame_size] += current[sample];
            }
        }
    }
    output[..out_channels * frame_size]
        .copy_from_slice(&output_buffer[..out_channels * frame_size]);
    for channel in 0..in_channels {
        let start = channel * (frame_size + kernel_size) + frame_size;
        state.history[channel * kernel_size..(channel + 1) * kernel_size]
            .copy_from_slice(&input_buffer[start..start + kernel_size]);
    }
    state.last_kernel[..kernel_count].copy_from_slice(&kernel_buffer[..kernel_count]);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adacomb_process_frame(
    state: &mut AdaCombState,
    output: &mut [f32],
    input: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer<'_>,
    gain_layer: &LinearLayer<'_>,
    global_gain_layer: &LinearLayer<'_>,
    pitch_lag: usize,
    feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    log_gain_limit: f32,
    window: &[f32],
) {
    assert!((1..=ADACOMB_MAX_KERNEL_SIZE).contains(&kernel_size));
    assert!(
        frame_size <= ADACOMB_MAX_FRAME_SIZE
            && overlap_size <= frame_size
            && overlap_size <= ADACOMB_MAX_OVERLAP_SIZE
    );
    assert!(pitch_lag <= ADACOMB_MAX_LAG && state.last_pitch_lag <= ADACOMB_MAX_LAG);
    assert!(left_padding < kernel_size);
    assert!(
        input.len() >= frame_size && output.len() >= frame_size && window.len() >= overlap_size
    );
    let history_size = kernel_size + ADACOMB_MAX_LAG;
    let mut input_buffer = [0.0; 396];
    input_buffer[..history_size].copy_from_slice(&state.history[..history_size]);
    input_buffer[history_size..history_size + frame_size].copy_from_slice(&input[..frame_size]);
    let mut kernel = [0.0; ADACOMB_MAX_KERNEL_SIZE];
    let mut gain = [0.0];
    let mut global_gain = [0.0];
    compute_generic_dense(
        kernel_layer,
        &mut kernel[..kernel_size],
        &features[..feature_dim],
        ACTIVATION_LINEAR,
        0,
    );
    compute_generic_dense(
        gain_layer,
        &mut gain,
        &features[..feature_dim],
        ACTIVATION_RELU,
        0,
    );
    compute_generic_dense(
        global_gain_layer,
        &mut global_gain,
        &features[..feature_dim],
        ACTIVATION_TANH,
        0,
    );
    gain[0] = libm::exp(f64::from(log_gain_limit - gain[0])) as f32;
    global_gain[0] = libm::exp(f64::from(filter_gain_a * global_gain[0] + filter_gain_b)) as f32;
    scale_kernel(&mut kernel, 1, 1, kernel_size, &gain);
    let mut last_kernel = [0.0; ADACOMB_MAX_KERNEL_SIZE];
    last_kernel[..kernel_size].copy_from_slice(&state.last_kernel[..kernel_size]);
    let mut previous = [0.0; ADACOMB_MAX_FRAME_SIZE];
    let mut current = [0.0; ADACOMB_MAX_FRAME_SIZE];
    celt_pitch_xcorr(
        &last_kernel,
        &input_buffer[history_size - left_padding - state.last_pitch_lag..],
        ADACOMB_MAX_KERNEL_SIZE,
        overlap_size,
        &mut previous,
    );
    celt_pitch_xcorr(
        &kernel,
        &input_buffer[history_size - left_padding - pitch_lag..],
        ADACOMB_MAX_KERNEL_SIZE,
        frame_size,
        &mut current,
    );
    for sample in 0..overlap_size {
        current[sample] = state.last_global_gain * window[sample] * previous[sample]
            + global_gain[0] * (1.0 - window[sample]) * current[sample];
    }
    for sample in 0..overlap_size {
        current[sample] += (window[sample] * state.last_global_gain
            + (1.0 - window[sample]) * global_gain[0])
            * input_buffer[history_size + sample];
    }
    for sample in overlap_size..frame_size {
        current[sample] = global_gain[0] * (current[sample] + input_buffer[history_size + sample]);
    }
    output[..frame_size].copy_from_slice(&current[..frame_size]);
    state.last_kernel[..kernel_size].copy_from_slice(&kernel[..kernel_size]);
    state.history[..history_size]
        .copy_from_slice(&input_buffer[frame_size..frame_size + history_size]);
    state.last_pitch_lag = pitch_lag;
    state.last_global_gain = global_gain[0];
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adashape_process_frame(
    state: &mut AdaShapeState,
    output: &mut [f32],
    input: &[f32],
    features: &[f32],
    alpha1f: &LinearLayer<'_>,
    alpha1t: &LinearLayer<'_>,
    alpha2: &LinearLayer<'_>,
    feature_dim: usize,
    frame_size: usize,
    avg_pool_k: usize,
    interpolate_k: usize,
) {
    assert!(
        avg_pool_k > 0
            && interpolate_k > 0
            && frame_size > 0
            && frame_size <= ADASHAPE_MAX_FRAME_SIZE
    );
    assert_eq!(frame_size % avg_pool_k, 0);
    assert_eq!(frame_size % interpolate_k, 0);
    let hidden_dim = frame_size / interpolate_k;
    let envelope_size = frame_size / avg_pool_k;
    assert!(feature_dim + envelope_size + 1 < ADASHAPE_MAX_INPUT_DIM);
    assert!(
        features.len() >= feature_dim && input.len() >= frame_size && output.len() >= frame_size
    );
    let mut in_buffer = [0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut envelope = [0.0f32; ADASHAPE_MAX_FRAME_SIZE + 1];
    let mut out_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut temporary = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);
    let mut mean = 0.0f32;
    let inverse_pool = 1.0 / avg_pool_k as f32;
    for index in 0..envelope_size {
        for sample in 0..avg_pool_k {
            envelope[index] += input[index * avg_pool_k + sample].abs();
        }
        envelope[index] = celt_log2(envelope[index] * inverse_pool + 1.525_878_906_25e-5)
            * core::f32::consts::LN_2;
        mean += envelope[index];
    }
    mean /= envelope_size as f32;
    for value in &mut envelope[..envelope_size] {
        *value -= mean;
    }
    envelope[envelope_size] = mean;
    compute_generic_conv1d(
        alpha1f,
        &mut out_buffer[..hidden_dim],
        &mut state.conv_alpha1f_state,
        &in_buffer[..feature_dim],
        feature_dim,
        ACTIVATION_LINEAR,
        0,
    );
    compute_generic_conv1d(
        alpha1t,
        &mut temporary[..hidden_dim],
        &mut state.conv_alpha1t_state,
        &envelope[..envelope_size + 1],
        envelope_size + 1,
        ACTIVATION_LINEAR,
        0,
    );
    for index in 0..hidden_dim {
        let value = out_buffer[index] + temporary[index];
        in_buffer[index] = if value >= 0.0 {
            value
        } else {
            (0.2 * f64::from(value)) as f32
        };
    }
    compute_generic_conv1d(
        alpha2,
        &mut temporary[..hidden_dim],
        &mut state.conv_alpha2_state,
        &in_buffer[..hidden_dim],
        hidden_dim,
        ACTIVATION_LINEAR,
        0,
    );
    for index in 0..hidden_dim {
        for sample in 0..interpolate_k {
            let alpha = (sample + 1) as f32 / interpolate_k as f32;
            out_buffer[index * interpolate_k + sample] =
                alpha * temporary[index] + (1.0 - alpha) * state.interpolate_state[0];
        }
        state.interpolate_state[0] = temporary[index];
    }
    compute_activation(&mut out_buffer[..frame_size], ACTIVATION_EXP);
    for sample in 0..frame_size {
        output[sample] = out_buffer[sample] * input[sample];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct TestLayer {
        weights: Vec<f32>,
        bias: Vec<f32>,
        inputs: usize,
        outputs: usize,
    }
    impl TestLayer {
        fn new(inputs: usize, outputs: usize) -> Self {
            Self {
                weights: (0..inputs * outputs)
                    .map(|i| ((i * 17 + 3) % 61) as f32 / 256.0 - 30.0 / 256.0)
                    .collect(),
                bias: (0..outputs)
                    .map(|i| (i as i32 % 7 - 3) as f32 / 16.0)
                    .collect(),
                inputs,
                outputs,
            }
        }
        fn view(&self) -> LinearLayer<'_> {
            LinearLayer {
                bias: Some(&self.bias),
                float_weights: Some(&self.weights),
                nb_inputs: self.inputs,
                nb_outputs: self.outputs,
                ..LinearLayer::default()
            }
        }
    }

    #[test]
    fn matches_pinned_scalar_c_adaptive_filters() {
        let fixture = include_str!("../../tests/fixtures/reference/nndsp-f32.txt");
        let mut expected = fixture.split_whitespace();
        let mut compare = |values: &[f32], stage: &str, frame: usize| {
            for (sample, value) in values.iter().enumerate() {
                let bits = u32::from_str_radix(expected.next().expect("C output"), 16).unwrap();
                assert_eq!(
                    value.to_bits(),
                    bits,
                    "{stage}, frame {frame}, sample {sample}"
                );
            }
        };
        let kernel = TestLayer::new(8, 30);
        let gain = TestLayer::new(8, 3);
        let comb_kernel = TestLayer::new(8, 5);
        let comb_gain = TestLayer::new(8, 1);
        let comb_global_gain = TestLayer::new(8, 1);
        let alpha1f = TestLayer::new(16, 20);
        let alpha1t = TestLayer::new(22, 20);
        let alpha2 = TestLayer::new(40, 20);
        let mut conv = AdaConvState::default();
        let mut comb = AdaCombState::default();
        let mut shape = AdaShapeState::default();
        let mut window = [0.0; 10];
        compute_overlap_window(&mut window, 10);
        compare(&window, "window", 0);
        for frame in 0..4 {
            let input: Vec<f32> = (0..80)
                .map(|i| (((i + frame * 40) * 13) % 251) as f32 / 128.0 - 125.0 / 128.0)
                .collect();
            let features: Vec<f32> = (0..8)
                .map(|i| (((i + frame) * 7) % 31) as f32 / 16.0 - 15.0 / 16.0)
                .collect();
            let mut output = [0.0; 120];
            adaconv_process_frame(
                &mut conv,
                &mut output,
                &input,
                &features,
                &kernel.view(),
                &gain.view(),
                8,
                40,
                10,
                2,
                3,
                5,
                4,
                0.5,
                -0.2,
                1.0,
                &window,
            );
            compare(&output, "convolution", frame);
            adacomb_process_frame(
                &mut comb,
                &mut output,
                &input,
                &features,
                &comb_kernel.view(),
                &comb_gain.view(),
                &comb_global_gain.view(),
                32 + frame * 3,
                8,
                40,
                10,
                5,
                2,
                0.5,
                -0.2,
                0.1,
                &window,
            );
            compare(&output[..40], "comb", frame);
            adashape_process_frame(
                &mut shape,
                &mut output,
                &input,
                &features,
                &alpha1f.view(),
                &alpha1t.view(),
                &alpha2.view(),
                8,
                40,
                4,
                2,
            );
            compare(&output[..40], "shape", frame);
        }
        assert!(expected.next().is_none());
    }
}
