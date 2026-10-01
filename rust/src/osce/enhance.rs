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

//! LACE and NoLACE feature conditioning and adaptive enhancement networks.
use super::models::*;
use super::nndsp::*;
use crate::nnet::{
    ACTIVATION_TANH, LinearLayer, compute_generic_conv1d, compute_generic_dense,
    compute_generic_gru,
};

#[derive(Clone, Debug)]
pub(crate) struct EnhancementState {
    conv2: [f32; 384],
    gru: [f32; 160],
    post: [[f32; 160]; 5],
    comb: [AdaCombState; 2],
    conv: [AdaConvState; 4],
    shape: [AdaShapeState; 3],
    preemph: f32,
    deemph: f32,
}
impl Default for EnhancementState {
    fn default() -> Self {
        Self {
            conv2: [0.0; 384],
            gru: [0.0; 160],
            post: [[0.0; 160]; 5],
            comb: core::array::from_fn(|_| AdaCombState::default()),
            conv: core::array::from_fn(|_| AdaConvState::default()),
            shape: core::array::from_fn(|_| AdaShapeState::default()),
            preemph: 0.0,
            deemph: 0.0,
        }
    }
}

struct FeatureLayers<'a> {
    pitch: LinearLayer<'a>,
    conv1: LinearLayer<'a>,
    conv2: LinearLayer<'a>,
    tconv: LinearLayer<'a>,
    gru_input: LinearLayer<'a>,
    gru_recurrent: LinearLayer<'a>,
    scales: [f32; 8],
    dim: usize,
}
fn feature_net(
    layers: &FeatureLayers<'_>,
    state: &mut EnhancementState,
    features: &[f32; 372],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) -> [f32; 640] {
    let mut embedded = [0.0; 16];
    let low = libm::log(50.0) as f32;
    let high = libm::log(650.0) as f32;
    // The pinned C CLIP macro only enforces the upper bound. Preserve its arithmetic.
    for (bit_index, &bits) in numbits.iter().enumerate() {
        let x = (libm::log(f64::from(bits)) as f32).min(high) - (high + low) / 2.0;
        for k in 0..8 {
            embedded[bit_index * 8 + k] = libm::sin(f64::from(x * layers.scales[k] - 0.5)) as f32;
        }
    }
    let mut accumulated = [0.0; 384];
    let mut input = [0.0; 173];
    let pitch = layers
        .pitch
        .float_weights
        .expect("pitch embeddings use float weights");
    for k in 0..4 {
        input[..93].copy_from_slice(&features[k * 93..(k + 1) * 93]);
        input[93..157].copy_from_slice(&pitch[periods[k] * 64..(periods[k] + 1) * 64]);
        input[157..].copy_from_slice(&embedded);
        compute_generic_conv1d(
            &layers.conv1,
            &mut accumulated[k * 96..(k + 1) * 96],
            &mut [],
            &input,
            173,
            ACTIVATION_TANH,
            0,
        );
    }
    let dim = layers.dim;
    let mut reduced = [0.0; 160];
    compute_generic_conv1d(
        &layers.conv2,
        &mut reduced[..dim],
        &mut state.conv2,
        &accumulated,
        384,
        ACTIVATION_TANH,
        0,
    );
    let mut upsampled = [0.0; 640];
    compute_generic_dense(
        &layers.tconv,
        &mut upsampled[..4 * dim],
        &reduced[..dim],
        ACTIVATION_TANH,
        0,
    );
    let mut output = [0.0; 640];
    for k in 0..4 {
        compute_generic_gru(
            &layers.gru_input,
            &layers.gru_recurrent,
            &mut state.gru[..dim],
            &upsampled[k * dim..(k + 1) * dim],
            0,
        );
        output[k * dim..(k + 1) * dim].copy_from_slice(&state.gru[..dim]);
    }
    output
}

impl LaceLayers {
    pub(crate) fn process(
        &self,
        state: &mut EnhancementState,
        output: &mut [f32; 320],
        input: &[f32; 320],
        features: &[f32; 372],
        numbits: &[f32; 2],
        periods: &[usize; 4],
    ) {
        let layers = FeatureLayers {
            pitch: self.lace_pitch_embedding.view(),
            conv1: self.lace_fnet_conv1.view(),
            conv2: self.lace_fnet_conv2.view(),
            tconv: self.lace_fnet_tconv.view(),
            gru_input: self.lace_fnet_gru_input.view(),
            gru_recurrent: self.lace_fnet_gru_recurrent.view(),
            dim: LACE_COND_DIM,
            scales: [
                LACE_NUMBITS_SCALE_0,
                LACE_NUMBITS_SCALE_1,
                LACE_NUMBITS_SCALE_2,
                LACE_NUMBITS_SCALE_3,
                LACE_NUMBITS_SCALE_4,
                LACE_NUMBITS_SCALE_5,
                LACE_NUMBITS_SCALE_6,
                LACE_NUMBITS_SCALE_7,
            ],
        };
        let mut signal = [0.0; 320];
        for k in 0..320 {
            signal[k] = input[k] - LACE_PREEMPH * state.preemph;
            state.preemph = input[k];
        }
        let condition = feature_net(&layers, state, features, numbits, periods);
        let mut window = [0.0; 40];
        compute_overlap_window(&mut window, 40);
        let kernels = [&self.lace_cf1_kernel, &self.lace_cf2_kernel];
        let gains = [&self.lace_cf1_gain, &self.lace_cf2_gain];
        let global = [&self.lace_cf1_global_gain, &self.lace_cf2_global_gain];
        for stage in 0..2 {
            for k in 0..4 {
                let old: [f32; 80] = signal[k * 80..(k + 1) * 80].try_into().unwrap();
                adacomb_process_frame(
                    &mut state.comb[stage],
                    &mut signal[k * 80..(k + 1) * 80],
                    &old,
                    &condition[k * 128..(k + 1) * 128],
                    &kernels[stage].view(),
                    &gains[stage].view(),
                    &global[stage].view(),
                    periods[k],
                    128,
                    80,
                    40,
                    16,
                    8,
                    LACE_CF1_FILTER_GAIN_A,
                    LACE_CF1_FILTER_GAIN_B,
                    LACE_CF1_LOG_GAIN_LIMIT,
                    &window,
                );
            }
        }
        for k in 0..4 {
            let old: [f32; 80] = signal[k * 80..(k + 1) * 80].try_into().unwrap();
            adaconv_process_frame(
                &mut state.conv[0],
                &mut signal[k * 80..(k + 1) * 80],
                &old,
                &condition[k * 128..(k + 1) * 128],
                &self.lace_af1_kernel.view(),
                &self.lace_af1_gain.view(),
                128,
                80,
                40,
                1,
                1,
                16,
                15,
                LACE_AF1_FILTER_GAIN_A,
                LACE_AF1_FILTER_GAIN_B,
                LACE_AF1_SHAPE_GAIN,
                &window,
            );
        }
        for k in 0..320 {
            output[k] = signal[k] + LACE_PREEMPH * state.deemph;
            state.deemph = output[k];
        }
    }
}

impl NoLaceLayers {
    pub(crate) fn process(
        &self,
        state: &mut EnhancementState,
        output: &mut [f32; 320],
        input: &[f32; 320],
        features: &[f32; 372],
        numbits: &[f32; 2],
        periods: &[usize; 4],
    ) {
        let layers = FeatureLayers {
            pitch: self.nolace_pitch_embedding.view(),
            conv1: self.nolace_fnet_conv1.view(),
            conv2: self.nolace_fnet_conv2.view(),
            tconv: self.nolace_fnet_tconv.view(),
            gru_input: self.nolace_fnet_gru_input.view(),
            gru_recurrent: self.nolace_fnet_gru_recurrent.view(),
            dim: NOLACE_COND_DIM,
            scales: [
                NOLACE_NUMBITS_SCALE_0,
                NOLACE_NUMBITS_SCALE_1,
                NOLACE_NUMBITS_SCALE_2,
                NOLACE_NUMBITS_SCALE_3,
                NOLACE_NUMBITS_SCALE_4,
                NOLACE_NUMBITS_SCALE_5,
                NOLACE_NUMBITS_SCALE_6,
                NOLACE_NUMBITS_SCALE_7,
            ],
        };
        let mut signal = [0.0; 640];
        for k in 0..320 {
            signal[k] = input[k] - NOLACE_PREEMPH * state.preemph;
            state.preemph = input[k];
        }
        let mut condition = feature_net(&layers, state, features, numbits, periods);
        let mut window = [0.0; 40];
        compute_overlap_window(&mut window, 40);
        let kernels = [&self.nolace_cf1_kernel, &self.nolace_cf2_kernel];
        let gains = [&self.nolace_cf1_gain, &self.nolace_cf2_gain];
        let global = [&self.nolace_cf1_global_gain, &self.nolace_cf2_global_gain];
        let posts = [
            &self.nolace_post_cf1,
            &self.nolace_post_cf2,
            &self.nolace_post_af1,
            &self.nolace_post_af2,
            &self.nolace_post_af3,
        ];
        for stage in 0..2 {
            let mut transformed = [0.0; 640];
            for k in 0..4 {
                let old: [f32; 80] = signal[k * 80..(k + 1) * 80].try_into().unwrap();
                adacomb_process_frame(
                    &mut state.comb[stage],
                    &mut signal[k * 80..(k + 1) * 80],
                    &old,
                    &condition[k * 160..(k + 1) * 160],
                    &kernels[stage].view(),
                    &gains[stage].view(),
                    &global[stage].view(),
                    periods[k],
                    160,
                    80,
                    40,
                    16,
                    8,
                    NOLACE_CF1_FILTER_GAIN_A,
                    NOLACE_CF1_FILTER_GAIN_B,
                    NOLACE_CF1_LOG_GAIN_LIMIT,
                    &window,
                );
                compute_generic_conv1d(
                    &posts[stage].view(),
                    &mut transformed[k * 160..(k + 1) * 160],
                    &mut state.post[stage],
                    &condition[k * 160..(k + 1) * 160],
                    160,
                    ACTIVATION_TANH,
                    0,
                );
            }
            condition = transformed;
        }
        let kernels = [
            &self.nolace_af1_kernel,
            &self.nolace_af2_kernel,
            &self.nolace_af3_kernel,
            &self.nolace_af4_kernel,
        ];
        let gains = [
            &self.nolace_af1_gain,
            &self.nolace_af2_gain,
            &self.nolace_af3_gain,
            &self.nolace_af4_gain,
        ];
        let shape_f = [
            &self.nolace_tdshape1_alpha1_f,
            &self.nolace_tdshape2_alpha1_f,
            &self.nolace_tdshape3_alpha1_f,
        ];
        let shape_t = [
            &self.nolace_tdshape1_alpha1_t,
            &self.nolace_tdshape2_alpha1_t,
            &self.nolace_tdshape3_alpha1_t,
        ];
        let shape_2 = [
            &self.nolace_tdshape1_alpha2,
            &self.nolace_tdshape2_alpha2,
            &self.nolace_tdshape3_alpha2,
        ];
        for stage in 0..4 {
            let in_channels = if stage == 0 { 1 } else { 2 };
            let out_channels = if stage == 3 { 1 } else { 2 };
            let mut next = [0.0; 640];
            let mut transformed = [0.0; 640];
            for k in 0..4 {
                if stage > 0 {
                    let start = k * 160 + 80;
                    let old: [f32; 80] = signal[start..start + 80].try_into().unwrap();
                    adashape_process_frame(
                        &mut state.shape[stage - 1],
                        &mut signal[start..start + 80],
                        &old,
                        &condition[k * 160..(k + 1) * 160],
                        &shape_f[stage - 1].view(),
                        &shape_t[stage - 1].view(),
                        &shape_2[stage - 1].view(),
                        160,
                        80,
                        4,
                        1,
                    );
                }
                adaconv_process_frame(
                    &mut state.conv[stage],
                    &mut next[k * 80 * out_channels..(k + 1) * 80 * out_channels],
                    &signal[k * 80 * in_channels..(k + 1) * 80 * in_channels],
                    &condition[k * 160..(k + 1) * 160],
                    &kernels[stage].view(),
                    &gains[stage].view(),
                    160,
                    80,
                    40,
                    in_channels,
                    out_channels,
                    16,
                    15,
                    NOLACE_AF1_FILTER_GAIN_A,
                    NOLACE_AF1_FILTER_GAIN_B,
                    NOLACE_AF1_SHAPE_GAIN,
                    &window,
                );
                if stage < 3 {
                    compute_generic_conv1d(
                        &posts[stage + 2].view(),
                        &mut transformed[k * 160..(k + 1) * 160],
                        &mut state.post[stage + 2],
                        &condition[k * 160..(k + 1) * 160],
                        160,
                        ACTIVATION_TANH,
                        0,
                    );
                }
            }
            signal = next;
            condition = transformed;
        }
        for k in 0..320 {
            output[k] = signal[k] + NOLACE_PREEMPH * state.deemph;
            state.deemph = output[k];
        }
    }
}
