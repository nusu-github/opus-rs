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

//! BBWENet conditioning, adaptive bandwidth extension, and float resampling.
use super::models::*;
use super::nndsp::*;
use crate::nnet::{
    ACTIVATION_TANH, compute_generic_conv1d, compute_generic_dense, compute_generic_gru,
};

#[derive(Clone, Debug, Default)]
struct ResamplerState {
    upsample: [[f32; 3]; 2],
    interpolate: [f32; 8],
}
#[derive(Clone, Debug)]
pub(crate) struct BweState {
    conv1: [f32; 228],
    conv2: [f32; 256],
    gru: [f32; 128],
    conv: [AdaConvState; 3],
    shape: [AdaShapeState; 2],
    resampler: [ResamplerState; 3],
    pub(crate) delay: [i16; 21],
}
impl Default for BweState {
    fn default() -> Self {
        Self {
            conv1: [0.0; 228],
            conv2: [0.0; 256],
            gru: [0.0; 128],
            conv: core::array::from_fn(|_| AdaConvState::default()),
            shape: core::array::from_fn(|_| AdaShapeState::default()),
            resampler: core::array::from_fn(|_| ResamplerState::default()),
            delay: [0; 21],
        }
    }
}
impl ResamplerState {
    fn upsample(&mut self, output: &mut [f32], input: &[f32]) {
        let coefficients = [
            [0.026641845703125, 0.228668212890625, -0.4036407470703125],
            [0.104583740234375, 0.3932037353515625, -0.152496337890625],
        ];
        for (k, &sample) in input.iter().enumerate() {
            for phase in 0..2 {
                let mut x = sample;
                for tap in 0..3 {
                    let coefficient = coefficients[phase][tap] + if tap == 2 { 1.0 } else { 0.0 };
                    let y = (x - self.upsample[phase][tap]) * coefficient;
                    let next = self.upsample[phase][tap] + y;
                    self.upsample[phase][tap] = x + y;
                    x = next;
                }
                output[2 * k + phase] = x;
            }
        }
    }
    fn interpolate(&mut self, output: &mut [f32], input: &[f32]) {
        const TAPS: [[f32; 8]; 3] = [
            [
                0.00576782,
                -0.01831055,
                0.01882935,
                0.9328308,
                0.09143066,
                -0.04196167,
                0.01296997,
                -0.00140381,
            ],
            [
                -3.14331055e-3,
                2.73437500e-2,
                -1.06414795e-1,
                3.64685059e-1,
                8.03863525e-1,
                -1.02233887e-1,
                1.61437988e-2,
                -1.22070312e-4,
            ],
            [
                -0.00146484,
                0.02313232,
                -0.12072754,
                0.7315979,
                0.4621277,
                -0.12075806,
                0.0295105,
                -0.00326538,
            ],
        ];
        let mut buffer = [0.0; 648];
        buffer[..8].copy_from_slice(&self.interpolate);
        buffer[8..8 + input.len()].copy_from_slice(input);
        for k in (0..input.len()).step_by(2) {
            for phase in 0..3 {
                let offset = k + usize::from(phase == 2);
                let mut sum = buffer[offset] * TAPS[phase][0];
                for tap in 1..8 {
                    sum += buffer[offset + tap] * TAPS[phase][tap];
                }
                output[k / 2 * 3 + phase] = sum;
            }
        }
        self.interpolate
            .copy_from_slice(&buffer[input.len()..input.len() + 8]);
    }
}
fn cos_norm2(mut x: f32) -> f32 {
    x = (f64::from(x) - 4.0 * libm::floor(0.25 * f64::from(x + 1.0))) as f32;
    let sign = if x > 1.0 { -1.0 } else { 1.0 };
    x -= if x > 1.0 { 2.0 } else { 0.0 };
    let xx = x * x;
    sign * (9.99999940395355224609375e-1f32
        + xx * (-1.23369824886322021484375f32
            + xx * (2.536507546901702880859375e-1f32
                + xx * (-2.081062830984592437744140625e-2f32
                    + xx * 8.581906440667808055877685546875e-4f32))))
}
fn valin_activation(signal: &mut [f32]) {
    for sample in signal {
        let y = (f64::from(sample.abs()) + f64::from(1e-6f32)) as f32;
        let y = crate::celt::celt_log2(y) * 0.6931471805599453f32;
        *sample *= cos_norm2(((0.5 * core::f64::consts::PI) * f64::from(y) - 1.0) as f32);
    }
}
impl BbweLayers {
    fn feature_net(&self, state: &mut BweState, features: &[f32], frames: usize) -> [f32; 512] {
        let mut conv1 = [0.0; 256];
        let mut conv2 = [0.0; 256];
        let mut upsampled = [0.0; 512];
        let mut output = [0.0; 512];
        for k in 0..frames {
            compute_generic_conv1d(
                &self.bbwenet_fnet_conv1.view(),
                &mut conv1[k * 128..(k + 1) * 128],
                &mut state.conv1,
                &features[k * 114..(k + 1) * 114],
                114,
                ACTIVATION_TANH,
                0,
            );
        }
        for k in 0..frames {
            compute_generic_conv1d(
                &self.bbwenet_fnet_conv2.view(),
                &mut conv2[k * 128..(k + 1) * 128],
                &mut state.conv2,
                &conv1[k * 128..(k + 1) * 128],
                128,
                ACTIVATION_TANH,
                0,
            );
        }
        for k in 0..frames {
            compute_generic_dense(
                &self.bbwenet_fnet_tconv.view(),
                &mut upsampled[k * 256..(k + 1) * 256],
                &conv2[k * 128..(k + 1) * 128],
                ACTIVATION_TANH,
                0,
            );
        }
        for k in 0..frames * 2 {
            compute_generic_gru(
                &self.bbwenet_fnet_gru_input.view(),
                &self.bbwenet_fnet_gru_recurrent.view(),
                &mut state.gru,
                &upsampled[k * 128..(k + 1) * 128],
                0,
            );
            output[k * 128..(k + 1) * 128].copy_from_slice(&state.gru);
        }
        output
    }
    pub(crate) fn process(
        &self,
        state: &mut BweState,
        output: &mut [f32],
        input: &[f32],
        features: &[f32],
    ) {
        assert!(input.len() == 160 || input.len() == 320);
        let subframes = input.len() / 80;
        let condition = self.feature_net(state, features, subframes / 2);
        let mut signal = [0.0; 2880];
        let mut next = [0.0; 2880];
        let mut windows = [[0.0; 120]; 3];
        for stage in 0..3 {
            compute_overlap_window(&mut windows[stage], 40 * (stage + 1));
        }
        for k in 0..subframes {
            adaconv_process_frame(
                &mut state.conv[0],
                &mut signal[k * 240..(k + 1) * 240],
                &input[k * 80..(k + 1) * 80],
                &condition[k * 128..(k + 1) * 128],
                &self.bbwenet_af1_kernel.view(),
                &self.bbwenet_af1_gain.view(),
                128,
                80,
                40,
                1,
                3,
                BBWENET_AF1_KERNEL_SIZE,
                BBWENET_AF1_LEFT_PADDING,
                BBWENET_AF1_FILTER_GAIN_A,
                BBWENET_AF1_FILTER_GAIN_B,
                1.0,
                &windows[0],
            );
        }
        for stage in 0..2 {
            let old_frame = 80 * (stage + 1);
            let frame = 80 * (stage + 2);
            let shape_f = if stage == 0 {
                &self.bbwenet_tdshape1_alpha1_f
            } else {
                &self.bbwenet_tdshape2_alpha1_f
            };
            let shape_t = if stage == 0 {
                &self.bbwenet_tdshape1_alpha1_t
            } else {
                &self.bbwenet_tdshape2_alpha1_t
            };
            let shape_2 = if stage == 0 {
                &self.bbwenet_tdshape1_alpha2
            } else {
                &self.bbwenet_tdshape2_alpha2
            };
            for k in 0..subframes {
                for channel in 0..3 {
                    let old =
                        &signal[(k * 3 + channel) * old_frame..(k * 3 + channel + 1) * old_frame];
                    let dst = &mut next[(k * 3 + channel) * frame..(k * 3 + channel + 1) * frame];
                    if stage == 0 {
                        state.resampler[channel].upsample(dst, old);
                    } else {
                        state.resampler[channel].interpolate(dst, old);
                    }
                }
                let start = (k * 3 + 1) * frame;
                let mut old = [0.0; 240];
                old[..frame].copy_from_slice(&next[start..start + frame]);
                adashape_process_frame(
                    &mut state.shape[stage],
                    &mut next[start..start + frame],
                    &old[..frame],
                    &condition[k * 128..(k + 1) * 128],
                    &shape_f.view(),
                    &shape_t.view(),
                    &shape_2.view(),
                    128,
                    frame,
                    4 * (stage + 2),
                    2,
                );
                valin_activation(&mut next[start + frame..start + 2 * frame]);
            }
            for k in 0..subframes {
                let channels = if stage == 0 { 3 } else { 1 };
                let kernel = if stage == 0 {
                    &self.bbwenet_af2_kernel
                } else {
                    &self.bbwenet_af3_kernel
                };
                let gain = if stage == 0 {
                    &self.bbwenet_af2_gain
                } else {
                    &self.bbwenet_af3_gain
                };
                let size = if stage == 0 {
                    BBWENET_AF2_KERNEL_SIZE
                } else {
                    BBWENET_AF3_KERNEL_SIZE
                };
                adaconv_process_frame(
                    &mut state.conv[stage + 1],
                    &mut signal[k * frame * channels..(k + 1) * frame * channels],
                    &next[k * frame * 3..(k + 1) * frame * 3],
                    &condition[k * 128..(k + 1) * 128],
                    &kernel.view(),
                    &gain.view(),
                    128,
                    frame,
                    frame / 2,
                    3,
                    channels,
                    size,
                    size - 1,
                    BBWENET_AF2_FILTER_GAIN_A,
                    BBWENET_AF2_FILTER_GAIN_B,
                    1.0,
                    &windows[stage + 1],
                );
            }
        }
        output[..input.len() * 3].copy_from_slice(&signal[..input.len() * 3]);
    }
}
