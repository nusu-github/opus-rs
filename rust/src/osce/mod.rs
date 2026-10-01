//! Safe Opus Speech Coding Enhancement (LACE, NoLACE, and bandwidth extension).
mod bwe;
mod enhance;
mod features;
mod models;
pub(crate) mod nndsp;
mod tables;
pub use crate::dnn_weights::WeightError;
pub use models::OsceModel;

use crate::silk::decoder_control::DecoderControl;
use crate::silk::decoder_state::DecoderState;
use features::{BweFeatures, OsceFeatures};

#[derive(Clone, Debug)]
pub(crate) struct OsceState {
    pub(crate) method: i32,
    features: OsceFeatures,
    network: enhance::EnhancementState,
}
impl Default for OsceState {
    fn default() -> Self {
        let mut state = Self {
            method: 0,
            features: OsceFeatures::default(),
            network: enhance::EnhancementState::default(),
        };
        state.features.reset = 2;
        state
    }
}
impl OsceState {
    pub(crate) fn reset(&mut self, method: i32) {
        *self = Self::default();
        self.method = method;
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct OsceBweState {
    features: BweFeatures,
    network: bwe::BweState,
}

pub(crate) fn enhance_frame(
    state: &mut DecoderState,
    control: &DecoderControl,
    pcm: &mut [i16],
    num_bits: i32,
) {
    if state.sample_rate.fs_khz != 16 || state.sample_rate.nb_subfr != 4 {
        state.osce.reset(state.osce.method);
        return;
    }
    let mut osce = core::mem::take(&mut state.osce);
    let (features, numbits, periods) =
        features::calculate(&mut osce.features, state, control, pcm, num_bits);
    let mut input = [0.0; 320];
    for k in 0..320 {
        input[k] = f32::from(pcm[k]) * (1.0 / 32768.0);
    }
    let mut output = input;
    if let Some(model) = &state.osce_model {
        match osce.method {
            1 => model.lace.process(
                &mut osce.network,
                &mut output,
                &input,
                &features,
                &numbits,
                &periods,
            ),
            2 => model.nolace.process(
                &mut osce.network,
                &mut output,
                &input,
                &features,
                &numbits,
                &periods,
            ),
            _ => {}
        }
    }
    if osce.features.reset > 1 {
        output = input;
        osce.features.reset -= 1;
    } else if osce.features.reset != 0 {
        features::crossfade(&mut output, &input);
        osce.features.reset = 0;
    }
    for k in 0..320 {
        pcm[k] = libm::rintf((32768.0 * output[k]).clamp(-32767.0, 32767.0)) as i16;
    }
    state.osce = osce;
}

pub(crate) fn extend_bandwidth(
    model: &OsceModel,
    state: &mut OsceBweState,
    output: &mut [i16],
    input: &[i16],
) {
    let features = features::calculate_bwe(&mut state.features, input);
    let mut scaled = [0.0; 320];
    for (dst, &sample) in scaled.iter_mut().zip(input) {
        *dst = f32::from(sample) * (1.0 / 32768.0);
    }
    let mut generated = [0.0; 960];
    model.bbwe.process(
        &mut state.network,
        &mut generated,
        &scaled[..input.len()],
        &features,
    );
    let count = input.len() * 3;
    output[..21].copy_from_slice(&state.network.delay);
    for k in 0..count {
        let value = libm::rintf((32768.0 * generated[k]).clamp(-32767.0, 32767.0)) as i16;
        if k < count - 21 {
            output[k + 21] = value;
        } else {
            state.network.delay[k - (count - 21)] = value;
        }
    }
}

pub(crate) fn crossfade_bwe(output: &mut [i16], input: &[i16]) {
    features::crossfade_bwe(output, input);
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec::Vec;
    #[test]
    #[ignore = "requires the pinned OSCE weight blob; run tools/reference/check_osce.sh"]
    fn networks_match_pinned_scalar_c() {
        let path = std::env::var("OSCE_WEIGHTS_PATH")
            .expect("OSCE_WEIGHTS_PATH must point to the exported Opus model blob");
        let data = std::fs::read(path).unwrap();
        let model = OsceModel::from_bytes(&data).unwrap();
        let mut lace = enhance::EnhancementState::default();
        let mut nolace = enhance::EnhancementState::default();
        let mut bwe = bwe::BweState::default();
        let mut vectors = {
            #[cfg(feature = "dnn_debug_float")]
            const VECTORS: &str = include_str!("../../tests/fixtures/reference/osce-networks.txt");
            #[cfg(not(feature = "dnn_debug_float"))]
            const VECTORS: &str =
                include_str!("../../tests/fixtures/reference/osce-networks-quantized.txt");
            VECTORS
        }
        .lines();
        for frame in 0..5usize {
            let input = core::array::from_fn(|k| {
                (((k * 37 + frame * 101) % 1024) as i32 - 512) as f32 * (1.0 / 2048.0)
            });
            let features = core::array::from_fn(|k| {
                (((k * 13 + frame * 17) % 127) as i32 - 63) as f32 * (1.0 / 64.0)
            });
            let bwe_features: [f32; 228] = core::array::from_fn(|k| {
                (((k * 19 + frame * 23) % 127) as i32 - 63) as f32 * (1.0 / 64.0)
            });
            let periods = core::array::from_fn(|k| 40 + (k * 31 + frame * 7) % 160);
            let bits = [(20 + frame * 200) as f32, (7 + frame * 31) as f32];
            let mut output = [0.0; 320];
            model
                .lace
                .process(&mut lace, &mut output, &input, &features, &bits, &periods);
            check(vectors.next().unwrap(), &output);
            model
                .nolace
                .process(&mut nolace, &mut output, &input, &features, &bits, &periods);
            check(vectors.next().unwrap(), &output);
            let mut extended = [0.0; 960];
            model
                .bbwe
                .process(&mut bwe, &mut extended, &input, &bwe_features);
            check(vectors.next().unwrap(), &extended);
        }
        assert!(vectors.next().is_none());
    }
    fn check(line: &str, output: &[f32]) {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len() - 2, output.len());
        for (k, &value) in output.iter().enumerate() {
            let expected = u32::from_str_radix(fields[k + 2], 16).unwrap();
            assert_eq!(
                value.to_bits(),
                expected,
                "{} frame {} sample {} (Rust {}, C {})",
                fields[0],
                fields[1],
                k,
                value,
                f32::from_bits(expected)
            );
        }
    }
}
