//! Owned neural model parameters. Borrowed layer views avoid leaked allocations.
use crate::dnn_weights::{WeightBlob, WeightError};
use crate::nnet::LinearLayer;
use alloc::vec::Vec;

#[path = "model_specs.rs"]
mod specs;
pub(crate) use specs::*;

struct LayerSpec {
    names: [Option<&'static str>; 7],
    inputs: usize,
    outputs: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedLayer {
    bias: Option<Vec<f32>>,
    subias: Option<Vec<f32>>,
    weights: Option<Vec<i8>>,
    float_weights: Option<Vec<f32>>,
    indices: Option<Vec<i32>>,
    diag: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
    inputs: usize,
    outputs: usize,
}

impl OwnedLayer {
    fn load(blob: &WeightBlob<'_>, spec: LayerSpec) -> Result<Self, WeightError> {
        fn bytes<'a>(
            blob: &WeightBlob<'a>,
            name: Option<&'static str>,
            count: usize,
            optional: bool,
        ) -> Result<Option<&'a [u8]>, WeightError> {
            let Some(name) = name else {
                return Ok(None);
            };
            let Some(array) = blob.find(name) else {
                return if optional {
                    Ok(None)
                } else {
                    Err(WeightError::MissingArray(name))
                };
            };
            if array.data.len() != count {
                return Err(WeightError::SizeMismatch(name));
            }
            Ok(Some(array.data))
        }
        fn floats(data: Option<&[u8]>) -> Option<Vec<f32>> {
            data.map(|data| {
                data.chunks_exact(4)
                    .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                    .collect()
            })
        }
        let [bias, subias, weights, float_weights, indices, diag, scale] = spec.names;
        let count = spec
            .inputs
            .checked_mul(spec.outputs)
            .ok_or(WeightError::InvalidBlob)?;
        if spec.inputs == 0 || spec.inputs > 2048 || spec.outputs == 0 || indices.is_some() {
            // The three canonical OSCE models contain dense layers only.
            return Err(WeightError::InvalidBlob);
        }
        let result = Self {
            bias: floats(bytes(blob, bias, spec.outputs * 4, false)?),
            subias: floats(bytes(blob, subias, spec.outputs * 4, false)?),
            weights: bytes(blob, weights, count, false)?
                .map(|v| v.iter().map(|&x| x as i8).collect()),
            float_weights: floats(bytes(blob, float_weights, count * 4, true)?),
            indices: None,
            diag: floats(bytes(blob, diag, spec.outputs * 4, false)?),
            scale: floats(bytes(blob, scale, spec.outputs * 4, false)?),
            inputs: spec.inputs,
            outputs: spec.outputs,
        };
        if result.weights.is_none() && result.float_weights.is_none() {
            return Err(WeightError::InvalidBlob);
        }
        Ok(result)
    }

    pub fn view(&self) -> LinearLayer<'_> {
        LinearLayer {
            bias: self.bias.as_deref(),
            subias: self.subias.as_deref(),
            weights: self.weights.as_deref(),
            float_weights: self.float_weights.as_deref(),
            weights_idx: self.indices.as_deref(),
            diag: self.diag.as_deref(),
            scale: self.scale.as_deref(),
            nb_inputs: self.inputs,
            nb_outputs: self.outputs,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OsceModel {
    pub(crate) lace: LaceLayers,
    pub(crate) nolace: NoLaceLayers,
    pub(crate) bbwe: BbweLayers,
}

impl OsceModel {
    /// Load all enhancement networks from an Opus weight blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, WeightError> {
        let blob = WeightBlob::parse(data)?;
        Ok(Self {
            lace: LaceLayers::load(&blob)?,
            nolace: NoLaceLayers::load(&blob)?,
            bbwe: BbweLayers::load(&blob)?,
        })
    }
}
