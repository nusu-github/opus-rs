#![cfg(feature = "deep_plc")]

use crate::dnn_weights::{WeightArray, WeightBlob, WeightError};
use crate::nnet::OwnedLinearLayer;
use alloc::borrow::Cow;
use alloc::vec::Vec;

fn find_array<'a>(
    blob: &'a WeightBlob<'a>,
    name: &'static str,
) -> Result<&'a WeightArray<'a>, WeightError> {
    blob.find(name).ok_or(WeightError::MissingArray(name))
}

#[allow(dead_code)]
fn array_len(array: &WeightArray<'_>, elem_size: usize) -> Result<usize, WeightError> {
    if array.size == 0 || array.size % elem_size != 0 {
        return Err(WeightError::InvalidBlob);
    }
    Ok(array.size / elem_size)
}

fn decode_f32(data: &[u8]) -> Result<Cow<'static, [f32]>, WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| WeightError::InvalidBlob)?;
        values.push(f32::from_le_bytes(bytes));
    }
    Ok(Cow::Owned(values))
}

fn decode_i8(data: &[u8]) -> Result<Cow<'static, [i8]>, WeightError> {
    let mut values = Vec::with_capacity(data.len());
    for &byte in data {
        values.push(byte as i8);
    }
    Ok(Cow::Owned(values))
}

fn decode_i32(data: &[u8]) -> Result<Cow<'static, [i32]>, WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| WeightError::InvalidBlob)?;
        values.push(i32::from_le_bytes(bytes));
    }
    Ok(Cow::Owned(values))
}

fn load_optional_f32(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<Cow<'static, [f32]>>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(decode_f32(array.data)?))
}

fn load_optional_i8(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<Cow<'static, [i8]>>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(decode_i8(array.data)?))
}

fn load_optional_i32(
    blob: &WeightBlob<'_>,
    name: Option<&'static str>,
) -> Result<Option<Cow<'static, [i32]>>, WeightError> {
    let Some(name) = name else {
        return Ok(None);
    };
    let array = find_array(blob, name)?;
    Ok(Some(decode_i32(array.data)?))
}

fn len_optional(array: Option<&[f32]>) -> Option<usize> {
    array.map(<[f32]>::len)
}

pub(crate) fn linear_layer_from_blob(
    blob: &WeightBlob<'_>,
    bias_name: Option<&'static str>,
    subias_name: Option<&'static str>,
    weights_name: Option<&'static str>,
    float_weights_name: Option<&'static str>,
    weights_idx_name: Option<&'static str>,
    diag_name: Option<&'static str>,
    scale_name: Option<&'static str>,
    expected_inputs: Option<usize>,
    expected_outputs: Option<usize>,
) -> Result<OwnedLinearLayer, WeightError> {
    let bias = load_optional_f32(blob, bias_name)?;
    let subias = load_optional_f32(blob, subias_name)?;
    // Quantized model exports omit the diagnostic float copy. C linear_init
    // accepts that omission, but still validates a copy when one is present.
    let float_weights = float_weights_name
        .and_then(|name| blob.find(name))
        .map(|array| decode_f32(array.data))
        .transpose()?;
    let weights = load_optional_i8(blob, weights_name)?;
    let weights_idx = load_optional_i32(blob, weights_idx_name)?;
    let diag = load_optional_f32(blob, diag_name)?;
    let scale = load_optional_f32(blob, scale_name)?;

    let weight_len = if let Some(weights) = &float_weights {
        weights.len()
    } else if let Some(weights) = &weights {
        weights.len()
    } else {
        return Err(WeightError::InvalidBlob);
    };

    let mut nb_outputs = expected_outputs
        .or_else(|| len_optional(bias.as_deref()))
        .or_else(|| len_optional(subias.as_deref()))
        .or_else(|| len_optional(scale.as_deref()));

    if nb_outputs.is_none() {
        if let Some(inputs) = expected_inputs {
            if inputs == 0 || weight_len % inputs != 0 {
                return Err(WeightError::InvalidBlob);
            }
            nb_outputs = Some(weight_len / inputs);
        }
    }

    let Some(nb_outputs) = nb_outputs else {
        return Err(WeightError::InvalidBlob);
    };
    if nb_outputs == 0 {
        return Err(WeightError::InvalidBlob);
    }

    let nb_inputs = if let Some(inputs) = expected_inputs {
        inputs
    } else {
        if weight_len % nb_outputs != 0 {
            return Err(WeightError::InvalidBlob);
        }
        weight_len / nb_outputs
    };

    if nb_inputs == 0 || nb_inputs.checked_mul(nb_outputs) != Some(weight_len) {
        return Err(WeightError::InvalidBlob);
    }

    if weights
        .as_ref()
        .is_some_and(|values| values.len() != weight_len)
    {
        return Err(WeightError::InvalidBlob);
    }
    if weights.is_some() && scale.is_none() {
        return Err(WeightError::InvalidBlob);
    }

    if let Some(bias) = &bias {
        if bias.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(subias) = &subias {
        if subias.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(scale) = &scale {
        if scale.len() != nb_outputs {
            return Err(WeightError::InvalidBlob);
        }
    }
    if let Some(diag) = &diag {
        if diag.len() % 3 != 0 {
            return Err(WeightError::InvalidBlob);
        }
    }

    Ok(OwnedLinearLayer {
        bias,
        subias,
        weights,
        float_weights,
        weights_idx,
        diag,
        scale,
        nb_inputs,
        nb_outputs,
    })
}

#[allow(dead_code)]
pub(crate) fn array_f32_len(
    blob: &WeightBlob<'_>,
    name: &'static str,
) -> Result<usize, WeightError> {
    let array = find_array(blob, name)?;
    array_len(array, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnet::{ACTIVATION_LINEAR, compute_generic_dense};

    fn add_float_array(blob: &mut Vec<u8>, name: &str, values: &[f32]) {
        let size = values.len() * 4;
        let mut header = [0u8; 64];
        header[12..16].copy_from_slice(&(size as i32).to_le_bytes());
        header[16..20].copy_from_slice(&(size as i32).to_le_bytes());
        header[20..20 + name.len()].copy_from_slice(name.as_bytes());
        blob.extend_from_slice(&header);
        for value in values {
            blob.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn quantized_layer_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        add_float_array(
            &mut bytes,
            "bias",
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        );
        add_float_array(&mut bytes, "scale", &[1.0 / 256.0; 8]);
        let mut header = [0u8; 64];
        header[8..12].copy_from_slice(&3i32.to_le_bytes());
        header[12..16].copy_from_slice(&32i32.to_le_bytes());
        header[16..20].copy_from_slice(&32i32.to_le_bytes());
        header[20..27].copy_from_slice(b"weights");
        bytes.extend_from_slice(&header);
        for _ in 0..8 {
            bytes.extend_from_slice(&[2, (-1i8) as u8, 1, 3]);
        }
        bytes
    }

    fn load_quantized_layer(bytes: &[u8]) -> Result<OwnedLinearLayer, WeightError> {
        linear_layer_from_blob(
            &WeightBlob::parse(bytes)?,
            Some("bias"),
            None,
            Some("weights"),
            Some("weights_float"),
            None,
            None,
            Some("scale"),
            Some(4),
            Some(8),
        )
    }

    #[test]
    fn quantized_model_accepts_an_omitted_diagnostic_float_copy() {
        let mut bytes = quantized_layer_bytes();
        let layer = load_quantized_layer(&bytes).unwrap();
        assert!(layer.float_weights.is_none());
        let mut output = [0.0; 8];
        compute_generic_dense(
            &layer.view(),
            &mut output,
            &[1.0, -1.0, 0.5, -0.5],
            ACTIVATION_LINEAR,
            0,
        );
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // A supplied diagnostic copy still takes precedence, matching C.
        add_float_array(&mut bytes, "weights_float", &[0.0; 32]);
        let diagnostic = load_quantized_layer(&bytes).unwrap();
        compute_generic_dense(
            &diagnostic.view(),
            &mut output,
            &[1.0, -1.0, 0.5, -0.5],
            ACTIVATION_LINEAR,
            0,
        );
        assert_eq!(output, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn quantized_model_rejects_a_malformed_diagnostic_float_copy() {
        let mut bytes = quantized_layer_bytes();
        add_float_array(&mut bytes, "weights_float", &[0.0; 31]);
        assert!(matches!(
            load_quantized_layer(&bytes),
            Err(WeightError::InvalidBlob)
        ));
    }

    #[cfg(feature = "deep_plc_weights")]
    #[test]
    fn bundled_deep_plc_submodels_load_independently() {
        let data = mousiki_deep_plc_weights::DNN_BLOB;
        crate::plc_model::PlcModel::from_weights(data).unwrap();
        crate::pitchdnn::PitchDnn::from_weights(data).unwrap();
        crate::fargan::FarganState::new().load_model(data).unwrap();
    }

    #[test]
    fn runtime_layer_owns_weights_after_blob_and_original_model_drop() {
        let cloned = {
            let mut bytes = Vec::new();
            add_float_array(&mut bytes, "bias", &[1.0, -1.0]);
            add_float_array(&mut bytes, "weights", &[2.0, 3.0, 4.0, 5.0]);
            let blob = WeightBlob::parse(&bytes).unwrap();
            let layer = linear_layer_from_blob(
                &blob,
                Some("bias"),
                None,
                None,
                Some("weights"),
                None,
                None,
                None,
                Some(2),
                Some(2),
            )
            .unwrap();
            assert!(matches!(layer.bias, Some(Cow::Owned(_))));
            assert!(matches!(layer.float_weights, Some(Cow::Owned(_))));
            layer.clone()
        };
        let mut output = [0.0; 2];
        compute_generic_dense(
            &cloned.view(),
            &mut output,
            &[2.0, 3.0],
            ACTIVATION_LINEAR,
            0,
        );
        assert_eq!(output, [17.0, 20.0]);
    }
}
