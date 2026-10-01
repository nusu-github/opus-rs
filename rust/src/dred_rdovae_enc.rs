//! RDOVAE encoder model and inference helpers.

use crate::dnn_weights::{WeightBlob, WeightError, optional_bytes, require_bytes};
use crate::dred_constants::{
    DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_PADDED_LATENT_DIM, DRED_PADDED_STATE_DIM,
    DRED_STATE_DIM,
};
use crate::dred_rdovae_enc_data::*;
use crate::nnet::{
    ACTIVATION_LINEAR, ACTIVATION_TANH, LinearLayer, OwnedLinearLayer, compute_generic_conv1d,
    compute_generic_conv1d_dilation, compute_generic_dense, compute_generic_gru,
};
use alloc::borrow::Cow;
use alloc::vec::Vec;

pub(crate) const ENC_DENSE1_OUT_SIZE: usize = 64;
pub(crate) const ENC_ZDENSE_OUT_SIZE: usize = 32;
pub(crate) const GDENSE1_OUT_SIZE: usize = 128;
pub(crate) const GDENSE2_OUT_SIZE: usize = 56;
pub(crate) const ENC_GRU1_OUT_SIZE: usize = 32;
pub(crate) const ENC_GRU1_STATE_SIZE: usize = 32;
pub(crate) const ENC_GRU1_IN_SIZE: usize = 64;
pub(crate) const ENC_GRU2_OUT_SIZE: usize = 32;
pub(crate) const ENC_GRU2_STATE_SIZE: usize = 32;
pub(crate) const ENC_GRU2_IN_SIZE: usize = 160;
pub(crate) const ENC_GRU3_OUT_SIZE: usize = 32;
pub(crate) const ENC_GRU3_STATE_SIZE: usize = 32;
pub(crate) const ENC_GRU3_IN_SIZE: usize = 256;
pub(crate) const ENC_GRU4_OUT_SIZE: usize = 32;
pub(crate) const ENC_GRU4_STATE_SIZE: usize = 32;
pub(crate) const ENC_GRU4_IN_SIZE: usize = 352;
pub(crate) const ENC_GRU5_OUT_SIZE: usize = 32;
pub(crate) const ENC_GRU5_STATE_SIZE: usize = 32;
pub(crate) const ENC_GRU5_IN_SIZE: usize = 448;
pub(crate) const ENC_CONV1_OUT_SIZE: usize = 64;
pub(crate) const ENC_CONV1_IN_SIZE: usize = 64;
pub(crate) const ENC_CONV1_STATE_SIZE: usize = (64 * (1));
pub(crate) const ENC_CONV2_OUT_SIZE: usize = 64;
pub(crate) const ENC_CONV2_IN_SIZE: usize = 64;
pub(crate) const ENC_CONV2_STATE_SIZE: usize = (64 * (1));
pub(crate) const ENC_CONV3_OUT_SIZE: usize = 64;
pub(crate) const ENC_CONV3_IN_SIZE: usize = 64;
pub(crate) const ENC_CONV3_STATE_SIZE: usize = (64 * (1));
pub(crate) const ENC_CONV4_OUT_SIZE: usize = 64;
pub(crate) const ENC_CONV4_IN_SIZE: usize = 64;
pub(crate) const ENC_CONV4_STATE_SIZE: usize = (64 * (1));
pub(crate) const ENC_CONV5_OUT_SIZE: usize = 64;
pub(crate) const ENC_CONV5_IN_SIZE: usize = 64;
pub(crate) const ENC_CONV5_STATE_SIZE: usize = (64 * (1));

const ENC_BUFFER_SIZE: usize = ENC_DENSE1_OUT_SIZE
    + ENC_GRU1_OUT_SIZE
    + ENC_GRU2_OUT_SIZE
    + ENC_GRU3_OUT_SIZE
    + ENC_GRU4_OUT_SIZE
    + ENC_GRU5_OUT_SIZE
    + ENC_CONV1_OUT_SIZE
    + ENC_CONV2_OUT_SIZE
    + ENC_CONV3_OUT_SIZE
    + ENC_CONV4_OUT_SIZE
    + ENC_CONV5_OUT_SIZE;

#[derive(Clone, Debug, Default)]
pub(crate) struct RdovaeEnc {
    pub enc_conv_dense5: OwnedLinearLayer,
    pub enc_conv_dense4: OwnedLinearLayer,
    pub enc_conv_dense3: OwnedLinearLayer,
    pub enc_conv_dense2: OwnedLinearLayer,
    pub enc_conv_dense1: OwnedLinearLayer,
    pub enc_dense1: OwnedLinearLayer,
    pub enc_zdense: OwnedLinearLayer,
    pub gdense1: OwnedLinearLayer,
    pub gdense2: OwnedLinearLayer,
    pub enc_gru1_input: OwnedLinearLayer,
    pub enc_gru1_recurrent: OwnedLinearLayer,
    pub enc_gru2_input: OwnedLinearLayer,
    pub enc_gru2_recurrent: OwnedLinearLayer,
    pub enc_gru3_input: OwnedLinearLayer,
    pub enc_gru3_recurrent: OwnedLinearLayer,
    pub enc_gru4_input: OwnedLinearLayer,
    pub enc_gru4_recurrent: OwnedLinearLayer,
    pub enc_gru5_input: OwnedLinearLayer,
    pub enc_gru5_recurrent: OwnedLinearLayer,
    pub enc_conv1: OwnedLinearLayer,
    pub enc_conv2: OwnedLinearLayer,
    pub enc_conv3: OwnedLinearLayer,
    pub enc_conv4: OwnedLinearLayer,
    pub enc_conv5: OwnedLinearLayer,
}

impl RdovaeEnc {
    pub(crate) fn new() -> Self {
        let mut model = Self::default();
        init_rdovaeenc(&mut model);
        model
    }

    pub(crate) fn from_weights(data: &[u8]) -> Result<Self, WeightError> {
        let blob = WeightBlob::parse(data)?;
        let mut model = Self::default();
        init_rdovaeenc_from_weights(&mut model, &blob)?;
        Ok(model)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RdovaeEncState {
    pub initialized: i32,
    pub gru1_state: [f32; ENC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; ENC_GRU2_STATE_SIZE],
    pub gru3_state: [f32; ENC_GRU3_STATE_SIZE],
    pub gru4_state: [f32; ENC_GRU4_STATE_SIZE],
    pub gru5_state: [f32; ENC_GRU5_STATE_SIZE],
    pub conv1_state: [f32; ENC_CONV1_STATE_SIZE],
    pub conv2_state: [f32; 2 * ENC_CONV2_STATE_SIZE],
    pub conv3_state: [f32; 2 * ENC_CONV3_STATE_SIZE],
    pub conv4_state: [f32; 2 * ENC_CONV4_STATE_SIZE],
    pub conv5_state: [f32; 2 * ENC_CONV5_STATE_SIZE],
}

impl Default for RdovaeEncState {
    fn default() -> Self {
        Self {
            initialized: 0,
            gru1_state: [0.0; ENC_GRU1_STATE_SIZE],
            gru2_state: [0.0; ENC_GRU2_STATE_SIZE],
            gru3_state: [0.0; ENC_GRU3_STATE_SIZE],
            gru4_state: [0.0; ENC_GRU4_STATE_SIZE],
            gru5_state: [0.0; ENC_GRU5_STATE_SIZE],
            conv1_state: [0.0; ENC_CONV1_STATE_SIZE],
            conv2_state: [0.0; 2 * ENC_CONV2_STATE_SIZE],
            conv3_state: [0.0; 2 * ENC_CONV3_STATE_SIZE],
            conv4_state: [0.0; 2 * ENC_CONV4_STATE_SIZE],
            conv5_state: [0.0; 2 * ENC_CONV5_STATE_SIZE],
        }
    }
}

fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, initialized: &mut i32) {
    if *initialized == 0 {
        for i in 0..dilation {
            let start = i * len;
            let end = start + len;
            mem[start..end].fill(0.0);
        }
    }
}

pub(crate) fn dred_rdovae_encode_dframe(
    enc_state: &mut RdovaeEncState,
    model: &RdovaeEnc,
    latents: &mut [f32],
    initial_state: &mut [f32],
    input: &[f32],
    arch: i32,
) {
    debug_assert!(latents.len() >= DRED_LATENT_DIM);
    debug_assert!(initial_state.len() >= DRED_STATE_DIM);
    debug_assert!(input.len() >= 2 * DRED_NUM_FEATURES);

    let mut padded_latents = [0.0f32; DRED_PADDED_LATENT_DIM];
    let mut padded_state = [0.0f32; DRED_PADDED_STATE_DIM];
    let mut buffer = [0.0f32; ENC_BUFFER_SIZE];
    let mut state_hidden = [0.0f32; GDENSE1_OUT_SIZE];
    let mut output_index = 0usize;

    compute_generic_dense(
        &model.enc_dense1.view(),
        &mut buffer[output_index..output_index + ENC_DENSE1_OUT_SIZE],
        &input[..2 * DRED_NUM_FEATURES],
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_DENSE1_OUT_SIZE;

    compute_generic_gru(
        &model.enc_gru1_input.view(),
        &model.enc_gru1_recurrent.view(),
        &mut enc_state.gru1_state,
        &buffer[..output_index],
        arch,
    );
    buffer[output_index..output_index + ENC_GRU1_OUT_SIZE].copy_from_slice(&enc_state.gru1_state);
    output_index += ENC_GRU1_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv1_state,
        ENC_CONV1_IN_SIZE,
        1,
        &mut enc_state.initialized,
    );
    let mut conv_tmp = [0.0f32; 64];
    compute_generic_dense(
        &model.enc_conv_dense1.view(),
        &mut conv_tmp,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    let (_, buffer_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d(
        &model.enc_conv1.view(),
        &mut buffer_out[..ENC_CONV1_OUT_SIZE],
        &mut enc_state.conv1_state,
        &conv_tmp,
        ENC_CONV1_OUT_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_CONV1_OUT_SIZE;

    compute_generic_gru(
        &model.enc_gru2_input.view(),
        &model.enc_gru2_recurrent.view(),
        &mut enc_state.gru2_state,
        &buffer[..output_index],
        arch,
    );
    buffer[output_index..output_index + ENC_GRU2_OUT_SIZE].copy_from_slice(&enc_state.gru2_state);
    output_index += ENC_GRU2_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv2_state,
        ENC_CONV2_IN_SIZE,
        2,
        &mut enc_state.initialized,
    );
    let mut conv_tmp = [0.0f32; 64];
    compute_generic_dense(
        &model.enc_conv_dense2.view(),
        &mut conv_tmp,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    let (_, buffer_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d_dilation(
        &model.enc_conv2.view(),
        &mut buffer_out[..ENC_CONV2_OUT_SIZE],
        &mut enc_state.conv2_state,
        &conv_tmp,
        ENC_CONV2_OUT_SIZE,
        2,
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_CONV2_OUT_SIZE;

    compute_generic_gru(
        &model.enc_gru3_input.view(),
        &model.enc_gru3_recurrent.view(),
        &mut enc_state.gru3_state,
        &buffer[..output_index],
        arch,
    );
    buffer[output_index..output_index + ENC_GRU3_OUT_SIZE].copy_from_slice(&enc_state.gru3_state);
    output_index += ENC_GRU3_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv3_state,
        ENC_CONV3_IN_SIZE,
        2,
        &mut enc_state.initialized,
    );
    let mut conv_tmp = [0.0f32; 64];
    compute_generic_dense(
        &model.enc_conv_dense3.view(),
        &mut conv_tmp,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    let (_, buffer_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d_dilation(
        &model.enc_conv3.view(),
        &mut buffer_out[..ENC_CONV3_OUT_SIZE],
        &mut enc_state.conv3_state,
        &conv_tmp,
        ENC_CONV3_OUT_SIZE,
        2,
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_CONV3_OUT_SIZE;

    compute_generic_gru(
        &model.enc_gru4_input.view(),
        &model.enc_gru4_recurrent.view(),
        &mut enc_state.gru4_state,
        &buffer[..output_index],
        arch,
    );
    buffer[output_index..output_index + ENC_GRU4_OUT_SIZE].copy_from_slice(&enc_state.gru4_state);
    output_index += ENC_GRU4_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv4_state,
        ENC_CONV4_IN_SIZE,
        2,
        &mut enc_state.initialized,
    );
    let mut conv_tmp = [0.0f32; 64];
    compute_generic_dense(
        &model.enc_conv_dense4.view(),
        &mut conv_tmp,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    let (_, buffer_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d_dilation(
        &model.enc_conv4.view(),
        &mut buffer_out[..ENC_CONV4_OUT_SIZE],
        &mut enc_state.conv4_state,
        &conv_tmp,
        ENC_CONV4_OUT_SIZE,
        2,
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_CONV4_OUT_SIZE;

    compute_generic_gru(
        &model.enc_gru5_input.view(),
        &model.enc_gru5_recurrent.view(),
        &mut enc_state.gru5_state,
        &buffer[..output_index],
        arch,
    );
    buffer[output_index..output_index + ENC_GRU5_OUT_SIZE].copy_from_slice(&enc_state.gru5_state);
    output_index += ENC_GRU5_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv5_state,
        ENC_CONV5_IN_SIZE,
        2,
        &mut enc_state.initialized,
    );
    let mut conv_tmp = [0.0f32; 64];
    compute_generic_dense(
        &model.enc_conv_dense5.view(),
        &mut conv_tmp,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    let (_, buffer_out) = buffer.split_at_mut(output_index);
    compute_generic_conv1d_dilation(
        &model.enc_conv5.view(),
        &mut buffer_out[..ENC_CONV5_OUT_SIZE],
        &mut enc_state.conv5_state,
        &conv_tmp,
        ENC_CONV5_OUT_SIZE,
        2,
        ACTIVATION_TANH,
        arch,
    );
    output_index += ENC_CONV5_OUT_SIZE;

    compute_generic_dense(
        &model.enc_zdense.view(),
        &mut padded_latents,
        &buffer[..output_index],
        ACTIVATION_LINEAR,
        arch,
    );
    latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

    compute_generic_dense(
        &model.gdense1.view(),
        &mut state_hidden,
        &buffer[..output_index],
        ACTIVATION_TANH,
        arch,
    );
    compute_generic_dense(
        &model.gdense2.view(),
        &mut padded_state,
        &state_hidden,
        ACTIVATION_LINEAR,
        arch,
    );
    initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
    enc_state.initialized = 1;
}

fn sparse_block_count(idx: &[i32], nb_inputs: usize, nb_outputs: usize) -> usize {
    let mut remain = idx.len() as i32;
    let mut out = nb_outputs as i32;
    let mut total_blocks = 0i32;
    let mut pos = 0usize;
    while remain > 0 {
        let nb_blocks = idx[pos];
        pos += 1;
        remain -= 1;
        if nb_blocks < 0 || remain < nb_blocks {
            return 0;
        }
        for _ in 0..nb_blocks {
            let offset = idx[pos];
            pos += 1;
            remain -= 1;
            if offset < 0 || offset + 3 >= nb_inputs as i32 || (offset & 0x3) != 0 {
                return 0;
            }
        }
        out -= 8;
        total_blocks += nb_blocks;
    }
    if out != 0 {
        return 0;
    }
    total_blocks as usize
}

fn linear_layer(
    bias: Option<&'static [f32]>,
    subias: Option<&'static [f32]>,
    weights: Option<&'static [i8]>,
    float_weights: Option<&'static [f32]>,
    weights_idx: Option<&'static [i32]>,
    diag: Option<&'static [f32]>,
    scale: Option<&'static [f32]>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> OwnedLinearLayer {
    // The standard C build omits debug float copies of quantized weights.
    // Runtime weight blobs retain their explicitly supplied representation.
    #[cfg(not(feature = "dnn_debug_float"))]
    let float_weights = if weights.is_some() {
        None
    } else {
        float_weights
    };
    if let Some(bias) = bias {
        debug_assert_eq!(bias.len(), nb_outputs);
    }
    if let Some(subias) = subias {
        debug_assert_eq!(subias.len(), nb_outputs);
    }
    if let Some(scale) = scale {
        debug_assert_eq!(scale.len(), nb_outputs);
    }
    if let Some(float_weights) = float_weights {
        if let Some(weights_idx) = weights_idx {
            let blocks = sparse_block_count(weights_idx, nb_inputs, nb_outputs);
            debug_assert_eq!(float_weights.len(), blocks * 32);
        } else {
            debug_assert_eq!(float_weights.len(), nb_inputs * nb_outputs);
        }
    }
    if let Some(weights) = weights {
        if let Some(weights_idx) = weights_idx {
            let blocks = sparse_block_count(weights_idx, nb_inputs, nb_outputs);
            debug_assert_eq!(weights.len(), blocks * 32);
        } else {
            debug_assert_eq!(weights.len(), nb_inputs * nb_outputs);
        }
    }

    LinearLayer {
        bias,
        subias,
        weights,
        float_weights,
        weights_idx,
        diag,
        scale,
        nb_inputs,
        nb_outputs,
    }
    .into()
}

fn init_rdovaeenc(model: &mut RdovaeEnc) {
    model.enc_dense1 = linear_layer(
        Some(&ENC_DENSE1_BIAS),
        None,
        None,
        Some(&ENC_DENSE1_WEIGHTS_FLOAT),
        None,
        None,
        None,
        40,
        64,
    );
    model.enc_zdense = linear_layer(
        Some(&ENC_ZDENSE_BIAS),
        Some(&ENC_ZDENSE_SUBIAS),
        Some(&ENC_ZDENSE_WEIGHTS_INT8),
        Some(&ENC_ZDENSE_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_ZDENSE_SCALE),
        544,
        32,
    );
    model.gdense2 = linear_layer(
        Some(&GDENSE2_BIAS),
        Some(&GDENSE2_SUBIAS),
        Some(&GDENSE2_WEIGHTS_INT8),
        Some(&GDENSE2_WEIGHTS_FLOAT),
        None,
        None,
        Some(&GDENSE2_SCALE),
        128,
        56,
    );
    model.gdense1 = linear_layer(
        Some(&GDENSE1_BIAS),
        Some(&GDENSE1_SUBIAS),
        Some(&GDENSE1_WEIGHTS_INT8),
        Some(&GDENSE1_WEIGHTS_FLOAT),
        Some(&GDENSE1_WEIGHTS_IDX),
        None,
        Some(&GDENSE1_SCALE),
        544,
        128,
    );
    model.enc_conv_dense1 = linear_layer(
        Some(&ENC_CONV_DENSE1_BIAS),
        Some(&ENC_CONV_DENSE1_SUBIAS),
        Some(&ENC_CONV_DENSE1_WEIGHTS_INT8),
        Some(&ENC_CONV_DENSE1_WEIGHTS_FLOAT),
        Some(&ENC_CONV_DENSE1_WEIGHTS_IDX),
        None,
        Some(&ENC_CONV_DENSE1_SCALE),
        96,
        64,
    );
    model.enc_conv_dense2 = linear_layer(
        Some(&ENC_CONV_DENSE2_BIAS),
        Some(&ENC_CONV_DENSE2_SUBIAS),
        Some(&ENC_CONV_DENSE2_WEIGHTS_INT8),
        Some(&ENC_CONV_DENSE2_WEIGHTS_FLOAT),
        Some(&ENC_CONV_DENSE2_WEIGHTS_IDX),
        None,
        Some(&ENC_CONV_DENSE2_SCALE),
        192,
        64,
    );
    model.enc_conv_dense3 = linear_layer(
        Some(&ENC_CONV_DENSE3_BIAS),
        Some(&ENC_CONV_DENSE3_SUBIAS),
        Some(&ENC_CONV_DENSE3_WEIGHTS_INT8),
        Some(&ENC_CONV_DENSE3_WEIGHTS_FLOAT),
        Some(&ENC_CONV_DENSE3_WEIGHTS_IDX),
        None,
        Some(&ENC_CONV_DENSE3_SCALE),
        288,
        64,
    );
    model.enc_conv_dense4 = linear_layer(
        Some(&ENC_CONV_DENSE4_BIAS),
        Some(&ENC_CONV_DENSE4_SUBIAS),
        Some(&ENC_CONV_DENSE4_WEIGHTS_INT8),
        Some(&ENC_CONV_DENSE4_WEIGHTS_FLOAT),
        Some(&ENC_CONV_DENSE4_WEIGHTS_IDX),
        None,
        Some(&ENC_CONV_DENSE4_SCALE),
        384,
        64,
    );
    model.enc_conv_dense5 = linear_layer(
        Some(&ENC_CONV_DENSE5_BIAS),
        Some(&ENC_CONV_DENSE5_SUBIAS),
        Some(&ENC_CONV_DENSE5_WEIGHTS_INT8),
        Some(&ENC_CONV_DENSE5_WEIGHTS_FLOAT),
        Some(&ENC_CONV_DENSE5_WEIGHTS_IDX),
        None,
        Some(&ENC_CONV_DENSE5_SCALE),
        480,
        64,
    );
    model.enc_gru1_input = linear_layer(
        Some(&ENC_GRU1_INPUT_BIAS),
        Some(&ENC_GRU1_INPUT_SUBIAS),
        Some(&ENC_GRU1_INPUT_WEIGHTS_INT8),
        Some(&ENC_GRU1_INPUT_WEIGHTS_FLOAT),
        Some(&ENC_GRU1_INPUT_WEIGHTS_IDX),
        None,
        Some(&ENC_GRU1_INPUT_SCALE),
        64,
        96,
    );
    model.enc_gru1_recurrent = linear_layer(
        Some(&ENC_GRU1_RECURRENT_BIAS),
        Some(&ENC_GRU1_RECURRENT_SUBIAS),
        Some(&ENC_GRU1_RECURRENT_WEIGHTS_INT8),
        Some(&ENC_GRU1_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_GRU1_RECURRENT_SCALE),
        32,
        96,
    );
    model.enc_gru2_input = linear_layer(
        Some(&ENC_GRU2_INPUT_BIAS),
        Some(&ENC_GRU2_INPUT_SUBIAS),
        Some(&ENC_GRU2_INPUT_WEIGHTS_INT8),
        Some(&ENC_GRU2_INPUT_WEIGHTS_FLOAT),
        Some(&ENC_GRU2_INPUT_WEIGHTS_IDX),
        None,
        Some(&ENC_GRU2_INPUT_SCALE),
        160,
        96,
    );
    model.enc_gru2_recurrent = linear_layer(
        Some(&ENC_GRU2_RECURRENT_BIAS),
        Some(&ENC_GRU2_RECURRENT_SUBIAS),
        Some(&ENC_GRU2_RECURRENT_WEIGHTS_INT8),
        Some(&ENC_GRU2_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_GRU2_RECURRENT_SCALE),
        32,
        96,
    );
    model.enc_gru3_input = linear_layer(
        Some(&ENC_GRU3_INPUT_BIAS),
        Some(&ENC_GRU3_INPUT_SUBIAS),
        Some(&ENC_GRU3_INPUT_WEIGHTS_INT8),
        Some(&ENC_GRU3_INPUT_WEIGHTS_FLOAT),
        Some(&ENC_GRU3_INPUT_WEIGHTS_IDX),
        None,
        Some(&ENC_GRU3_INPUT_SCALE),
        256,
        96,
    );
    model.enc_gru3_recurrent = linear_layer(
        Some(&ENC_GRU3_RECURRENT_BIAS),
        Some(&ENC_GRU3_RECURRENT_SUBIAS),
        Some(&ENC_GRU3_RECURRENT_WEIGHTS_INT8),
        Some(&ENC_GRU3_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_GRU3_RECURRENT_SCALE),
        32,
        96,
    );
    model.enc_gru4_input = linear_layer(
        Some(&ENC_GRU4_INPUT_BIAS),
        Some(&ENC_GRU4_INPUT_SUBIAS),
        Some(&ENC_GRU4_INPUT_WEIGHTS_INT8),
        Some(&ENC_GRU4_INPUT_WEIGHTS_FLOAT),
        Some(&ENC_GRU4_INPUT_WEIGHTS_IDX),
        None,
        Some(&ENC_GRU4_INPUT_SCALE),
        352,
        96,
    );
    model.enc_gru4_recurrent = linear_layer(
        Some(&ENC_GRU4_RECURRENT_BIAS),
        Some(&ENC_GRU4_RECURRENT_SUBIAS),
        Some(&ENC_GRU4_RECURRENT_WEIGHTS_INT8),
        Some(&ENC_GRU4_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_GRU4_RECURRENT_SCALE),
        32,
        96,
    );
    model.enc_gru5_input = linear_layer(
        Some(&ENC_GRU5_INPUT_BIAS),
        Some(&ENC_GRU5_INPUT_SUBIAS),
        Some(&ENC_GRU5_INPUT_WEIGHTS_INT8),
        Some(&ENC_GRU5_INPUT_WEIGHTS_FLOAT),
        Some(&ENC_GRU5_INPUT_WEIGHTS_IDX),
        None,
        Some(&ENC_GRU5_INPUT_SCALE),
        448,
        96,
    );
    model.enc_gru5_recurrent = linear_layer(
        Some(&ENC_GRU5_RECURRENT_BIAS),
        Some(&ENC_GRU5_RECURRENT_SUBIAS),
        Some(&ENC_GRU5_RECURRENT_WEIGHTS_INT8),
        Some(&ENC_GRU5_RECURRENT_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_GRU5_RECURRENT_SCALE),
        32,
        96,
    );
    model.enc_conv1 = linear_layer(
        Some(&ENC_CONV1_BIAS),
        Some(&ENC_CONV1_SUBIAS),
        Some(&ENC_CONV1_WEIGHTS_INT8),
        Some(&ENC_CONV1_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_CONV1_SCALE),
        128,
        64,
    );
    model.enc_conv2 = linear_layer(
        Some(&ENC_CONV2_BIAS),
        Some(&ENC_CONV2_SUBIAS),
        Some(&ENC_CONV2_WEIGHTS_INT8),
        Some(&ENC_CONV2_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_CONV2_SCALE),
        128,
        64,
    );
    model.enc_conv3 = linear_layer(
        Some(&ENC_CONV3_BIAS),
        Some(&ENC_CONV3_SUBIAS),
        Some(&ENC_CONV3_WEIGHTS_INT8),
        Some(&ENC_CONV3_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_CONV3_SCALE),
        128,
        64,
    );
    model.enc_conv4 = linear_layer(
        Some(&ENC_CONV4_BIAS),
        Some(&ENC_CONV4_SUBIAS),
        Some(&ENC_CONV4_WEIGHTS_INT8),
        Some(&ENC_CONV4_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_CONV4_SCALE),
        128,
        64,
    );
    model.enc_conv5 = linear_layer(
        Some(&ENC_CONV5_BIAS),
        Some(&ENC_CONV5_SUBIAS),
        Some(&ENC_CONV5_WEIGHTS_INT8),
        Some(&ENC_CONV5_WEIGHTS_FLOAT),
        None,
        None,
        Some(&ENC_CONV5_SCALE),
        128,
        64,
    );
}

fn bytes_len(blob: &WeightBlob<'_>, name: &'static str) -> Result<usize, WeightError> {
    let array = blob.find(name).ok_or(WeightError::MissingArray(name))?;
    Ok(array.size)
}

fn decode_f32(data: &[u8]) -> Result<Cow<'static, [f32]>, WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut out = Vec::with_capacity(data.len() / core::mem::size_of::<f32>());
    let mut chunk = data;
    while chunk.len() >= 4 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&chunk[..4]);
        out.push(f32::from_le_bytes(buf));
        chunk = &chunk[4..];
    }
    Ok(Cow::Owned(out))
}

fn decode_i8(data: &[u8]) -> Result<Cow<'static, [i8]>, WeightError> {
    let mut out = Vec::with_capacity(data.len());
    out.extend(data.iter().map(|&v| v as i8));
    Ok(Cow::Owned(out))
}

fn decode_i32(data: &[u8]) -> Result<Cow<'static, [i32]>, WeightError> {
    if data.len() % 4 != 0 {
        return Err(WeightError::InvalidBlob);
    }
    let mut out = Vec::with_capacity(data.len() / core::mem::size_of::<i32>());
    let mut chunk = data;
    while chunk.len() >= 4 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&chunk[..4]);
        out.push(i32::from_le_bytes(buf));
        chunk = &chunk[4..];
    }
    Ok(Cow::Owned(out))
}

fn validate_sparse_idx(
    idx: &[i32],
    nb_inputs: usize,
    nb_outputs: usize,
    name: &'static str,
) -> Result<usize, WeightError> {
    let mut remain = idx.len() as i32;
    let mut out = nb_outputs as i32;
    let mut total_blocks = 0i32;
    let mut pos = 0usize;
    while remain > 0 {
        let nb_blocks = idx[pos];
        pos += 1;
        remain -= 1;
        if nb_blocks < 0 || remain < nb_blocks {
            return Err(WeightError::InvalidIndex(name));
        }
        for _ in 0..nb_blocks {
            let offset = idx[pos];
            pos += 1;
            remain -= 1;
            if offset < 0 || offset + 3 >= nb_inputs as i32 || (offset & 0x3) != 0 {
                return Err(WeightError::InvalidIndex(name));
            }
        }
        out -= 8;
        total_blocks += nb_blocks;
    }
    if out != 0 {
        return Err(WeightError::InvalidIndex(name));
    }
    Ok(total_blocks as usize)
}

fn linear_layer_from_weights(
    blob: &WeightBlob<'_>,
    bias: Option<&'static str>,
    subias: Option<&'static str>,
    weights: Option<&'static str>,
    float_weights: Option<&'static str>,
    weights_idx: Option<&'static str>,
    diag: Option<&'static str>,
    scale: Option<&'static str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<OwnedLinearLayer, WeightError> {
    let bias = match bias {
        Some(name) => Some(decode_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };
    let subias = match subias {
        Some(name) => Some(decode_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };
    let diag = match diag {
        Some(name) => Some(decode_f32(require_bytes(
            blob,
            name,
            nb_outputs * core::mem::size_of::<f32>(),
        )?)?),
        None => None,
    };

    let mut total_blocks = None;
    let weights_idx = match weights_idx {
        Some(name) => {
            let data = require_bytes(blob, name, bytes_len(blob, name)?)?;
            let idx = decode_i32(data)?;
            let blocks = validate_sparse_idx(&idx, nb_inputs, nb_outputs, name)?;
            total_blocks = Some(blocks);
            Some(idx)
        }
        None => None,
    };

    let (weights, float_weights) = if let Some(blocks) = total_blocks {
        let expected_i8 = 32usize
            .checked_mul(blocks)
            .ok_or(WeightError::InvalidBlob)?;
        let weights = match weights {
            Some(name) => Some(decode_i8(require_bytes(blob, name, expected_i8)?)?),
            None => None,
        };
        let expected_f32 = expected_i8
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(WeightError::InvalidBlob)?;
        let float_weights = match float_weights {
            Some(name) => match optional_bytes(blob, name, expected_f32)? {
                Some(data) => Some(decode_f32(data)?),
                None => None,
            },
            None => None,
        };
        (weights, float_weights)
    } else {
        let expected_i8 = nb_inputs
            .checked_mul(nb_outputs)
            .ok_or(WeightError::InvalidBlob)?;
        let weights = match weights {
            Some(name) => Some(decode_i8(require_bytes(blob, name, expected_i8)?)?),
            None => None,
        };
        let expected_f32 = expected_i8
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(WeightError::InvalidBlob)?;
        let float_weights = match float_weights {
            Some(name) => match optional_bytes(blob, name, expected_f32)? {
                Some(data) => Some(decode_f32(data)?),
                None => None,
            },
            None => None,
        };
        (weights, float_weights)
    };

    let scale = match scale {
        Some(name) => match weights {
            Some(_) => Some(decode_f32(require_bytes(
                blob,
                name,
                nb_outputs * core::mem::size_of::<f32>(),
            )?)?),
            None => None,
        },
        None => None,
    };

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

fn init_rdovaeenc_from_weights(
    model: &mut RdovaeEnc,
    blob: &WeightBlob<'_>,
) -> Result<(), WeightError> {
    model.enc_dense1 = linear_layer_from_weights(
        blob,
        Some("enc_dense1_bias"),
        None,
        None,
        Some("enc_dense1_weights_float"),
        None,
        None,
        None,
        40,
        64,
    )?;
    model.enc_zdense = linear_layer_from_weights(
        blob,
        Some("enc_zdense_bias"),
        Some("enc_zdense_subias"),
        Some("enc_zdense_weights_int8"),
        Some("enc_zdense_weights_float"),
        None,
        None,
        Some("enc_zdense_scale"),
        544,
        32,
    )?;
    model.gdense2 = linear_layer_from_weights(
        blob,
        Some("gdense2_bias"),
        Some("gdense2_subias"),
        Some("gdense2_weights_int8"),
        Some("gdense2_weights_float"),
        None,
        None,
        Some("gdense2_scale"),
        128,
        56,
    )?;
    model.gdense1 = linear_layer_from_weights(
        blob,
        Some("gdense1_bias"),
        Some("gdense1_subias"),
        Some("gdense1_weights_int8"),
        Some("gdense1_weights_float"),
        Some("gdense1_weights_idx"),
        None,
        Some("gdense1_scale"),
        544,
        128,
    )?;
    model.enc_conv_dense1 = linear_layer_from_weights(
        blob,
        Some("enc_conv_dense1_bias"),
        Some("enc_conv_dense1_subias"),
        Some("enc_conv_dense1_weights_int8"),
        Some("enc_conv_dense1_weights_float"),
        Some("enc_conv_dense1_weights_idx"),
        None,
        Some("enc_conv_dense1_scale"),
        96,
        64,
    )?;
    model.enc_conv_dense2 = linear_layer_from_weights(
        blob,
        Some("enc_conv_dense2_bias"),
        Some("enc_conv_dense2_subias"),
        Some("enc_conv_dense2_weights_int8"),
        Some("enc_conv_dense2_weights_float"),
        Some("enc_conv_dense2_weights_idx"),
        None,
        Some("enc_conv_dense2_scale"),
        192,
        64,
    )?;
    model.enc_conv_dense3 = linear_layer_from_weights(
        blob,
        Some("enc_conv_dense3_bias"),
        Some("enc_conv_dense3_subias"),
        Some("enc_conv_dense3_weights_int8"),
        Some("enc_conv_dense3_weights_float"),
        Some("enc_conv_dense3_weights_idx"),
        None,
        Some("enc_conv_dense3_scale"),
        288,
        64,
    )?;
    model.enc_conv_dense4 = linear_layer_from_weights(
        blob,
        Some("enc_conv_dense4_bias"),
        Some("enc_conv_dense4_subias"),
        Some("enc_conv_dense4_weights_int8"),
        Some("enc_conv_dense4_weights_float"),
        Some("enc_conv_dense4_weights_idx"),
        None,
        Some("enc_conv_dense4_scale"),
        384,
        64,
    )?;
    model.enc_conv_dense5 = linear_layer_from_weights(
        blob,
        Some("enc_conv_dense5_bias"),
        Some("enc_conv_dense5_subias"),
        Some("enc_conv_dense5_weights_int8"),
        Some("enc_conv_dense5_weights_float"),
        Some("enc_conv_dense5_weights_idx"),
        None,
        Some("enc_conv_dense5_scale"),
        480,
        64,
    )?;
    model.enc_gru1_input = linear_layer_from_weights(
        blob,
        Some("enc_gru1_input_bias"),
        Some("enc_gru1_input_subias"),
        Some("enc_gru1_input_weights_int8"),
        Some("enc_gru1_input_weights_float"),
        Some("enc_gru1_input_weights_idx"),
        None,
        Some("enc_gru1_input_scale"),
        64,
        96,
    )?;
    model.enc_gru1_recurrent = linear_layer_from_weights(
        blob,
        Some("enc_gru1_recurrent_bias"),
        Some("enc_gru1_recurrent_subias"),
        Some("enc_gru1_recurrent_weights_int8"),
        Some("enc_gru1_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru1_recurrent_scale"),
        32,
        96,
    )?;
    model.enc_gru2_input = linear_layer_from_weights(
        blob,
        Some("enc_gru2_input_bias"),
        Some("enc_gru2_input_subias"),
        Some("enc_gru2_input_weights_int8"),
        Some("enc_gru2_input_weights_float"),
        Some("enc_gru2_input_weights_idx"),
        None,
        Some("enc_gru2_input_scale"),
        160,
        96,
    )?;
    model.enc_gru2_recurrent = linear_layer_from_weights(
        blob,
        Some("enc_gru2_recurrent_bias"),
        Some("enc_gru2_recurrent_subias"),
        Some("enc_gru2_recurrent_weights_int8"),
        Some("enc_gru2_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru2_recurrent_scale"),
        32,
        96,
    )?;
    model.enc_gru3_input = linear_layer_from_weights(
        blob,
        Some("enc_gru3_input_bias"),
        Some("enc_gru3_input_subias"),
        Some("enc_gru3_input_weights_int8"),
        Some("enc_gru3_input_weights_float"),
        Some("enc_gru3_input_weights_idx"),
        None,
        Some("enc_gru3_input_scale"),
        256,
        96,
    )?;
    model.enc_gru3_recurrent = linear_layer_from_weights(
        blob,
        Some("enc_gru3_recurrent_bias"),
        Some("enc_gru3_recurrent_subias"),
        Some("enc_gru3_recurrent_weights_int8"),
        Some("enc_gru3_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru3_recurrent_scale"),
        32,
        96,
    )?;
    model.enc_gru4_input = linear_layer_from_weights(
        blob,
        Some("enc_gru4_input_bias"),
        Some("enc_gru4_input_subias"),
        Some("enc_gru4_input_weights_int8"),
        Some("enc_gru4_input_weights_float"),
        Some("enc_gru4_input_weights_idx"),
        None,
        Some("enc_gru4_input_scale"),
        352,
        96,
    )?;
    model.enc_gru4_recurrent = linear_layer_from_weights(
        blob,
        Some("enc_gru4_recurrent_bias"),
        Some("enc_gru4_recurrent_subias"),
        Some("enc_gru4_recurrent_weights_int8"),
        Some("enc_gru4_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru4_recurrent_scale"),
        32,
        96,
    )?;
    model.enc_gru5_input = linear_layer_from_weights(
        blob,
        Some("enc_gru5_input_bias"),
        Some("enc_gru5_input_subias"),
        Some("enc_gru5_input_weights_int8"),
        Some("enc_gru5_input_weights_float"),
        Some("enc_gru5_input_weights_idx"),
        None,
        Some("enc_gru5_input_scale"),
        448,
        96,
    )?;
    model.enc_gru5_recurrent = linear_layer_from_weights(
        blob,
        Some("enc_gru5_recurrent_bias"),
        Some("enc_gru5_recurrent_subias"),
        Some("enc_gru5_recurrent_weights_int8"),
        Some("enc_gru5_recurrent_weights_float"),
        None,
        None,
        Some("enc_gru5_recurrent_scale"),
        32,
        96,
    )?;
    model.enc_conv1 = linear_layer_from_weights(
        blob,
        Some("enc_conv1_bias"),
        Some("enc_conv1_subias"),
        Some("enc_conv1_weights_int8"),
        Some("enc_conv1_weights_float"),
        None,
        None,
        Some("enc_conv1_scale"),
        128,
        64,
    )?;
    model.enc_conv2 = linear_layer_from_weights(
        blob,
        Some("enc_conv2_bias"),
        Some("enc_conv2_subias"),
        Some("enc_conv2_weights_int8"),
        Some("enc_conv2_weights_float"),
        None,
        None,
        Some("enc_conv2_scale"),
        128,
        64,
    )?;
    model.enc_conv3 = linear_layer_from_weights(
        blob,
        Some("enc_conv3_bias"),
        Some("enc_conv3_subias"),
        Some("enc_conv3_weights_int8"),
        Some("enc_conv3_weights_float"),
        None,
        None,
        Some("enc_conv3_scale"),
        128,
        64,
    )?;
    model.enc_conv4 = linear_layer_from_weights(
        blob,
        Some("enc_conv4_bias"),
        Some("enc_conv4_subias"),
        Some("enc_conv4_weights_int8"),
        Some("enc_conv4_weights_float"),
        None,
        None,
        Some("enc_conv4_scale"),
        128,
        64,
    )?;
    model.enc_conv5 = linear_layer_from_weights(
        blob,
        Some("enc_conv5_bias"),
        Some("enc_conv5_subias"),
        Some("enc_conv5_weights_int8"),
        Some("enc_conv5_weights_float"),
        None,
        None,
        Some("enc_conv5_scale"),
        128,
        64,
    )?;
    Ok(())
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use crate::dred_rdovae_dec::{
        RdovaeDec, RdovaeDecState, rdovae_dec_init_states, rdovae_decode_qframe,
    };
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
    fn rdovae_matches_pinned_scalar_c() {
        let enc = RdovaeEnc::new();
        let dec = RdovaeDec::new();
        let mut es = RdovaeEncState::default();
        let mut ds = RdovaeDecState::default();
        let mut lines = {
            #[cfg(feature = "dnn_debug_float")]
            const VECTORS: &str = include_str!("../tests/fixtures/reference/dred-networks.txt");
            #[cfg(not(feature = "dnn_debug_float"))]
            const VECTORS: &str =
                include_str!("../tests/fixtures/reference/dred-networks-quantized.txt");
            VECTORS
        }
        .lines();
        for frame in 0..16usize {
            let input = core::array::from_fn::<_, 40, _>(|k| {
                (((k * 19 + frame * 23) % 127) as i32 - 63) as f32 * (1.0 / 64.0)
            });
            let mut latent = [0.0; DRED_LATENT_DIM];
            let mut state = [0.0; DRED_STATE_DIM];
            let mut features = [0.0; 80];
            dred_rdovae_encode_dframe(&mut es, &enc, &mut latent, &mut state, &input, 0);
            check(lines.next().unwrap(), &latent);
            check(lines.next().unwrap(), &state);
            if frame == 0 || frame == 8 {
                rdovae_dec_init_states(&mut ds, &dec, &state, 0);
            }
            let mut conditioned = [0.0; DRED_LATENT_DIM + 1];
            conditioned[..DRED_LATENT_DIM].copy_from_slice(&latent);
            conditioned[DRED_LATENT_DIM] = (frame % 16) as f32 * 0.125 - 1.0;
            rdovae_decode_qframe(&mut ds, &dec, &mut features, &conditioned, 0);
            check(lines.next().unwrap(), &features);
        }
        assert!(lines.next().is_none());
    }
}
