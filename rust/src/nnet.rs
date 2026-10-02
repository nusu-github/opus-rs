//! Neural network helpers for DRED/PLC inference.
//!
//! This is a scalar-only port of the C helpers in `dnn/nnet.c` and `dnn/vec.h`.

use crate::dred_constants::DRED_MAX_CONV_INPUTS;
use alloc::borrow::Cow;

const NNET_MAX_RNN_NEURONS: usize = 512;
const MAX_CONV_INPUTS_ALL: usize = if DRED_MAX_CONV_INPUTS > 1024 {
    DRED_MAX_CONV_INPUTS
} else {
    1024
};

pub(crate) const ACTIVATION_LINEAR: i32 = 0;
pub(crate) const ACTIVATION_SIGMOID: i32 = 1;
pub(crate) const ACTIVATION_TANH: i32 = 2;
pub(crate) const ACTIVATION_RELU: i32 = 3;
pub(crate) const ACTIVATION_SOFTMAX: i32 = 4;
pub(crate) const ACTIVATION_SWISH: i32 = 5;
pub(crate) const ACTIVATION_EXP: i32 = 6;

const MAX_INPUTS: usize = 2048;
const SPARSE_BLOCK_SIZE: usize = 32;
const MAX_CONV2D_INPUTS: usize = 8192;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearLayer<'a> {
    pub bias: Option<&'a [f32]>,
    #[allow(dead_code)]
    pub subias: Option<&'a [f32]>,
    pub weights: Option<&'a [i8]>,
    pub float_weights: Option<&'a [f32]>,
    pub weights_idx: Option<&'a [i32]>,
    pub diag: Option<&'a [f32]>,
    pub scale: Option<&'a [f32]>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

/// Model-owned parameters, retaining zero-copy access to compiled weights.
/// Runtime-loaded buffers are released when the model is replaced or dropped.
#[derive(Clone, Debug, Default)]
pub(crate) struct OwnedLinearLayer {
    pub bias: Option<Cow<'static, [f32]>>,
    pub subias: Option<Cow<'static, [f32]>>,
    pub weights: Option<Cow<'static, [i8]>>,
    pub float_weights: Option<Cow<'static, [f32]>>,
    pub weights_idx: Option<Cow<'static, [i32]>>,
    pub diag: Option<Cow<'static, [f32]>>,
    pub scale: Option<Cow<'static, [f32]>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

impl OwnedLinearLayer {
    pub(crate) fn view(&self) -> LinearLayer<'_> {
        LinearLayer {
            bias: self.bias.as_deref(),
            subias: self.subias.as_deref(),
            weights: self.weights.as_deref(),
            float_weights: self.float_weights.as_deref(),
            weights_idx: self.weights_idx.as_deref(),
            diag: self.diag.as_deref(),
            scale: self.scale.as_deref(),
            nb_inputs: self.nb_inputs,
            nb_outputs: self.nb_outputs,
        }
    }
}

impl From<LinearLayer<'static>> for OwnedLinearLayer {
    fn from(layer: LinearLayer<'static>) -> Self {
        Self {
            bias: layer.bias.map(Cow::Borrowed),
            subias: layer.subias.map(Cow::Borrowed),
            weights: layer.weights.map(Cow::Borrowed),
            float_weights: layer.float_weights.map(Cow::Borrowed),
            weights_idx: layer.weights_idx.map(Cow::Borrowed),
            diag: layer.diag.map(Cow::Borrowed),
            scale: layer.scale.map(Cow::Borrowed),
            nb_inputs: layer.nb_inputs,
            nb_outputs: layer.nb_outputs,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Conv2dLayer<'a> {
    pub bias: Option<&'a [f32]>,
    pub float_weights: Option<&'a [f32]>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

/// Convolution parameters with scoped ownership for runtime model reloads.
#[derive(Clone, Debug, Default)]
pub(crate) struct OwnedConv2dLayer {
    pub bias: Option<Cow<'static, [f32]>>,
    pub float_weights: Option<Cow<'static, [f32]>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

impl OwnedConv2dLayer {
    pub(crate) fn view(&self) -> Conv2dLayer<'_> {
        Conv2dLayer {
            bias: self.bias.as_deref(),
            float_weights: self.float_weights.as_deref(),
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            ktime: self.ktime,
            kheight: self.kheight,
        }
    }
}

impl From<Conv2dLayer<'static>> for OwnedConv2dLayer {
    fn from(layer: Conv2dLayer<'static>) -> Self {
        Self {
            bias: layer.bias.map(Cow::Borrowed),
            float_weights: layer.float_weights.map(Cow::Borrowed),
            in_channels: layer.in_channels,
            out_channels: layer.out_channels,
            ktime: layer.ktime,
            kheight: layer.kheight,
        }
    }
}

#[inline]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[allow(clippy::excessive_precision)]
#[inline]
fn tanh_approx(x: f32) -> f32 {
    const N0: f32 = 952.528_015_14;
    const N1: f32 = 96.392_356_87;
    const N2: f32 = 0.608_630_42;
    const D0: f32 = 952.723_999_02;
    const D1: f32 = 413.368_011_47;
    const D2: f32 = 11.886_009_22;

    let x2 = x * x;
    let num = fmadd(fmadd(N2, x2, N1), x2, N0);
    let den = fmadd(fmadd(D2, x2, D1), x2, D0);
    let value = num * x / den;
    value.clamp(-1.0, 1.0)
}

#[inline]
fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

fn lpcnet_exp(x: f32) -> f32 {
    let x = x * 1.44269504f32;
    let integer = libm::floorf(x) as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    let value =
        0.99992522f32 + frac * (0.69583354f32 + frac * (0.22606716f32 + 0.078024523f32 * frac));
    f32::from_bits(
        value
            .to_bits()
            .wrapping_add((integer as u32).wrapping_shl(23))
            & 0x7fffffff,
    )
}

fn softmax(output: &mut [f32], input: &[f32]) {
    let mut sum = 0.0f32;
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        let value = lpcnet_exp(src);
        *dst = value;
        sum += value;
    }
    let scale = 1.0 / (sum + 1.0e-30);
    for dst in output.iter_mut() {
        *dst *= scale;
    }
}

fn vec_tanh_inplace(output: &mut [f32]) {
    for value in output.iter_mut() {
        *value = tanh_approx(*value);
    }
}

fn vec_sigmoid_inplace(output: &mut [f32]) {
    for value in output.iter_mut() {
        *value = sigmoid_approx(*value);
    }
}

fn vec_swish_inplace(output: &mut [f32]) {
    let count = output.len();
    let mut tmp = [0.0f32; MAX_INPUTS];
    debug_assert!(count <= tmp.len());
    tmp[..count].copy_from_slice(output);
    vec_sigmoid_inplace(&mut tmp[..count]);
    for idx in 0..count {
        output[idx] = output[idx] * tmp[idx];
    }
}

pub(crate) fn compute_activation(output: &mut [f32], activation: i32) {
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid_inplace(output),
        ACTIVATION_TANH => vec_tanh_inplace(output),
        ACTIVATION_SWISH => vec_swish_inplace(output),
        ACTIVATION_RELU => {
            for dst in output.iter_mut() {
                *dst = (*dst).max(0.0);
            }
        }
        ACTIVATION_EXP => {
            for value in output.iter_mut() {
                *value = lpcnet_exp(*value);
            }
        }
        ACTIVATION_SOFTMAX => {
            let count = output.len();
            let mut tmp = [0.0f32; MAX_INPUTS];
            debug_assert!(count <= tmp.len());
            tmp[..count].copy_from_slice(output);
            softmax(output, &tmp[..count]);
        }
        _ => {
            debug_assert_eq!(activation, ACTIVATION_LINEAR);
        }
    }
}

fn conv2d_float(
    output: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        out_row.fill(0.0);
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    let weight_base = ((i * in_channels + m) * ktime + t) * kheight + h;
                    let weight = weights[weight_base];
                    let input_base = (t * in_channels + m) * in_stride + h;
                    for j in 0..height {
                        out_row[j] += weight * input[input_base + j];
                    }
                }
            }
        }
    }
}

fn conv2d_3x3_float(
    output: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let kheight = 3;
    let ktime = 3;
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        out_row.fill(0.0);
        for m in 0..in_channels {
            for j in 0..height {
                let weight_base = (i * in_channels + m) * ktime * kheight;
                let input_base = m * in_stride + j;
                out_row[j] += weights[weight_base + 0] * input[input_base + 0]
                    + weights[weight_base + 1] * input[input_base + 1]
                    + weights[weight_base + 2] * input[input_base + 2]
                    + weights[weight_base + 3] * input[input_base + in_channels * in_stride + 0]
                    + weights[weight_base + 4] * input[input_base + in_channels * in_stride + 1]
                    + weights[weight_base + 5] * input[input_base + in_channels * in_stride + 2]
                    + weights[weight_base + 6]
                        * input[input_base + 2 * in_channels * in_stride + 0]
                    + weights[weight_base + 7]
                        * input[input_base + 2 * in_channels * in_stride + 1]
                    + weights[weight_base + 8]
                        * input[input_base + 2 * in_channels * in_stride + 2];
            }
        }
    }
}

pub(crate) fn compute_conv2d(
    layer: &Conv2dLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
    _arch: i32,
) {
    let Some(weights) = layer.float_weights else {
        output.fill(0.0);
        return;
    };

    let time_stride = layer
        .in_channels
        .checked_mul(height + layer.kheight - 1)
        .expect("conv2d time stride overflow");
    let total_inputs = layer
        .ktime
        .checked_mul(time_stride)
        .expect("conv2d input length overflow");
    debug_assert!(
        total_inputs <= MAX_CONV2D_INPUTS,
        "conv2d input buffer too large"
    );
    let mem_len = (layer.ktime - 1) * time_stride;
    debug_assert!(mem_len <= mem.len());
    debug_assert!(time_stride <= input.len());
    debug_assert!(output.len() >= layer.out_channels * hstride);

    let mut input_buf = [0.0f32; MAX_CONV2D_INPUTS];
    input_buf[..mem_len].copy_from_slice(&mem[..mem_len]);
    input_buf[mem_len..mem_len + time_stride].copy_from_slice(&input[..time_stride]);
    if mem_len > 0 {
        let start = time_stride;
        let end = start + mem_len;
        mem[..mem_len].copy_from_slice(&input_buf[start..end]);
    }

    if layer.kheight == 3 && layer.ktime == 3 {
        conv2d_3x3_float(
            output,
            weights,
            layer.in_channels,
            layer.out_channels,
            &input_buf[..total_inputs],
            height,
            hstride,
        );
    } else {
        conv2d_float(
            output,
            weights,
            layer.in_channels,
            layer.out_channels,
            layer.ktime,
            layer.kheight,
            &input_buf[..total_inputs],
            height,
            hstride,
        );
    }

    if let Some(bias) = layer.bias {
        debug_assert!(bias.len() >= layer.out_channels);
        for i in 0..layer.out_channels {
            let out_row = &mut output[i * hstride..i * hstride + height];
            let value = bias[i];
            for slot in out_row.iter_mut() {
                *slot += value;
            }
        }
    }

    for i in 0..layer.out_channels {
        let out_row = &mut output[i * hstride..i * hstride + height];
        compute_activation(out_row, activation);
    }
}

fn sgemv(out: &mut [f32], weights: &[f32], rows: usize, cols: usize, input: &[f32]) {
    out.fill(0.0);
    for i in 0..rows {
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += weights[j * rows + i] * input[j];
        }
        out[i] = acc;
    }
}

fn sparse_sgemv8x4(out: &mut [f32], weights: &[f32], idx: &[i32], rows: usize, input: &[f32]) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);

    let mut w_pos = 0usize;
    let mut idx_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            for column in 0..4 {
                for row_offset in 0..8 {
                    out[row + row_offset] +=
                        weights[w_pos + column * 8 + row_offset] * input[pos + column];
                }
            }
            w_pos += SPARSE_BLOCK_SIZE;
        }
        row += 8;
    }
}

fn quantize_input(int_buf: &mut [i8], input: &[f32]) {
    debug_assert!(input.len() <= int_buf.len());
    for (dst, &src) in int_buf.iter_mut().zip(input.iter()) {
        let value = libm::floor(f64::from(127.0 * src) + 0.5);
        *dst = (value as i32) as i8;
    }
}

fn sparse_cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);
    debug_assert!(cols <= MAX_INPUTS);

    let mut x = [0i8; MAX_INPUTS];
    quantize_input(&mut x[..cols], input);

    let mut w_pos = 0usize;
    let mut idx_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _ in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = x[pos] as i32;
            let xj1 = x[pos + 1] as i32;
            let xj2 = x[pos + 2] as i32;
            let xj3 = x[pos + 3] as i32;
            let y = &mut out[row..row + 8];
            y[0] += (weights[w_pos + 0] as i32 * xj0
                + weights[w_pos + 1] as i32 * xj1
                + weights[w_pos + 2] as i32 * xj2
                + weights[w_pos + 3] as i32 * xj3) as f32;
            y[1] += (weights[w_pos + 4] as i32 * xj0
                + weights[w_pos + 5] as i32 * xj1
                + weights[w_pos + 6] as i32 * xj2
                + weights[w_pos + 7] as i32 * xj3) as f32;
            y[2] += (weights[w_pos + 8] as i32 * xj0
                + weights[w_pos + 9] as i32 * xj1
                + weights[w_pos + 10] as i32 * xj2
                + weights[w_pos + 11] as i32 * xj3) as f32;
            y[3] += (weights[w_pos + 12] as i32 * xj0
                + weights[w_pos + 13] as i32 * xj1
                + weights[w_pos + 14] as i32 * xj2
                + weights[w_pos + 15] as i32 * xj3) as f32;
            y[4] += (weights[w_pos + 16] as i32 * xj0
                + weights[w_pos + 17] as i32 * xj1
                + weights[w_pos + 18] as i32 * xj2
                + weights[w_pos + 19] as i32 * xj3) as f32;
            y[5] += (weights[w_pos + 20] as i32 * xj0
                + weights[w_pos + 21] as i32 * xj1
                + weights[w_pos + 22] as i32 * xj2
                + weights[w_pos + 23] as i32 * xj3) as f32;
            y[6] += (weights[w_pos + 24] as i32 * xj0
                + weights[w_pos + 25] as i32 * xj1
                + weights[w_pos + 26] as i32 * xj2
                + weights[w_pos + 27] as i32 * xj3) as f32;
            y[7] += (weights[w_pos + 28] as i32 * xj0
                + weights[w_pos + 29] as i32 * xj1
                + weights[w_pos + 30] as i32 * xj2
                + weights[w_pos + 31] as i32 * xj3) as f32;
            w_pos += SPARSE_BLOCK_SIZE;
        }
        row += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

fn cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) {
    out.fill(0.0);
    debug_assert!(rows % 8 == 0);
    debug_assert!(cols <= MAX_INPUTS);

    let mut x = [0i8; MAX_INPUTS];
    quantize_input(&mut x[..cols], input);

    let mut w_pos = 0usize;
    let mut row = 0usize;
    while row < rows {
        let mut col = 0usize;
        while col < cols {
            let xj0 = x[col] as i32;
            let xj1 = x[col + 1] as i32;
            let xj2 = x[col + 2] as i32;
            let xj3 = x[col + 3] as i32;
            let y = &mut out[row..row + 8];
            y[0] += (weights[w_pos + 0] as i32 * xj0
                + weights[w_pos + 1] as i32 * xj1
                + weights[w_pos + 2] as i32 * xj2
                + weights[w_pos + 3] as i32 * xj3) as f32;
            y[1] += (weights[w_pos + 4] as i32 * xj0
                + weights[w_pos + 5] as i32 * xj1
                + weights[w_pos + 6] as i32 * xj2
                + weights[w_pos + 7] as i32 * xj3) as f32;
            y[2] += (weights[w_pos + 8] as i32 * xj0
                + weights[w_pos + 9] as i32 * xj1
                + weights[w_pos + 10] as i32 * xj2
                + weights[w_pos + 11] as i32 * xj3) as f32;
            y[3] += (weights[w_pos + 12] as i32 * xj0
                + weights[w_pos + 13] as i32 * xj1
                + weights[w_pos + 14] as i32 * xj2
                + weights[w_pos + 15] as i32 * xj3) as f32;
            y[4] += (weights[w_pos + 16] as i32 * xj0
                + weights[w_pos + 17] as i32 * xj1
                + weights[w_pos + 18] as i32 * xj2
                + weights[w_pos + 19] as i32 * xj3) as f32;
            y[5] += (weights[w_pos + 20] as i32 * xj0
                + weights[w_pos + 21] as i32 * xj1
                + weights[w_pos + 22] as i32 * xj2
                + weights[w_pos + 23] as i32 * xj3) as f32;
            y[6] += (weights[w_pos + 24] as i32 * xj0
                + weights[w_pos + 25] as i32 * xj1
                + weights[w_pos + 26] as i32 * xj2
                + weights[w_pos + 27] as i32 * xj3) as f32;
            y[7] += (weights[w_pos + 28] as i32 * xj0
                + weights[w_pos + 29] as i32 * xj1
                + weights[w_pos + 30] as i32 * xj2
                + weights[w_pos + 31] as i32 * xj3) as f32;
            w_pos += SPARSE_BLOCK_SIZE;
            col += 4;
        }
        row += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

fn compute_linear(layer: &LinearLayer, out: &mut [f32], input: &[f32]) {
    debug_assert!(input.len() >= layer.nb_inputs);
    debug_assert!(out.len() >= layer.nb_outputs);

    if let Some(float_weights) = layer.float_weights {
        if let Some(weights_idx) = layer.weights_idx {
            sparse_sgemv8x4(out, float_weights, weights_idx, layer.nb_outputs, input);
        } else {
            sgemv(out, float_weights, layer.nb_outputs, layer.nb_inputs, input);
        }
    } else if let Some(weights) = layer.weights {
        let scale = layer.scale.expect("quantized weights require scale values");
        if let Some(weights_idx) = layer.weights_idx {
            sparse_cgemv8x4(
                out,
                weights,
                weights_idx,
                scale,
                layer.nb_outputs,
                layer.nb_inputs,
                input,
            );
        } else {
            cgemv8x4(
                out,
                weights,
                scale,
                layer.nb_outputs,
                layer.nb_inputs,
                input,
            );
        }
    } else {
        out.fill(0.0);
    }

    if let Some(bias) = layer.bias {
        for (dst, &value) in out.iter_mut().take(layer.nb_outputs).zip(bias.iter()) {
            *dst += value;
        }
    }

    if let Some(diag) = layer.diag {
        let m = layer.nb_inputs;
        debug_assert_eq!(3 * m, layer.nb_outputs);
        for i in 0..m {
            out[i] += diag[i] * input[i];
            out[i + m] += diag[i + m] * input[i];
            out[i + 2 * m] += diag[i + 2 * m] * input[i];
        }
    }
}

pub(crate) fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
    _arch: i32,
) {
    compute_linear(layer, output, input);
    compute_activation(output, activation);
}

pub(crate) fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
    _arch: i32,
) {
    let n = recurrent_weights.nb_inputs;
    debug_assert_eq!(recurrent_weights.nb_outputs, 3 * n);
    debug_assert_eq!(input_weights.nb_outputs, recurrent_weights.nb_outputs);
    debug_assert!(n <= NNET_MAX_RNN_NEURONS);
    debug_assert!(state.len() >= n);

    let mut zrh = [0.0f32; 3 * NNET_MAX_RNN_NEURONS];
    let mut recur = [0.0f32; 3 * NNET_MAX_RNN_NEURONS];

    compute_linear(input_weights, &mut zrh[..3 * n], input);
    compute_linear(recurrent_weights, &mut recur[..3 * n], state);
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    compute_activation(&mut zrh[..2 * n], ACTIVATION_SIGMOID);
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i];
    }
    compute_activation(&mut zrh[2 * n..2 * n + n], ACTIVATION_TANH);
    for i in 0..n {
        let z = zrh[i];
        let h = zrh[2 * n + i];
        state[i] = z * state[i] + (1.0 - z) * h;
    }
}

pub(crate) fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32], _arch: i32) {
    debug_assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let mut act2 = [0.0f32; MAX_INPUTS];
    let count = layer.nb_outputs;
    debug_assert!(count <= act2.len());
    compute_linear(layer, &mut act2[..count], input);
    compute_activation(&mut act2[..count], ACTIVATION_SIGMOID);
    if core::ptr::eq(output.as_ptr(), input.as_ptr()) {
        for i in 0..count {
            output[i] *= act2[i];
        }
    } else {
        for i in 0..count {
            output[i] = input[i] * act2[i];
        }
    }
}

pub(crate) fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: i32,
    _arch: i32,
) {
    let mut tmp = [0.0f32; MAX_CONV_INPUTS_ALL];
    let total_inputs = layer.nb_inputs;
    debug_assert!(total_inputs <= tmp.len());
    debug_assert_eq!(input_size, input.len());

    if total_inputs != input_size {
        let offset = total_inputs - input_size;
        tmp[..offset].copy_from_slice(&mem[..offset]);
        tmp[offset..offset + input_size].copy_from_slice(input);
    } else {
        tmp[..input_size].copy_from_slice(input);
    }

    compute_linear(layer, output, &tmp[..total_inputs]);
    compute_activation(output, activation);

    if total_inputs != input_size {
        let offset = total_inputs - input_size;
        mem[..offset].copy_from_slice(&tmp[input_size..input_size + offset]);
    }
}

pub(crate) fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: i32,
    _arch: i32,
) {
    let mut tmp = [0.0f32; MAX_CONV_INPUTS_ALL];
    let total_inputs = layer.nb_inputs;
    debug_assert!(total_inputs <= tmp.len());
    debug_assert_eq!(input_size, input.len());
    let ksize = total_inputs / input_size;

    if dilation == 1 {
        let offset = total_inputs - input_size;
        tmp[..offset].copy_from_slice(&mem[..offset]);
    } else {
        for i in 0..ksize - 1 {
            let src = i * input_size * dilation;
            tmp[i * input_size..(i + 1) * input_size].copy_from_slice(&mem[src..src + input_size]);
        }
    }

    tmp[total_inputs - input_size..total_inputs].copy_from_slice(input);
    compute_linear(layer, output, &tmp[..total_inputs]);
    compute_activation(output, activation);

    if dilation == 1 {
        let offset = total_inputs - input_size;
        mem[..offset].copy_from_slice(&tmp[input_size..input_size + offset]);
    } else {
        let span = input_size * dilation * (ksize - 1) - input_size;
        mem.copy_within(input_size..input_size + span, 0);
        mem[span..span + input_size].copy_from_slice(input);
    }
}

#[cfg(test)]
mod scalar_reference_tests {
    use super::*;
    #[test]
    fn sparse_layout_quantization_and_exp_match_c() {
        let weights: [f32; 96] =
            core::array::from_fn(|k| ((k % 31) as i32 - 15) as f32 * 0.01234567);
        let mut input: [f32; 12] = core::array::from_fn(|k| (k as i32 - 6) as f32 * 0.1357911);
        let scale: [f32; 16] = core::array::from_fn(|k| (k + 1) as f32 * 0.00001234567);
        let quantized: [i8; 192] = core::array::from_fn(|k| ((k * 17 % 255) as i32 - 127) as i8);
        let indices = [2, 0, 8, 1, 4];
        let mut expected = include_str!("../tests/fixtures/reference/nnet-vectors.txt").lines();
        let mut output = [0.0; 16];
        sparse_sgemv8x4(&mut output, &weights, &indices, 16, &input);
        check(expected.next().unwrap(), &output);
        sparse_cgemv8x4(&mut output, &quantized, &indices, &scale, 16, 12, &input);
        check(expected.next().unwrap(), &output);
        cgemv8x4(&mut output, &quantized, &scale, 16, 12, &input);
        check(expected.next().unwrap(), &output);
        for k in 0..12 {
            input[k] = (k as i32 - 6) as f32 * 2.34567;
        }
        let mut activation = input;
        compute_activation(&mut activation, ACTIVATION_EXP);
        check(expected.next().unwrap(), &activation);
        activation = input;
        compute_activation(&mut activation, ACTIVATION_SOFTMAX);
        check(expected.next().unwrap(), &activation);
        assert!(expected.next().is_none());
    }
    fn check(line: &str, output: &[f32]) {
        let expected: alloc::vec::Vec<_> = line
            .split_whitespace()
            .map(|x| u32::from_str_radix(x, 16).unwrap())
            .collect();
        let actual: alloc::vec::Vec<_> = output.iter().map(|x| x.to_bits()).collect();
        assert_eq!(actual, expected);
    }
}
