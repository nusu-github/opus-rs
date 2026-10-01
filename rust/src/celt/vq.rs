#![allow(dead_code)]

//! Vector quantisation helpers ported from `celt/vq.c`.
//!
//! The routines in this module operate on the normalised coefficient buffers
//! used by CELT's pulse shaping stage.  They are largely self-contained and
//! map closely to their C counterparts, making them ideal candidates for early
//! porting efforts.

use alloc::vec;
use core::convert::TryFrom;

use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
#[cfg(feature = "fixed_point")]
use crate::celt::entcode::ec_ilog;
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_arch::{EPSILON as FIXED_EPSILON, Q15_ONE as Q15_ONE_FIXED};
#[cfg(feature = "fixed_point")]
use crate::celt::fixed_ops::{
    add32, extract16, mac16_16, mult16_16, mult16_16_q15, mult16_32_q15, mult16_32_q16,
    mult32_32_q31, pshr32, shr32, vshr32,
};
#[cfg(feature = "fixed_point")]
use crate::celt::math::celt_ilog2;
use crate::celt::math::{
    celt_atan2p_norm, celt_cos_norm, celt_div, celt_rcp, celt_rsqrt_norm, celt_sqrt,
};
#[cfg(feature = "fixed_point")]
use crate::celt::math_fixed::{
    celt_cos_norm as celt_cos_norm_fixed, celt_rcp as celt_rcp_fixed,
    celt_rsqrt_norm as celt_rsqrt_norm_fixed, celt_rsqrt_norm32,
};
use crate::celt::pitch::celt_inner_prod;
#[cfg(feature = "fixed_point")]
use crate::celt::types::FixedCeltNorm;
#[cfg(feature = "fixed_point")]
use crate::celt::types::{FixedOpusVal16, FixedOpusVal32};
use crate::celt::types::{OpusInt32, OpusVal16, OpusVal32};
use libm::floorf;

#[cfg(feature = "fixed_point")]
pub(crate) fn norm_scaledown(x: &mut [FixedCeltNorm], shift: u32) {
    if shift > 0 {
        for value in x {
            *value = pshr32(*value, shift);
        }
    }
}
#[cfg(feature = "fixed_point")]
pub(crate) fn norm_scaleup(x: &mut [FixedCeltNorm], shift: u32) {
    if shift > 0 {
        for value in x {
            *value = value.wrapping_shl(shift);
        }
    }
}
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_inner_prod_norm(x: &[FixedCeltNorm], y: &[FixedCeltNorm]) -> i32 {
    x.iter()
        .zip(y)
        .fold(0i32, |sum, (&a, &b)| sum.wrapping_add(a.wrapping_mul(b)))
}
#[cfg(feature = "fixed_point")]
pub(crate) fn celt_inner_prod_norm_shift(x: &[FixedCeltNorm], y: &[FixedCeltNorm]) -> i32 {
    (x.iter().zip(y).fold(0i64, |sum, (&a, &b)| {
        sum.wrapping_add(i64::from(a) * i64::from(b))
    }) >> 20) as i32
}

/// Spread decisions mirrored from `celt/bands.h`.
pub(crate) const SPREAD_NONE: i32 = 0;
pub(crate) const SPREAD_LIGHT: i32 = 1;
pub(crate) const SPREAD_NORMAL: i32 = 2;
pub(crate) const SPREAD_AGGRESSIVE: i32 = 3;

const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
const Q15_ONE: OpusVal16 = 1.0;
const EPSILON: OpusVal32 = 1e-15;
// CELT mode construction rejects any band wider than 208 coefficients, which
// lets the decoder mirror the C stack allocation here without heap traffic.
const MAX_PVQ_BAND_SIZE: usize = 208;

#[inline]
fn select_pvq_candidate_float(
    best_id: &mut usize,
    best_den: &mut OpusVal16,
    best_num: &mut OpusVal16,
    candidate_id: usize,
    candidate_den: OpusVal16,
    candidate_num: OpusVal16,
) {
    if *best_den * candidate_num > candidate_den * *best_num {
        *best_id = candidate_id;
        *best_den = candidate_den;
        *best_num = candidate_num;
    }
}

#[cfg(feature = "fixed_point")]
#[inline]
fn select_pvq_candidate_fixed(
    best_id: &mut usize,
    best_den: &mut FixedOpusVal16,
    best_num: &mut OpusInt32,
    candidate_id: usize,
    candidate_den: FixedOpusVal16,
    candidate_num: FixedOpusVal16,
) {
    let left = mult16_16(*best_den, candidate_num);
    let right = i32::from(candidate_den).wrapping_mul(*best_num);
    if left > right {
        *best_id = candidate_id;
        *best_den = candidate_den;
        *best_num = i32::from(candidate_num);
    }
}

#[cfg(feature = "fixed_point")]
#[inline]
fn add16(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_add(b)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn sub16(a: FixedOpusVal16, b: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_sub(b)
}

#[cfg(feature = "fixed_point")]
#[inline]
fn neg16(a: FixedOpusVal16) -> FixedOpusVal16 {
    a.wrapping_neg()
}

fn exp_rotation1(x: &mut [OpusVal16], stride: usize, c: OpusVal16, s: OpusVal16) {
    let len = x.len();
    if stride == 0 || len <= stride {
        return;
    }

    let ms = -s;

    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }

    if len > 2 * stride {
        let limit = len - 2 * stride - 1;
        for i in (0..=limit).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

/// Port of `exp_rotation()` from `celt/vq.c`.
///
/// Applies a spreading rotation to the coefficient buffer in-place.  The logic
/// matches the float build of the reference implementation, relying on Rust's
/// slice handling for safety while keeping the numerical behaviour intact.
pub(crate) fn exp_rotation(
    x: &mut [OpusVal16],
    len: usize,
    dir: i32,
    stride: usize,
    k: i32,
    spread: i32,
) {
    if len == 0 || stride == 0 {
        return;
    }

    let slice_len = len.min(x.len());
    let x = &mut x[..slice_len];

    if 2 * k >= slice_len as i32 || spread == SPREAD_NONE {
        return;
    }

    let spread_index = match spread {
        SPREAD_LIGHT => 0,
        SPREAD_NORMAL => 1,
        SPREAD_AGGRESSIVE => 2,
        _ => return,
    };

    let factor = SPREAD_FACTOR[spread_index];
    let gain = celt_div(
        Q15_ONE * slice_len as OpusVal16,
        (slice_len + (factor * k) as usize) as OpusVal16,
    );
    let theta = 0.5 * gain * gain;
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(Q15_ONE - theta);

    let mut stride2 = 0usize;
    if slice_len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < slice_len {
            stride2 += 1;
        }
    }

    let len_div = slice_len / stride;
    if len_div == 0 {
        return;
    }

    for band in 0..stride {
        let start = band * len_div;
        let end = start + len_div;
        let band_slice = &mut x[start..end];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, c);
            }
            exp_rotation1(band_slice, 1, c, s);
        } else {
            exp_rotation1(band_slice, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(band_slice, stride2, s, -c);
            }
        }
    }
}

#[cfg(feature = "fixed_point")]
fn exp_rotation1_fixed(
    x: &mut [FixedCeltNorm],
    stride: usize,
    c: FixedOpusVal16,
    s: FixedOpusVal16,
) {
    let len = x.len();
    if stride == 0 || len < stride {
        return;
    }

    norm_scaledown(x, 10);
    let ms = neg16(s);

    for i in 0..(len - stride) {
        let x1 = x[i] as i16;
        let x2 = x[i + stride] as i16;
        let acc1 = mac16_16(mult16_16(c, x2), s, x1);
        let acc2 = mac16_16(mult16_16(c, x1), ms, x2);
        x[i + stride] = i32::from(extract16(pshr32(acc1, 15)));
        x[i] = i32::from(extract16(pshr32(acc2, 15)));
    }

    if len > 2 * stride {
        let limit = len - 2 * stride - 1;
        for i in (0..=limit).rev() {
            let x1 = x[i] as i16;
            let x2 = x[i + stride] as i16;
            let acc1 = mac16_16(mult16_16(c, x2), s, x1);
            let acc2 = mac16_16(mult16_16(c, x1), ms, x2);
            x[i + stride] = i32::from(extract16(pshr32(acc1, 15)));
            x[i] = i32::from(extract16(pshr32(acc2, 15)));
        }
    }
    norm_scaleup(x, 10);
}

#[cfg(feature = "fixed_point")]
pub(crate) fn exp_rotation_fixed(
    x: &mut [FixedCeltNorm],
    len: usize,
    dir: i32,
    stride: usize,
    k: i32,
    spread: i32,
) {
    if len == 0 || stride == 0 {
        return;
    }

    let slice_len = len.min(x.len());
    let x = &mut x[..slice_len];

    if 2 * k >= slice_len as i32 || spread == SPREAD_NONE {
        return;
    }

    let spread_index = match spread {
        SPREAD_LIGHT => 0,
        SPREAD_NORMAL => 1,
        SPREAD_AGGRESSIVE => 2,
        _ => return,
    };

    let factor = SPREAD_FACTOR[spread_index];
    let denom = slice_len as i32 + factor * k;
    let numer = mult16_16(Q15_ONE_FIXED, slice_len as i16);
    let gain = extract16(mult32_32_q31(numer, celt_rcp_fixed(denom)));
    let theta = (mult16_16_q15(gain, gain) >> 1) as i16;
    let c = celt_cos_norm_fixed(i32::from(theta));
    let s = celt_cos_norm_fixed(i32::from(sub16(Q15_ONE_FIXED, theta)));

    let mut stride2 = 0usize;
    if slice_len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < slice_len {
            stride2 += 1;
        }
    }

    let len_div = celt_udiv(slice_len as u32, stride as u32) as usize;
    if len_div == 0 {
        return;
    }

    for band in 0..stride {
        let start = band * len_div;
        let end = start + len_div;
        let band_slice = &mut x[start..end];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1_fixed(band_slice, stride2, s, c);
            }
            exp_rotation1_fixed(band_slice, 1, c, s);
        } else {
            exp_rotation1_fixed(band_slice, 1, c, neg16(s));
            if stride2 > 0 {
                exp_rotation1_fixed(band_slice, stride2, s, neg16(c));
            }
        }
    }
}

/// Port of `normalise_residual()` from `celt/vq.c`.
///
/// The helper mixes the decoded PVQ pulses with the pitch vector so that the
/// resulting excitation has unit energy. The float build performs the scaling
/// by computing the reciprocal square root of the accumulated pulse energy and
/// multiplying by the supplied gain. The Rust port follows the same approach
/// while clamping the slice lengths to avoid overruns.
pub(crate) fn normalise_residual(
    pulses: &[i32],
    x: &mut [OpusVal16],
    n: usize,
    ryy: OpusVal32,
    gain: OpusVal32,
) {
    if n == 0 {
        return;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");
    debug_assert!(x.len() >= n, "output buffer shorter than band size");

    let len = n.min(pulses.len()).min(x.len());
    if len == 0 {
        return;
    }

    let scale = celt_rsqrt_norm(ryy) * gain;
    for (dst, &pulse) in x.iter_mut().take(len).zip(pulses.iter()) {
        *dst = scale * pulse as OpusVal16;
    }
}

#[cfg(feature = "fixed_point")]
pub(crate) fn normalise_residual_fixed(
    pulses: &[i32],
    x: &mut [FixedCeltNorm],
    n: usize,
    ryy: FixedOpusVal32,
    gain: FixedOpusVal32,
) {
    if n == 0 {
        return;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");
    debug_assert!(x.len() >= n, "output buffer shorter than band size");

    let len = n.min(pulses.len()).min(x.len());
    if len == 0 || ryy == 0 {
        return;
    }

    let k = celt_ilog2(ryy) >> 1;
    let t = vshr32(ryy, 2 * (k - 7) - 15);
    let g = mult32_32_q31(celt_rsqrt_norm32(t), gain);

    for (dst, &pulse) in x.iter_mut().take(len).zip(pulses.iter()) {
        *dst = vshr32(mult16_32_q15(pulse as i16, g), k + 15 - 24);
    }
}

/// Float port of `op_pvq_search_c()` from `celt/vq.c`.
///
/// The helper distributes `K` algebraic pulses across the `N`-dimensional
/// coefficient vector `x`, returning the squared energy of the chosen pulse
/// vector. The routine mirrors the reference implementation by performing a
/// greedy search that maximises the correlation proxy `Rxy / sqrt(Ryy)` without
/// taking expensive square roots inside the inner loop.
pub(crate) fn op_pvq_search(
    x: &mut [OpusVal16],
    pulses: &mut [OpusInt32],
    n: usize,
    k: i32,
    _arch: i32,
) -> OpusVal32 {
    assert!(n > 0, "vector dimension must be positive");
    assert!(k >= 0, "pulse count must be non-negative");
    assert!(x.len() >= n, "coefficient buffer shorter than band size");
    assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let mut y = vec![0.0f32; n];
    let mut sign = vec![false; n];

    for (idx, sample) in x.iter_mut().enumerate().take(n) {
        let value = *sample;
        sign[idx] = value < 0.0;
        *sample = value.abs();
        pulses[idx] = 0;
        y[idx] = 0.0;
    }

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k;

    if k > ((n as i32) >> 1) {
        let mut sum = 0.0f32;
        for &sample in x.iter().take(n) {
            sum += sample;
        }

        if !(sum > EPSILON && sum < 64.0) {
            if n > 0 {
                x[0] = 1.0;
                for coeff in x.iter_mut().take(n).skip(1) {
                    *coeff = 0.0;
                }
            }
            sum = 1.0;
        }

        let rcp = (k as OpusVal32 + 0.8) * celt_rcp(sum);
        for idx in 0..n {
            let projected = floorf(rcp * x[idx]);
            let pulse = projected as OpusInt32;
            pulses[idx] = pulse;
            let val = pulse as OpusVal16;
            y[idx] = val;
            yy += val * val;
            xy += x[idx] * val;
            y[idx] *= 2.0;
            pulses_left -= pulse;
        }
    }

    debug_assert!(pulses_left >= 0, "pulse allocation exceeded target count");
    if pulses_left < 0 {
        pulses_left = 0;
    }

    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as OpusVal16;
        yy += tmp * tmp;
        yy += tmp * y[0];
        pulses[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        yy += 1.0;

        let mut best_id = 0usize;
        let mut best_den = yy + y[0];
        let mut best_num = (xy + x[0]) * (xy + x[0]);

        let x_tail = &x[1..n];
        let y_tail = &y[1..n];
        let mut x_chunks = x_tail.chunks_exact(4);
        let mut y_chunks = y_tail.chunks_exact(4);
        for (chunk_idx, (x_chunk, y_chunk)) in (&mut x_chunks).zip(&mut y_chunks).enumerate() {
            let base = 1 + chunk_idx * 4;
            let rxy0 = xy + x_chunk[0];
            let ryy0 = yy + y_chunk[0];
            let num0 = rxy0 * rxy0;
            select_pvq_candidate_float(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base,
                ryy0,
                num0,
            );

            let rxy1 = xy + x_chunk[1];
            let ryy1 = yy + y_chunk[1];
            let num1 = rxy1 * rxy1;
            select_pvq_candidate_float(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 1,
                ryy1,
                num1,
            );

            let rxy2 = xy + x_chunk[2];
            let ryy2 = yy + y_chunk[2];
            let num2 = rxy2 * rxy2;
            select_pvq_candidate_float(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 2,
                ryy2,
                num2,
            );

            let rxy3 = xy + x_chunk[3];
            let ryy3 = yy + y_chunk[3];
            let num3 = rxy3 * rxy3;
            select_pvq_candidate_float(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 3,
                ryy3,
                num3,
            );
        }

        let rem_start = n - x_chunks.remainder().len();
        for idx in rem_start..n {
            let rxy = xy + x[idx];
            let ryy = yy + y[idx];
            let num = rxy * rxy;
            select_pvq_candidate_float(&mut best_id, &mut best_den, &mut best_num, idx, ryy, num);
        }

        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        pulses[best_id] += 1;
    }

    for (idx, pulse) in pulses.iter_mut().take(n).enumerate() {
        if sign[idx] {
            *pulse = -*pulse;
        }
    }

    yy
}

#[cfg(feature = "fixed_point")]
pub(crate) fn op_pvq_search_fixed(
    x: &mut [FixedCeltNorm],
    pulses: &mut [OpusInt32],
    n: usize,
    k: i32,
    _arch: i32,
) -> FixedOpusVal16 {
    assert!(n > 0, "vector dimension must be positive");
    assert!(k >= 0, "pulse count must be non-negative");
    assert!(x.len() >= n, "coefficient buffer shorter than band size");
    assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let shift = ((celt_ilog2(1 + celt_inner_prod_norm_shift(&x[..n], &x[..n])) + 1) / 2 - 4).max(0);
    norm_scaledown(&mut x[..n], shift as u32);
    let mut y = vec![0i16; n];
    let mut signx = vec![0i32; n];

    for j in 0..n {
        let value = x[j];
        signx[j] = i32::from(value < 0);
        x[j] = if value < 0 {
            value.wrapping_neg()
        } else {
            value
        };
        pulses[j] = 0;
        y[j] = 0;
    }

    let mut xy: i32 = 0;
    let mut yy: i16 = 0;
    let mut pulses_left = k;

    if k > ((n as i32) >> 1) {
        let mut sum: i32 = 0;
        for &sample in x.iter().take(n) {
            sum = sum.wrapping_add(sample as i32);
        }

        if sum <= k {
            if n > 0 {
                x[0] = 16_384;
                for coeff in x.iter_mut().take(n).skip(1) {
                    *coeff = 0;
                }
            }
            sum = 16_384;
        }

        let rcp = extract16(mult16_32_q16(k as i16, celt_rcp_fixed(sum)));
        for j in 0..n {
            let iy = mult16_16_q15(x[j] as i16, rcp) as i32;
            pulses[j] = iy;
            let yj = iy as i16;
            y[j] = yj;
            yy = extract16(mac16_16(i32::from(yy), yj, yj));
            xy = mac16_16(xy, x[j] as i16, yj);
            y[j] = y[j].wrapping_shl(1);
            pulses_left -= iy;
        }
    }

    debug_assert!(pulses_left >= 0, "pulse allocation exceeded target count");
    if pulses_left < 0 {
        pulses_left = 0;
    }

    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as i16;
        yy = extract16(mac16_16(i32::from(yy), tmp, tmp));
        yy = extract16(mac16_16(i32::from(yy), tmp, y[0]));
        pulses[0] += pulses_left;
        pulses_left = 0;
    }

    for i in 0..pulses_left {
        let rshift = 1 + celt_ilog2(k - pulses_left + i + 1);
        yy = add16(yy, 1);

        let mut best_id = 0usize;
        let rxy = extract16(shr32(add32(xy, i32::from(x[0])), rshift as u32));
        let ryy = add16(yy, y[0]);
        let rxy_sq = mult16_16_q15(rxy, rxy);
        let mut best_den = ryy;
        let mut best_num: i32 = i32::from(rxy_sq);

        let x_tail = &x[1..n];
        let y_tail = &y[1..n];
        let mut x_chunks = x_tail.chunks_exact(4);
        let mut y_chunks = y_tail.chunks_exact(4);
        for (chunk_idx, (x_chunk, y_chunk)) in (&mut x_chunks).zip(&mut y_chunks).enumerate() {
            let base = 1 + chunk_idx * 4;

            let rxy0 = extract16(shr32(add32(xy, i32::from(x_chunk[0])), rshift as u32));
            let ryy0 = add16(yy, y_chunk[0]);
            let num0 = mult16_16_q15(rxy0, rxy0);
            select_pvq_candidate_fixed(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base,
                ryy0,
                num0,
            );

            let rxy1 = extract16(shr32(add32(xy, i32::from(x_chunk[1])), rshift as u32));
            let ryy1 = add16(yy, y_chunk[1]);
            let num1 = mult16_16_q15(rxy1, rxy1);
            select_pvq_candidate_fixed(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 1,
                ryy1,
                num1,
            );

            let rxy2 = extract16(shr32(add32(xy, i32::from(x_chunk[2])), rshift as u32));
            let ryy2 = add16(yy, y_chunk[2]);
            let num2 = mult16_16_q15(rxy2, rxy2);
            select_pvq_candidate_fixed(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 2,
                ryy2,
                num2,
            );

            let rxy3 = extract16(shr32(add32(xy, i32::from(x_chunk[3])), rshift as u32));
            let ryy3 = add16(yy, y_chunk[3]);
            let num3 = mult16_16_q15(rxy3, rxy3);
            select_pvq_candidate_fixed(
                &mut best_id,
                &mut best_den,
                &mut best_num,
                base + 3,
                ryy3,
                num3,
            );
        }

        let rem_start = n - x_chunks.remainder().len();
        for j in rem_start..n {
            let rxy = extract16(shr32(add32(xy, i32::from(x[j])), rshift as u32));
            let ryy = add16(yy, y[j]);
            let rxy_sq = mult16_16_q15(rxy, rxy);
            select_pvq_candidate_fixed(&mut best_id, &mut best_den, &mut best_num, j, ryy, rxy_sq);
        }

        xy = add32(xy, i32::from(x[best_id]));
        yy = add16(yy, y[best_id]);
        y[best_id] = y[best_id].wrapping_add(2);
        pulses[best_id] += 1;
    }

    for j in 0..n {
        let sign = -signx[j];
        pulses[j] = (pulses[j] ^ sign) + signx[j];
    }

    yy
}

/// Algebraic pulse quantiser from `celt/vq.c`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alg_quant(
    x: &mut [OpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    enc: &mut EcEnc<'_>,
    gain: OpusVal32,
    resynth: bool,
    arch: i32,
) -> u32 {
    assert!(k > 0, "alg_quant requires at least one pulse");
    assert!(n > 1, "alg_quant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n + 3];

    exp_rotation(x, n, 1, b, k, spread);

    let yy = op_pvq_search(x, &mut pulses, n, k, arch);

    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    encode_pulses(&pulses[..n], n, total_pulses, enc);

    if resynth {
        normalise_residual(&pulses[..n], x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }

    extract_collapse_mask(&pulses[..n], n, b)
}

#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn alg_quant_fixed(
    x: &mut [FixedCeltNorm],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    enc: &mut EcEnc<'_>,
    gain: FixedOpusVal32,
    resynth: bool,
    arch: i32,
) -> u32 {
    assert!(k > 0, "alg_quant requires at least one pulse");
    assert!(n > 1, "alg_quant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");

    let mut pulses = vec![0i32; n + 3];

    exp_rotation_fixed(x, n, 1, b, k, spread);

    let yy = op_pvq_search_fixed(x, &mut pulses, n, k, arch);

    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    encode_pulses(&pulses[..n], n, total_pulses, enc);

    if resynth {
        normalise_residual_fixed(&pulses[..n], x, n, i32::from(yy), gain);
        exp_rotation_fixed(x, n, -1, b, k, spread);
    }

    extract_collapse_mask(&pulses[..n], n, b)
}

/// Algebraic pulse decoder mirroring `alg_unquant()`.
pub(crate) fn alg_unquant(
    x: &mut [OpusVal16],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    dec: &mut EcDec<'_>,
    gain: OpusVal32,
) -> u32 {
    assert!(k > 0, "alg_unquant requires at least one pulse");
    assert!(n > 1, "alg_unquant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");
    assert!(
        n <= MAX_PVQ_BAND_SIZE,
        "alg_unquant band size exceeds CELT maximum"
    );

    let mut pulses = [0i32; MAX_PVQ_BAND_SIZE];
    let pulses = &mut pulses[..n];
    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    let ryy = decode_pulses(pulses, n, total_pulses, dec);
    normalise_residual(pulses, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(pulses, n, b)
}

#[cfg(feature = "fixed_point")]
pub(crate) fn alg_unquant_fixed(
    x: &mut [FixedCeltNorm],
    n: usize,
    k: i32,
    spread: i32,
    b: usize,
    dec: &mut EcDec<'_>,
    gain: FixedOpusVal32,
) -> u32 {
    assert!(k > 0, "alg_unquant requires at least one pulse");
    assert!(n > 1, "alg_unquant requires at least two dimensions");
    assert!(x.len() >= n, "input vector shorter than band size");
    assert!(
        n <= MAX_PVQ_BAND_SIZE,
        "alg_unquant band size exceeds CELT maximum"
    );

    let mut pulses = [0i32; MAX_PVQ_BAND_SIZE];
    let pulses = &mut pulses[..n];
    let total_pulses = usize::try_from(k).expect("pulse count must fit in usize");
    let _ = decode_pulses(pulses, n, total_pulses, dec);

    let mut ryy: i32 = 0;
    for &pulse in pulses.iter() {
        ryy = mac16_16(ryy, pulse as i16, pulse as i16);
    }

    normalise_residual_fixed(pulses, x, n, ryy, gain);
    exp_rotation_fixed(x, n, -1, b, k, spread);
    extract_collapse_mask(pulses, n, b)
}

/// Renormalises a vector to unit gain, matching `renormalise_vector()`.
pub(crate) fn renormalise_vector(x: &mut [OpusVal16], n: usize, gain: OpusVal32, arch: i32) {
    assert!(x.len() >= n, "input vector shorter than band size");
    let len = n.min(x.len());
    let slice = &mut x[..len];

    let energy = EPSILON + celt_inner_prod(slice, slice);
    let scale = celt_rsqrt_norm(energy) * gain;

    if arch != 0 {
        let _ = arch;
    }

    for sample in slice.iter_mut() {
        *sample *= scale;
    }
}

#[cfg(feature = "fixed_point")]
pub(crate) fn renormalise_vector_fixed(
    x: &mut [FixedCeltNorm],
    n: usize,
    gain: FixedOpusVal32,
    arch: i32,
) {
    assert!(x.len() >= n, "input vector shorter than band size");
    let len = n.min(x.len());
    let slice = &mut x[..len];

    let mut energy = i32::from(FIXED_EPSILON);
    norm_scaledown(slice, 10);
    energy = energy.wrapping_add(celt_inner_prod_norm(slice, slice));
    let k = (ec_ilog(energy as u32) - 1) >> 1;
    let t = vshr32(energy, 2 * (k - 7));
    let g = mult32_32_q31(i32::from(celt_rsqrt_norm_fixed(t)), gain) as i16;

    if arch != 0 {
        let _ = arch;
    }

    for sample in slice.iter_mut() {
        let scaled = mult16_16(g, *sample as i16);
        *sample = i32::from(extract16(pshr32(scaled, (k + 1) as u32))).wrapping_shl(10);
    }
}

#[cfg(feature = "fixed_point")]
pub(crate) fn stereo_itheta_fixed(x: &[i32], y: &[i32], stereo: bool, n: usize, _arch: i32) -> i32 {
    let (mut mid, mut side) = (0i32, 0i32);
    if stereo {
        for i in 0..n {
            let m = pshr32(x[i].wrapping_add(y[i]), 11) as i16;
            let s = pshr32(x[i].wrapping_sub(y[i]), 11) as i16;
            mid = mac16_16(mid, m, m);
            side = mac16_16(side, s, s);
        }
    } else {
        mid = celt_inner_prod_norm_shift(&x[..n], &x[..n]);
        side = celt_inner_prod_norm_shift(&y[..n], &y[..n]);
    }
    crate::celt::math_fixed::celt_atan2p_norm(
        crate::celt::math_fixed::celt_sqrt32(side),
        crate::celt::math_fixed::celt_sqrt32(mid),
    ) >> 16
}

/// Computes the stereo intensity angle mirroring `stereo_itheta()`.
pub(crate) fn stereo_itheta(
    x: &[OpusVal16],
    y: &[OpusVal16],
    stereo: bool,
    n: usize,
    arch: i32,
) -> i32 {
    assert!(x.len() >= n, "mid channel shorter than requested length");
    assert!(y.len() >= n, "side channel shorter than requested length");

    let len = n.min(x.len()).min(y.len());
    let mut emid = 0.0;
    let mut eside = 0.0;

    if stereo {
        for i in 0..len {
            let m = x[i] + y[i];
            let s = x[i] - y[i];
            emid += m * m;
            eside += s * s;
        }
    } else {
        let mid = &x[..len];
        let side = &y[..len];
        emid += celt_inner_prod(mid, mid);
        eside += celt_inner_prod(side, side);
    }

    if arch != 0 {
        let _ = arch;
    }

    let mid = celt_sqrt(emid);
    let side = celt_sqrt(eside);
    let angle = celt_atan2p_norm(side, mid);

    (floorf(0.5 + 65_536.0 * 16_384.0 * angle) as i32) >> 16
}

/// Mirrors `extract_collapse_mask()` from `celt/vq.c`.
///
/// The helper inspects the quantised PVQ pulses and determines which of the
/// folded bands received any energy. The mask is later used to decide whether
/// a band "collapsed" during quantisation and therefore needs spectral
/// spreading in the decoder.
#[must_use]
pub(crate) fn extract_collapse_mask(pulses: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }

    if n == 0 {
        return 0;
    }

    debug_assert!(pulses.len() >= n, "pulse buffer shorter than band size");

    let n0 = celt_udiv(n as u32, b as u32) as usize;
    debug_assert!(n0 > 0, "sub-band width must be non-zero");

    let mut collapse_mask = 0u32;
    for band in 0..b {
        let start = band * n0;
        let end = (start + n0).min(pulses.len());
        let mut accumulator = 0;
        for &value in &pulses[start..end] {
            accumulator |= value;
        }
        if accumulator != 0 {
            collapse_mask |= 1 << band;
        }
    }

    collapse_mask
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    #[cfg(feature = "fixed_point")]
    use super::normalise_residual_fixed;
    use super::{
        SPREAD_NORMAL, alg_quant, alg_unquant, exp_rotation, extract_collapse_mask,
        normalise_residual, renormalise_vector, stereo_itheta,
    };
    use crate::celt::entdec::EcDec;
    use crate::celt::entenc::EcEnc;

    fn seed_samples(len: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let sample = ((seed >> 16) & 0x7fff) as i32 - 16_384;
            out.push(sample as f32);
        }
        out
    }

    fn snr_db(original: &[f32], processed: &[f32]) -> f64 {
        let mut err = 0.0;
        let mut ener = 0.0;
        for (&orig, &proc) in original.iter().zip(processed.iter()) {
            let diff = f64::from(orig - proc);
            err += diff * diff;
            ener += f64::from(orig) * f64::from(orig);
        }
        if err == 0.0 {
            return f64::INFINITY;
        }
        20.0 * (ener / err).log10()
    }

    fn rotation_case(len: usize, k: i32) {
        let baseline = seed_samples(len);
        let mut rotated = baseline.clone();

        exp_rotation(&mut rotated, len, 1, 1, k, SPREAD_NORMAL);
        let forward_snr = snr_db(&baseline, &rotated);

        exp_rotation(&mut rotated, len, -1, 1, k, SPREAD_NORMAL);
        let inverse_snr = snr_db(&baseline, &rotated);

        assert!(inverse_snr > 60.0, "inverse SNR too low: {inverse_snr}");
        assert!(
            forward_snr < 20.0,
            "forward SNR unexpectedly high: {forward_snr}"
        );
    }

    #[test]
    fn rotation_matches_reference_behaviour() {
        for &(len, k) in &[(15, 3), (23, 5), (50, 3), (80, 1)] {
            rotation_case(len, k);
        }
    }

    #[test]
    fn residual_normalisation_scales_by_gain() {
        let pulses = [2, -1, 0, 3];
        let mut output = [0.0f32; 4];
        let ryy = 25.0;
        let gain = 0.5;

        normalise_residual(&pulses, &mut output, pulses.len(), ryy, gain);

        let expected_scale = gain * (1.0 / ryy.sqrt());
        for (value, &pulse) in output.iter().zip(&pulses) {
            let expected = expected_scale * pulse as f32;
            assert!(
                (value - expected).abs() <= 1e-6,
                "expected {expected}, found {value}"
            );
        }
    }

    #[test]
    fn collapse_mask_sets_bits_for_active_bands() {
        let pulses = [0, 0, 1, 0, 0, 0, 2, 3];
        let mask = extract_collapse_mask(&pulses, pulses.len(), 4);
        assert_eq!(mask, 0b1010);
    }

    #[test]
    fn collapse_mask_for_single_band_is_one() {
        let pulses = [0, 0, 0, 0];
        assert_eq!(extract_collapse_mask(&pulses, pulses.len(), 1), 1);
    }

    #[test]
    fn op_pvq_search_distributes_requested_pulses() {
        let mut coeffs = [0.6f32, -0.4, 0.2, 0.1];
        let mut pulses = vec![0i32; coeffs.len()];
        let len = coeffs.len();
        let energy = super::op_pvq_search(&mut coeffs, &mut pulses, len, 5, 0);

        let total: i32 = pulses.iter().map(|&p| p.abs()).sum();
        assert_eq!(total, 5);
        assert!(energy > 0.0);
    }

    #[test]
    fn alg_quant_and_unquant_round_trip() {
        let coeffs = seed_samples(6);
        let mut encoded = coeffs.clone();
        let gain = 1.0f32;
        let n = coeffs.len();
        let k = 5;
        let spread = SPREAD_NORMAL;
        let b = 2usize;
        let arch = 0;

        let mut buffer = vec![0u8; 64];
        let mask;
        {
            let mut enc = EcEnc::new(&mut buffer);
            mask = alg_quant(&mut encoded, n, k, spread, b, &mut enc, gain, true, arch);
            enc.enc_done();
        }

        let mut decoded = coeffs.clone();
        let mut dec = EcDec::new(&mut buffer);
        let mask_dec = alg_unquant(&mut decoded, n, k, spread, b, &mut dec, gain);

        assert_eq!(mask, mask_dec);
        let tolerance = 1e-6;
        for (a, b) in encoded.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < tolerance);
        }
    }

    #[test]
    fn renormalise_vector_matches_expected_gain() {
        let mut data = seed_samples(8);
        let gain = 0.75f32;
        let len = data.len();
        renormalise_vector(&mut data, len, gain, 0);
        let energy: f32 = data.iter().map(|&v| v * v).sum();
        assert!((energy - gain * gain).abs() < 1e-5);
    }

    #[test]
    fn stereo_itheta_returns_expected_half_pi_value() {
        let x = [0.0f32; 8];
        let y = [1.0f32; 8];
        let angle = stereo_itheta(&x, &y, false, x.len(), 0);
        assert!((angle - 16_384).abs() <= 1);
    }

    #[cfg(feature = "fixed_point")]
    #[test]
    fn fixed_residual_normalisation_preserves_signs() {
        let pulses = [3, -2, 1, 0];
        let mut output = [0i32; 4];
        let ryy = 1 << 14;
        let gain = 1 << 14;

        normalise_residual_fixed(&pulses, &mut output, pulses.len(), ryy, gain);

        for (value, &pulse) in output.iter().zip(&pulses) {
            if pulse == 0 {
                assert_eq!(*value, 0);
            } else {
                let sign = i32::from(value.signum());
                assert!(sign == 0 || sign == pulse.signum());
            }
        }
    }

    #[cfg(feature = "fixed_point")]
    mod fixed_tests {
        use crate::celt::entdec::EcDec;
        use crate::celt::entenc::EcEnc;
        use crate::celt::vq::{
            alg_quant_fixed, alg_unquant_fixed, exp_rotation_fixed, exp_rotation1_fixed,
            normalise_residual_fixed, op_pvq_search_fixed, renormalise_vector_fixed,
        };
        use alloc::vec;
        use alloc::vec::Vec;

        #[test]
        fn fixed_vq_matches_pinned_q24_reference() {
            let script = include_str!("../../tests/fixtures/reference/fixed-vq.script");
            let expected = include_str!("../../tests/fixtures/reference/fixed-vq.txt");
            assert_eq!(script.lines().count(), expected.lines().count());
            let mut cases = 0;
            for (source, expected) in script.lines().zip(expected.lines()) {
                let mut source = source.split_whitespace();
                let operation = source.next().unwrap();
                let name = source.next().unwrap();
                let args: Vec<i32> = source.map(|v| v.parse().unwrap()).collect();
                let (n, a, b, c, d, e) = (
                    args[0] as usize,
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                );
                let mut x = args[6..].to_vec();
                assert_eq!(x.len(), n);
                let mut y = vec![0i32; n];
                let mut packet = [0u8; 256];
                let (mut scalar1, mut scalar2, mut range) = (0i32, 0i32, 0u32);
                match operation {
                    "normalise" | "normrot" => {
                        let pulses = x.clone();
                        normalise_residual_fixed(&pulses, &mut x, n, a, b);
                        if operation == "normrot" {
                            exp_rotation_fixed(&mut x, n, c, d as usize, e, 2);
                        }
                    }
                    "rotate" => exp_rotation_fixed(&mut x, n, a, b as usize, c, d),
                    "rotate1" => exp_rotation1_fixed(&mut x, a as usize, b as i16, c as i16),
                    "renorm" => renormalise_vector_fixed(&mut x, n, a, 0),
                    "search" => {
                        scalar1 = op_pvq_search_fixed(&mut x, &mut y, n, a, 0) as i32;
                    }
                    "alg" => {
                        let mut enc = EcEnc::new(&mut packet);
                        scalar1 =
                            alg_quant_fixed(&mut x, n, a, b, c as usize, &mut enc, d, e != 0, 0)
                                as i32;
                        range = enc.ctx().rng;
                        enc.enc_done();
                        assert_eq!(enc.ctx().error, 0);
                        let mut dec = EcDec::new(&packet);
                        scalar2 =
                            alg_unquant_fixed(&mut y, n, a, b, c as usize, &mut dec, d) as i32;
                        assert_eq!(dec.ctx().rng, range);
                        assert_eq!(dec.ctx().error, 0);
                    }
                    _ => panic!("unknown reference operation {operation}"),
                }
                let mut expected = expected.split_whitespace();
                assert_eq!(expected.next().unwrap(), name);
                assert_eq!(
                    scalar1,
                    expected.next().unwrap().parse::<i32>().unwrap(),
                    "{name}: first scalar"
                );
                assert_eq!(
                    scalar2,
                    expected.next().unwrap().parse::<i32>().unwrap(),
                    "{name}: second scalar"
                );
                assert_eq!(
                    range,
                    expected.next().unwrap().parse::<u32>().unwrap(),
                    "{name}: range"
                );
                for (i, value) in x.iter().chain(y.iter()).enumerate() {
                    assert_eq!(
                        *value,
                        expected.next().unwrap().parse::<i32>().unwrap(),
                        "{name}: vector index {i}"
                    );
                }
                let hex = expected.next().unwrap();
                if operation == "alg" {
                    for (i, &byte) in packet.iter().enumerate() {
                        assert_eq!(
                            byte,
                            u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap(),
                            "{name}: packet byte {i}"
                        );
                    }
                } else {
                    assert_eq!(hex, "-");
                }
                assert!(expected.next().is_none());
                cases += 1;
            }
            assert_eq!(cases, 26);
        }
    }
}
