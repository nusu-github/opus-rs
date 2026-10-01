//! Scalable pulse-vector refinement and cubic quantisation from `celt/vq.c`.

use super::cwrs::{decode_pulses, encode_pulses};
use super::entcode::ec_tell;
use super::entdec::EcDec;
use super::entenc::EcEnc;
#[cfg(feature = "fixed_point")]
use super::fixed_ops::{mult16_32_q15, mult32_32_q31, pshr32, vshr32};
#[cfg(feature = "fixed_point")]
use super::math::celt_ilog2;
#[cfg(feature = "fixed_point")]
use super::math_fixed::{celt_rcp_norm32, celt_rsqrt_norm32};
#[cfg(not(feature = "fixed_point"))]
use super::types::OpusVal16;
use super::vq;
use alloc::vec;

#[cfg(feature = "fixed_point")]
pub(crate) type Scalar = i32;
#[cfg(not(feature = "fixed_point"))]
pub(crate) type Scalar = f32;

fn energy(iy: &[i32], shift: i32) -> Scalar {
    #[cfg(feature = "fixed_point")]
    {
        let sum = iy
            .iter()
            .fold(0i64, |s, &v| s + i64::from(v) * i64::from(v));
        ((sum + ((1i64 << (2 * shift)) >> 1)) >> (2 * shift)) as i32
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let _ = shift;
        iy.iter().fold(0f32, |s, &v| s + v as f32 * v as f32)
    }
}

fn normalise(iy: &[i32], x: &mut [Scalar], yy: Scalar, gain: Scalar, shift: i32) {
    #[cfg(not(feature = "fixed_point"))]
    {
        let _ = shift;
        vq::normalise_residual(iy, x, iy.len(), yy, gain);
    }
    #[cfg(feature = "fixed_point")]
    {
        if shift == 0 {
            vq::normalise_residual_fixed(iy, x, iy.len(), yy, gain);
        } else {
            let k = celt_ilog2(yy) >> 1;
            let t = vshr32(yy, 2 * (k - 7) - 15);
            let g = mult32_32_q31(celt_rsqrt_norm32(t), gain);
            let total_shift = 25 - k - shift;
            for (out, &pulse) in x.iter_mut().zip(iy) {
                let scaled = if total_shift >= 0 {
                    pulse.wrapping_shl(total_shift as u32)
                } else {
                    pshr32(pulse, (-total_shift) as u32)
                };
                *out = mult32_32_q31(g, scaled);
            }
        }
    }
}

fn search_pair(
    x: &[Scalar],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    shift: i32,
) -> (Scalar, i32) {
    let sum = x[0].abs() + x[1].abs();
    #[cfg(feature = "fixed_point")]
    let tiny = sum < 1;
    #[cfg(not(feature = "fixed_point"))]
    let tiny = sum < 1e-15;
    if tiny {
        iy[0] = k;
        up_iy[0] = up * k;
        iy[1] = 0;
        up_iy[1] = 0;
        #[cfg(feature = "fixed_point")]
        let yy =
            (i64::from(k) * i64::from(k) * i64::from(up) * i64::from(up) >> (2 * shift)) as i32;
        #[cfg(not(feature = "fixed_point"))]
        let yy = k as f32 * k as f32 * up as f32 * up as f32;
        return (yy, 0);
    }
    #[cfg(feature = "fixed_point")]
    {
        let sum_shift = 30 - celt_ilog2(sum);
        let rcp = celt_rcp_norm32(sum.wrapping_shl(sum_shift as u32));
        let x0 = mult32_32_q31(x[0].wrapping_shl(sum_shift as u32), rcp);
        iy[0] = pshr32(mult32_32_q31(k << 8, x0), 7);
        up_iy[0] = pshr32(mult32_32_q31((up * k) << 8, x0), 7);
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let rcp = 1.0 / sum;
        iy[0] = libm::floorf(0.5 + k as f32 * x[0] * rcp) as i32;
        up_iy[0] = libm::floorf(0.5 + (up * k) as f32 * x[0] * rcp) as i32;
    }
    up_iy[0] = up_iy[0].clamp(up * iy[0] - (up - 1) / 2, up * iy[0] + (up - 1) / 2);
    let mut offset = up_iy[0] - up * iy[0];
    iy[1] = k - iy[0].abs();
    up_iy[1] = up * k - up_iy[0].abs();
    if x[1] < 0 as Scalar {
        iy[1] = -iy[1];
        up_iy[1] = -up_iy[1];
        offset = -offset;
    }
    (energy(up_iy, shift), offset)
}

fn refine(
    xn: &[Scalar],
    iy: &mut [i32],
    iy0: Option<&[i32]>,
    k: i32,
    up: i32,
    margin: i32,
) -> bool {
    let mut rounding = vec![0 as Scalar; xn.len()];
    for i in 0..xn.len() {
        #[cfg(feature = "fixed_point")]
        let tmp = mult32_32_q31(k << 8, xn[i]);
        #[cfg(not(feature = "fixed_point"))]
        let tmp = k as f32 * xn[i];
        #[cfg(feature = "fixed_point")]
        {
            iy[i] = (tmp + 64) >> 7;
            rounding[i] = tmp - (iy[i] << 7);
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            iy[i] = libm::floor(0.5 + f64::from(tmp)) as i32;
            rounding[i] = tmp - iy[i] as f32;
        }
    }
    if let Some(base) = iy0 {
        for (out, &value) in iy.iter_mut().zip(base) {
            *out = (*out).clamp(up * value - up + 1, up * value + up - 1);
        }
    }
    let mut sum: i32 = iy.iter().sum();
    if (sum - k).abs() > 32 {
        return false;
    }
    let dir = if sum < k { 1 } else { -1 };
    while sum != k {
        let mut best = (-1000000 * dir) as Scalar;
        let mut pos = 0;
        for i in 0..xn.len() {
            let base = iy0.map_or(iy[i], |v| v[i]);
            if (rounding[i] - best) * dir as Scalar > 0 as Scalar
                && (iy[i] - up * base).abs() < margin - 1
                && !(dir == -1 && iy[i] == 0)
            {
                best = rounding[i];
                pos = i;
            }
        }
        iy[pos] += dir;
        #[cfg(feature = "fixed_point")]
        {
            rounding[pos] -= dir << 15;
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            rounding[pos] -= dir as f32;
        }
        sum += dir;
    }
    true
}

fn search_extra(
    x: &[Scalar],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    residual: &mut [i32],
    shift: i32,
) -> Scalar {
    let mut sum = 0 as Scalar;
    for &value in x {
        sum += value.abs();
    }
    #[cfg(feature = "fixed_point")]
    let tiny = sum < 1;
    #[cfg(not(feature = "fixed_point"))]
    let tiny = sum < 1e-15;
    let mut xn = vec![0 as Scalar; x.len()];
    if !tiny {
        #[cfg(feature = "fixed_point")]
        {
            let sum_shift = 30 - celt_ilog2(sum);
            let rcp = celt_rcp_norm32(sum.wrapping_shl(sum_shift as u32));
            for (out, &v) in xn.iter_mut().zip(x) {
                *out = mult32_32_q31(v.abs().wrapping_shl(sum_shift as u32), rcp);
            }
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            let rcp = super::math::celt_rcp(sum);
            for (out, &v) in xn.iter_mut().zip(x) {
                *out = v.abs() * rcp;
            }
        }
    }
    if tiny || !refine(&xn, iy, None, k, 1, k + 1) || !refine(&xn, up_iy, Some(iy), up * k, up, up)
    {
        iy.fill(0);
        up_iy.fill(0);
        iy[0] = k;
        up_iy[0] = up * k;
    }
    let yy = energy(up_iy, shift);
    for i in 0..x.len() {
        if x[i] < 0 as Scalar {
            iy[i] = -iy[i];
            up_iy[i] = -up_iy[i];
        }
        residual[i] = up_iy[i] - up * iy[i];
    }
    yy
}

fn encode_refine(enc: &mut EcEnc<'_>, refine: i32, up: i32, bits: u32, entropy: bool) {
    let large = refine.abs() > up / 2 && refine != up / 2 + 1;
    enc.enc_bit_logp(i32::from(large), if entropy { 3 } else { 1 });
    if large {
        enc.enc_bits(u32::from(refine < 0), 1);
        enc.enc_bits((refine.abs() - up / 2 - 1) as u32, bits - 1);
    } else {
        enc.enc_bits((refine + up / 2) as u32, bits);
    }
}
fn decode_refine(dec: &mut EcDec<'_>, up: i32, bits: u32, entropy: bool) -> i32 {
    if dec.dec_bit_logp(if entropy { 3 } else { 1 }) != 0 {
        let sign = dec.dec_bits(1);
        let r = dec.dec_bits(bits - 1) as i32 + up / 2 + 1;
        if sign != 0 { -r } else { r }
    } else {
        dec.dec_bits(bits) as i32 - up / 2
    }
}

pub(crate) fn alg_quant(
    x: &mut [Scalar],
    k: i32,
    spread: i32,
    blocks: usize,
    enc: &mut EcEnc<'_>,
    gain: Scalar,
    resynth: bool,
    ext: &mut EcEnc<'_>,
    extra_bits: i32,
    arch: i32,
) -> u32 {
    let n = x.len();
    if extra_bits < 2 {
        #[cfg(feature = "fixed_point")]
        return vq::alg_quant_fixed(x, n, k, spread, blocks, enc, gain, resynth, arch);
        #[cfg(not(feature = "fixed_point"))]
        return vq::alg_quant(x, n, k, spread, blocks, enc, gain, resynth, arch);
    }
    #[cfg(feature = "fixed_point")]
    vq::exp_rotation_fixed(x, n, 1, blocks, k, spread);
    #[cfg(not(feature = "fixed_point"))]
    vq::exp_rotation(x, n, 1, blocks, k, spread);
    let up = (1 << extra_bits) - 1;
    let shift = (extra_bits - 7).max(0);
    let mut iy = vec![0; n];
    let mut up_iy = vec![0; n];
    let yy;
    if n == 2 {
        let (energy, residual) = search_pair(x, &mut iy, &mut up_iy, k, up, shift);
        yy = energy;
        encode_pulses(&iy, n, k as usize, enc);
        ext.enc_uint((residual + (up - 1) / 2) as u32, up as u32);
    } else {
        let mut residual = vec![0; n];
        yy = search_extra(x, &mut iy, &mut up_iy, k, up, &mut residual, shift);
        encode_pulses(&iy, n, k as usize, enc);
        let entropy = ext
            .ctx()
            .storage
            .wrapping_mul(8)
            .wrapping_sub(ec_tell(ext.ctx()) as u32)
            > (n as u32 - 1) * (extra_bits as u32 + 3) + 1;
        for &r in &residual[..n - 1] {
            encode_refine(ext, r, up, extra_bits as u32, entropy);
        }
        if iy[n - 1] == 0 {
            ext.enc_bits(u32::from(up_iy[n - 1] < 0), 1);
        }
    }
    let mask = vq::extract_collapse_mask(&up_iy, n, blocks);
    if resynth {
        normalise(&up_iy, x, yy, gain, shift);
        #[cfg(feature = "fixed_point")]
        vq::exp_rotation_fixed(x, n, -1, blocks, k, spread);
        #[cfg(not(feature = "fixed_point"))]
        vq::exp_rotation(x, n, -1, blocks, k, spread);
    }
    mask
}

pub(crate) fn alg_unquant(
    x: &mut [Scalar],
    k: i32,
    spread: i32,
    blocks: usize,
    dec: &mut EcDec<'_>,
    gain: Scalar,
    ext: &mut EcDec<'_>,
    extra_bits: i32,
) -> u32 {
    let n = x.len();
    if extra_bits < 2 {
        #[cfg(feature = "fixed_point")]
        return vq::alg_unquant_fixed(x, n, k, spread, blocks, dec, gain);
        #[cfg(not(feature = "fixed_point"))]
        return vq::alg_unquant(x, n, k, spread, blocks, dec, gain);
    }
    let mut iy = vec![0; n];
    let _ = decode_pulses(&mut iy, n, k as usize, dec);
    let up = (1 << extra_bits) - 1;
    let shift = (extra_bits - 7).max(0);
    if n == 2 {
        let refine = ext.dec_uint(up as u32) as i32 - (up - 1) / 2;
        iy[0] *= up;
        iy[1] *= up;
        if iy[1] == 0 {
            iy[1] = if iy[0] > 0 { -refine } else { refine };
            iy[0] += if i64::from(refine) * i64::from(iy[0]) > 0 {
                -refine
            } else {
                refine
            };
        } else if iy[1] > 0 {
            iy[0] += refine;
            iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
        } else {
            iy[0] -= refine;
            iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
        }
    } else {
        let entropy = ext
            .ctx()
            .storage
            .wrapping_mul(8)
            .wrapping_sub(ec_tell(ext.ctx()) as u32)
            > (n as u32 - 1) * (extra_bits as u32 + 3) + 1;
        let mut residual = vec![0; n - 1];
        for r in &mut residual {
            *r = decode_refine(ext, up, extra_bits as u32, entropy);
        }
        let sign = if iy[n - 1] == 0 {
            ext.dec_bits(1) != 0
        } else {
            iy[n - 1] < 0
        };
        for i in 0..n - 1 {
            iy[i] = iy[i] * up + residual[i];
        }
        iy[n - 1] = up * k;
        for i in 0..n - 1 {
            iy[n - 1] -= iy[i].abs();
        }
        if sign {
            iy[n - 1] = -iy[n - 1];
        }
    }
    normalise(&iy, x, energy(&iy, shift), gain, shift);
    #[cfg(feature = "fixed_point")]
    vq::exp_rotation_fixed(x, n, -1, blocks, k, spread);
    #[cfg(not(feature = "fixed_point"))]
    vq::exp_rotation(x, n, -1, blocks, k, spread);
    vq::extract_collapse_mask(&iy, n, blocks)
}

fn cubic_synthesis(x: &mut [Scalar], iy: &[i32], k: i32, face: usize, sign: bool, gain: Scalar) {
    for (out, &v) in x.iter_mut().zip(iy) {
        *out = (1 + 2 * v - k) as Scalar;
    }
    x[face] = if sign { -k } else { k } as Scalar;
    #[cfg(feature = "fixed_point")]
    {
        let shift = (celt_ilog2(k) + celt_ilog2(x.len() as i32) / 2 - 13).max(0);
        let sum = x.iter().fold(0i32, |sum, &v| {
            sum.wrapping_add(pshr32(
                (v as i16 as i32) * (v as i16 as i32),
                (2 * shift) as u32,
            ))
        });
        let sum_shift = (29 - celt_ilog2(sum)) >> 1;
        let mag = celt_rsqrt_norm32(sum.wrapping_shl((2 * sum_shift + 1) as u32));
        for v in x {
            *v = vshr32(
                mult16_32_q15(*v as i16, mult32_32_q31(mag, gain)),
                shift - sum_shift + 5,
            );
        }
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let sum = x.iter().fold(0.0f32, |s, &v| s + v * v);
        let mag = (1.0 / libm::sqrt(f64::from(sum))) as f32;
        for v in x {
            *v *= mag * gain;
        }
    }
}

pub(crate) fn cubic_quant(
    x: &mut [Scalar],
    res: i32,
    blocks: usize,
    enc: &mut EcEnc<'_>,
    gain: Scalar,
    resynth: bool,
) -> u32 {
    let mut k = 1 << res;
    if blocks != 1 {
        k = (k - 1).max(1);
    }
    if k == 1 {
        if resynth {
            x.fill(0 as Scalar);
        }
        return 0;
    }
    let mut iy = vec![0; x.len()];
    let mut face = 0;
    let mut faceval = -1 as Scalar;
    for (i, &v) in x.iter().enumerate() {
        if v.abs() > faceval {
            faceval = v.abs();
            face = i;
        }
    }
    let sign = x[face] < 0 as Scalar;
    enc.enc_uint(face as u32, x.len() as u32);
    enc.enc_bits(u32::from(sign), 1);
    #[cfg(feature = "fixed_point")]
    if faceval != 0 {
        let shift = 30 - celt_ilog2(faceval);
        let norm = mult16_32_q15(
            k as i16,
            celt_rcp_norm32(faceval.wrapping_shl(shift as u32)),
        );
        for (out, &v) in iy.iter_mut().zip(x.iter()) {
            let scaled = vshr32(v + faceval, 1 - shift);
            *out = (k - 1).min(mult32_32_q31(scaled, norm) >> 15);
        }
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        let norm = 0.5 * k as f32 / (faceval + 1e-15);
        for (out, &v) in iy.iter_mut().zip(x.iter()) {
            *out = (k - 1).min(libm::floorf((v + faceval) * norm) as i32);
        }
    }
    for (i, &v) in iy.iter().enumerate() {
        if i != face {
            enc.enc_bits(v as u32, res as u32);
        }
    }
    if resynth {
        cubic_synthesis(x, &iy, k, face, sign, gain);
    }
    (1 << blocks) - 1
}

pub(crate) fn cubic_unquant(
    x: &mut [Scalar],
    res: i32,
    blocks: usize,
    dec: &mut EcDec<'_>,
    gain: Scalar,
) -> u32 {
    let mut k = 1 << res;
    if blocks != 1 {
        k = (k - 1).max(1);
    }
    if k == 1 {
        x.fill(0 as Scalar);
        return 0;
    }
    let face = dec.dec_uint(x.len() as u32) as usize;
    let sign = dec.dec_bits(1) != 0;
    let mut iy = vec![0; x.len()];
    for (i, v) in iy.iter_mut().enumerate() {
        if i != face {
            *v = dec.dec_bits(res as u32) as i32;
        }
    }
    cubic_synthesis(x, &iy, k, face, sign, gain);
    (1 << blocks) - 1
}

#[cfg(feature = "fixed_point")]
pub(crate) fn stereo_itheta(x: &[i32], y: &[i32], stereo: bool) -> i32 {
    let n = x.len();
    let (mut mid, mut side) = (0i32, 0i32);
    if stereo {
        for i in 0..n {
            let m = pshr32(x[i].wrapping_add(y[i]), 11) as i16;
            let s = pshr32(x[i].wrapping_sub(y[i]), 11) as i16;
            mid = super::fixed_ops::mac16_16(mid, m, m);
            side = super::fixed_ops::mac16_16(side, s, s);
        }
    } else {
        mid = vq::celt_inner_prod_norm_shift(&x[..n], &x[..n]);
        side = vq::celt_inner_prod_norm_shift(&y[..n], &y[..n]);
    }
    crate::celt::math_fixed::celt_atan2p_norm(
        crate::celt::math_fixed::celt_sqrt32(side),
        crate::celt::math_fixed::celt_sqrt32(mid),
    )
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn stereo_itheta(x: &[OpusVal16], y: &[OpusVal16], stereo: bool) -> i32 {
    let n = x.len();
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
        emid += super::pitch::celt_inner_prod(mid, mid);
        eside += super::pitch::celt_inner_prod(side, side);
    }

    let mid = super::math::celt_sqrt(emid);
    let side = super::math::celt_sqrt(eside);
    let angle = super::math::celt_atan2p_norm(side, mid);

    libm::floorf(0.5 + 65_536.0 * 16_384.0 * angle) as i32
}

#[cfg(not(feature = "fixed_point"))]
pub(crate) fn cos_norm2(mut x: f32) -> f32 {
    x = (f64::from(x) - 4.0 * libm::floor(0.25 * f64::from(x + 1.0))) as f32;
    let sign = if x > 1.0 { -1.0 } else { 1.0 };
    if x > 1.0 {
        x -= 2.0;
    }
    let x2 = x * x;
    sign * (0.999999940395355224609375
        + x2 * (-1.23369824886322021484375
            + x2 * (0.2536507546901702880859375
                + x2 * (-0.02081062830984592437744140625
                    + x2 * 0.0008581906440667808055877685546875))))
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use crate::celt::entcode::{ec_tell, ec_tell_frac};
    use alloc::vec::Vec;

    fn words(values: &[Scalar]) -> Vec<u32> {
        values
            .iter()
            .map(|&value| {
                #[cfg(feature = "fixed_point")]
                {
                    value as u32
                }
                #[cfg(not(feature = "fixed_point"))]
                {
                    value.to_bits()
                }
            })
            .collect()
    }

    fn unhex(text: &str) -> Vec<u8> {
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn qext_vectors_match_pinned_scalar_c() {
        let script = include_str!("../../tests/fixtures/reference/qext-vq.script");
        #[cfg(not(feature = "fixed_point"))]
        let expected = include_str!("../../tests/fixtures/reference/qext-vq.txt");
        #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
        let expected = include_str!("../../tests/fixtures/reference/fixed-qext-vq.txt");
        #[cfg(feature = "enable_res24")]
        let expected = include_str!("../../tests/fixtures/reference/fixed-res24-qext-vq.txt");
        let mut results = expected.lines();
        let mut count = 0;
        for line in script.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            let output: Vec<_> = results.next().unwrap().split_whitespace().collect();
            let number = |index: usize| fields[index].parse::<i32>().unwrap();
            let n = number(2) as usize;
            let k = number(3);
            let spread = number(4);
            let blocks = number(5) as usize;
            let gain_q15 = number(6);
            let resynth = number(7) != 0;
            let extra_bits = number(8);
            let base_cap = number(9) as usize;
            let ext_cap = number(10) as usize;
            assert_eq!(fields[1], output[0]);
            assert_eq!(fields.len(), 11 + n);
            assert_eq!(output.len(), 15 + 2 * n);
            let raw: Vec<i32> = fields[11..]
                .iter()
                .map(|value| value.parse().unwrap())
                .collect();
            #[cfg(not(feature = "fixed_point"))]
            let (mut x, gain) = (
                raw.iter()
                    .map(|&value| value as f32 * (1.0 / 16_777_216.0))
                    .collect::<Vec<_>>(),
                gain_q15 as f32 * (1.0 / 32768.0),
            );
            #[cfg(feature = "fixed_point")]
            let (mut x, gain) = (
                raw,
                if gain_q15 == 32768 {
                    i32::MAX
                } else {
                    gain_q15 * 65536
                },
            );
            let mut y = vec![Scalar::default(); n];
            let mut packet = vec![0u8; base_cap];
            let mut extension = vec![0u8; ext_cap];
            let mut enc = EcEnc::new(&mut packet);
            let mut ext = EcEnc::new(&mut extension);
            let mask = if fields[0] == "alg" {
                alg_quant(
                    &mut x, k, spread, blocks, &mut enc, gain, resynth, &mut ext, extra_bits, 0,
                )
            } else {
                cubic_quant(&mut x, k, blocks, &mut enc, gain, resynth)
            };
            let header = [
                mask,
                enc.ctx().rng,
                ext.ctx().rng,
                ec_tell(enc.ctx()) as u32,
                ec_tell(ext.ctx()) as u32,
                ec_tell_frac(enc.ctx()),
                ec_tell_frac(ext.ctx()),
            ];
            for (&value, column) in header.iter().zip([1, 3, 4, 5, 6, 7, 8]) {
                assert_eq!(
                    value,
                    output[column].parse::<u32>().unwrap(),
                    "{} encode field {}",
                    fields[1],
                    column
                );
            }
            enc.enc_done();
            ext.enc_done();
            assert_eq!(enc.ctx().error, 0, "{} base coder error", fields[1]);
            assert_eq!(ext.ctx().error, 0, "{} extension coder error", fields[1]);
            drop(enc);
            drop(ext);
            assert_eq!(
                packet,
                unhex(output[13 + 2 * n]),
                "{} base packet",
                fields[1]
            );
            assert_eq!(
                extension,
                unhex(output[14 + 2 * n]),
                "{} extension packet",
                fields[1]
            );
            let mut dec = EcDec::new(&packet);
            let mut ext = EcDec::new(&extension);
            let mask = if fields[0] == "alg" {
                alg_unquant(
                    &mut y, k, spread, blocks, &mut dec, gain, &mut ext, extra_bits,
                )
            } else {
                cubic_unquant(&mut y, k, blocks, &mut dec, gain)
            };
            let header = [
                mask,
                dec.ctx().rng,
                ext.ctx().rng,
                ec_tell(dec.ctx()) as u32,
                ec_tell(ext.ctx()) as u32,
            ];
            for (&value, column) in header.iter().zip([2, 9, 10, 11, 12]) {
                assert_eq!(
                    value,
                    output[column].parse::<u32>().unwrap(),
                    "{} decode field {}",
                    fields[1],
                    column
                );
            }
            assert_eq!(dec.ctx().error, 0, "{} base decoder error", fields[1]);
            assert_eq!(ext.ctx().error, 0, "{} extension decoder error", fields[1]);
            let x_expected: Vec<_> = output[13..13 + n]
                .iter()
                .map(|value| u32::from_str_radix(value, 16).unwrap())
                .collect();
            let y_expected: Vec<_> = output[13 + n..13 + 2 * n]
                .iter()
                .map(|value| u32::from_str_radix(value, 16).unwrap())
                .collect();
            assert_eq!(words(&x), x_expected, "{} encoder coefficients", fields[1]);
            assert_eq!(words(&y), y_expected, "{} decoder coefficients", fields[1]);
            count += 1;
        }
        assert!(results.next().is_none());
        assert_eq!(count, 1888);
    }
}
