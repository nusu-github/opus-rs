// Copyright (c) 2026 Xiph.Org Foundation. BSD-3-Clause; see LICENSE.
//! Safe scalar translation of the experimental Opus prime-factor FFT.

#[cfg(feature = "fixed_point")]
use super::FixedKissFftCpx;
use super::KissFftCpx;
#[cfg(feature = "fixed_point")]
use super::fixed_ops::{mult16_32_q15, pshr32};
use alloc::vec;
use alloc::vec::Vec;

#[path = "pfa_tables.rs"]
mod tables;

#[derive(Clone, Copy)]
struct Coefficient(f32, i16, i32);
const COS_5: Coefficient = Coefficient(0.309017003, 10126, 663608942);
const COS_10: Coefficient = Coefficient(0.809017003, 26510, 1737350766);
const SIN_5: Coefficient = Coefficient(0.95105654, 31164, 2042378317);
const SIN_10: Coefficient = Coefficient(0.587785244, 19261, 1262259218);
const SIN_3: Coefficient = Coefficient(0.866025388, 28378, 1859775393);
const COS: [Coefficient; 9] = [
    Coefficient(1.0, 32767, 2147483647),
    Coefficient(0.980785251, 32138, 2106220289),
    Coefficient(0.923879504, 30274, 1984016128),
    Coefficient(0.831469595, 27246, 1785567359),
    Coefficient(0.707106769, 23170, 1518500224),
    Coefficient(0.555570245, 18205, 1193078016),
    Coefficient(0.382683426, 12540, 821806400),
    Coefficient(0.195090324, 6393, 418953281),
    Coefficient(0.0, 0, 0),
];

#[cfg(feature = "enable_qext")]
const COS_64: [Coefficient; 17] = [
    Coefficient(1.0, 32767, 2147483647),
    Coefficient(0.99518472, 0, 2137142913),
    Coefficient(0.980785251, 0, 2106220289),
    Coefficient(0.956940353, 0, 2055013760),
    Coefficient(0.923879504, 0, 1984016128),
    Coefficient(0.881921291, 0, 1893911551),
    Coefficient(0.831469595, 0, 1785567359),
    Coefficient(0.773010433, 0, 1660027265),
    Coefficient(0.707106769, 0, 1518500224),
    Coefficient(0.634393275, 0, 1362349184),
    Coefficient(0.555570245, 0, 1193078016),
    Coefficient(0.471396744, 0, 1012316799),
    Coefficient(0.382683426, 0, 821806400),
    Coefficient(0.290284663, 0, 623381567),
    Coefficient(0.195090324, 0, 418953281),
    Coefficient(0.0980171412, 0, 210490208),
    Coefficient(0.0, 0, 0),
];

trait Scalar: Copy + Default {
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn half(self) -> Self;
    fn mul(self, coefficient: Coefficient) -> Self;
    fn downshift(self, shift: i32) -> Self;
}
impl Scalar for f32 {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn neg(self) -> Self {
        -self
    }
    fn half(self) -> Self {
        self * 0.5
    }
    fn mul(self, coefficient: Coefficient) -> Self {
        self * coefficient.0
    }
    fn downshift(self, _: i32) -> Self {
        self
    }
}
#[cfg(feature = "fixed_point")]
impl Scalar for i32 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
    fn half(self) -> Self {
        self >> 1
    }
    fn mul(self, coefficient: Coefficient) -> Self {
        #[cfg(feature = "enable_qext")]
        {
            super::fixed_ops::mult32_32_p31(coefficient.2, self)
        }
        #[cfg(not(feature = "enable_qext"))]
        {
            mult16_32_q15(coefficient.1, self)
        }
    }
    fn downshift(self, shift: i32) -> Self {
        if shift == 1 {
            self >> 1
        } else if shift > 0 {
            pshr32(self, shift as u32)
        } else {
            self
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Complex<T> {
    r: T,
    i: T,
}
impl<T> Complex<T> {
    fn new(r: T, i: T) -> Self {
        Self { r, i }
    }
}

fn downshift<T: Scalar>(values: &mut [Complex<T>], remaining: &mut i32, step: i32) {
    let shift = step.min(*remaining);
    *remaining -= shift;
    for value in values {
        value.r = value.r.downshift(shift);
        value.i = value.i.downshift(shift);
    }
}

fn butterfly<T: Scalar>(
    out: &mut [Complex<T>],
    k: usize,
    quarter: usize,
    t1: T,
    t2: T,
    t5: T,
    t6: T,
) {
    let a0 = out[k];
    let a1 = out[k + quarter];
    let t3 = t5.sub(t1);
    let t5 = t5.add(t1);
    out[k + 2 * quarter].r = a0.r.sub(t5);
    out[k].r = a0.r.add(t5);
    out[k + 3 * quarter].i = a1.i.sub(t3);
    out[k + quarter].i = a1.i.add(t3);
    let t4 = t2.sub(t6);
    let t6 = t2.add(t6);
    out[k + 3 * quarter].r = a1.r.sub(t4);
    out[k + quarter].r = a1.r.add(t4);
    out[k + 2 * quarter].i = a0.i.sub(t6);
    out[k].i = a0.i.add(t6);
}

fn split_radix<T: Scalar>(src: &[Complex<T>]) -> Vec<Complex<T>> {
    let n = src.len();
    let mut out = vec![Complex::default(); n];
    if n == 2 {
        out[0] = Complex::new(src[0].r.add(src[1].r), src[0].i.add(src[1].i));
        out[1] = Complex::new(src[0].r.sub(src[1].r), src[0].i.sub(src[1].i));
    } else if n == 4 {
        let t3 = src[0].r.sub(src[1].r);
        let t1 = src[0].r.add(src[1].r);
        let t8 = src[3].r.sub(src[2].r);
        let t6 = src[3].r.add(src[2].r);
        out[2].r = t1.sub(t6);
        out[0].r = t1.add(t6);
        let t4 = src[0].i.sub(src[1].i);
        let t2 = src[0].i.add(src[1].i);
        let t7 = src[2].i.sub(src[3].i);
        let t5 = src[2].i.add(src[3].i);
        out[3].i = t4.sub(t8);
        out[1].i = t4.add(t8);
        out[3].r = t3.sub(t7);
        out[1].r = t3.add(t7);
        out[2].i = t2.sub(t5);
        out[0].i = t2.add(t5);
    } else {
        let quarter = n / 4;
        out[..n / 2].copy_from_slice(&split_radix(&src[..n / 2]));
        out[n / 2..3 * quarter].copy_from_slice(&split_radix(&src[n / 2..3 * quarter]));
        out[3 * quarter..].copy_from_slice(&split_radix(&src[3 * quarter..]));
        for k in 0..quarter {
            let a2 = out[k + 2 * quarter];
            let a3 = out[k + 3 * quarter];
            if k == 0 && n < 32 {
                butterfly(&mut out, k, quarter, a2.r, a2.i, a3.r, a3.i);
            } else {
                let (wr, wi) = {
                    #[cfg(feature = "enable_qext")]
                    if n == 64 {
                        (COS_64[k], COS_64[16 - k])
                    } else {
                        (COS[k * 32 / n], COS[8 - k * 32 / n])
                    }
                    #[cfg(not(feature = "enable_qext"))]
                    {
                        (COS[k * 32 / n], COS[8 - k * 32 / n])
                    }
                };
                let t1 = a2.r.mul(wr).add(a2.i.mul(wi));
                let t2 = a2.i.mul(wr).sub(a2.r.mul(wi));
                let t5 = a3.r.mul(wr).sub(a3.i.mul(wi));
                let t6 = a3.r.mul(wi).add(a3.i.mul(wr));
                butterfly(&mut out, k, quarter, t1, t2, t5, t6);
            }
        }
    }
    out
}

fn fft3<T: Scalar>(a: Complex<T>, b: Complex<T>, c: Complex<T>) -> [Complex<T>; 3] {
    let rs = b.r.add(c.r);
    let rd = b.r.sub(c.r);
    let is = b.i.add(c.i);
    let id = b.i.sub(c.i);
    let tr = id.mul(SIN_3);
    let ti = rd.mul(SIN_3);
    let br = a.r.sub(rs.half());
    let bi = a.i.sub(is.half());
    [
        Complex::new(a.r.add(rs), a.i.add(is)),
        Complex::new(br.add(tr), bi.sub(ti)),
        Complex::new(br.sub(tr), bi.add(ti)),
    ]
}

fn fft5<T: Scalar>(
    input: &[Complex<T>],
    output: &mut [Complex<T>],
    indices: [usize; 5],
    stride: usize,
) {
    let dc = input[0];
    let rd14 = input[1].r.sub(input[4].r);
    let rs14 = input[1].r.add(input[4].r);
    let id14 = input[1].i.sub(input[4].i);
    let is14 = input[1].i.add(input[4].i);
    let rd23 = input[2].r.sub(input[3].r);
    let rs23 = input[2].r.add(input[3].r);
    let id23 = input[2].i.sub(input[3].i);
    let is23 = input[2].i.add(input[3].i);
    output[indices[0] * stride] = Complex::new(dc.r.add(rs14.add(rs23)), dc.i.add(is14.add(is23)));
    let rt4 = rs14.mul(COS_5).sub(rs23.mul(COS_10));
    let rt0 = rs23.mul(COS_5).sub(rs14.mul(COS_10));
    let it4 = is14.mul(COS_5).sub(is23.mul(COS_10));
    let it0 = is23.mul(COS_5).sub(is14.mul(COS_10));
    let rt5 = id14.mul(SIN_5).add(id23.mul(SIN_10));
    let rt1 = id14.mul(SIN_10).sub(id23.mul(SIN_5));
    let it5 = rd14.mul(SIN_5).add(rd23.mul(SIN_10)).neg();
    let it1 = rd23.mul(SIN_5).sub(rd14.mul(SIN_10));
    let br4 = dc.r.add(rt4);
    let bi4 = dc.i.add(it4);
    let br0 = dc.r.add(rt0);
    let bi0 = dc.i.add(it0);
    output[indices[1] * stride] = Complex::new(br4.add(rt5), bi4.add(it5));
    output[indices[2] * stride] = Complex::new(br0.add(rt1), bi0.add(it1));
    output[indices[3] * stride] = Complex::new(br0.sub(rt1), bi0.sub(it1));
    output[indices[4] * stride] = Complex::new(br4.sub(rt5), bi4.sub(it5));
}

fn fft15<T: Scalar>(input: &[Complex<T>], output: &mut [Complex<T>], stride: usize) {
    let mut temp = [Complex::default(); 15];
    for (column, indices) in [[2, 0, 1], [13, 5, 9], [11, 3, 7], [14, 6, 10], [12, 4, 8]]
        .into_iter()
        .enumerate()
    {
        let values = fft3(input[indices[0]], input[indices[1]], input[indices[2]]);
        for row in 0..3 {
            temp[column + row * 5] = values[row];
        }
    }
    fft5(&temp, output, [0, 3, 6, 9, 12], stride);
    fft5(&temp[5..], output, [5, 8, 11, 14, 2], stride);
    fft5(&temp[10..], output, [10, 13, 1, 4, 7], stride);
}

pub(super) fn supported(n: usize) -> bool {
    matches!(n, 60 | 120 | 240 | 480) || (cfg!(feature = "enable_qext") && n == 960)
}

fn transform<T: Scalar>(input: &[Complex<T>], mut shift: i32, inverse: bool) -> Vec<Complex<T>> {
    use tables::*;
    let n = input.len();
    let m = n / 15;
    let (input_map, output_map, perm): (&[usize], &[usize], &[usize]) = match n {
        60 => (&MDCT_60, &PFA_60, &[0, 2, 1, 3]),
        120 => (&MDCT_120, &PFA_120, &[0, 4, 2, 6, 1, 5, 7, 3]),
        240 => (
            &MDCT_240,
            &PFA_240,
            &[0, 8, 4, 12, 2, 10, 14, 6, 1, 9, 5, 13, 15, 7, 3, 11],
        ),
        480 => (
            &MDCT_480,
            &PFA_480,
            &[
                0, 16, 8, 24, 4, 20, 28, 12, 2, 18, 10, 26, 30, 14, 6, 22, 1, 17, 9, 25, 5, 21, 29,
                13, 31, 15, 7, 23, 3, 19, 27, 11,
            ],
        ),
        #[cfg(feature = "enable_qext")]
        960 => (
            &MDCT_960,
            &PFA_960,
            &[
                0, 32, 16, 48, 8, 40, 56, 24, 4, 36, 20, 52, 60, 28, 12, 44, 2, 34, 18, 50, 10, 42,
                58, 26, 62, 30, 14, 46, 6, 38, 54, 22, 1, 33, 17, 49, 9, 41, 57, 25, 5, 37, 21, 53,
                61, 29, 13, 45, 63, 31, 15, 47, 7, 39, 55, 23, 3, 35, 19, 51, 59, 27, 11, 43,
            ],
        ),
        _ => unreachable!("unsupported PFA transform length"),
    };
    let mut staged = vec![Complex::default(); n];
    for i in 0..n {
        staged[input_map[i]] = input[i];
    }
    downshift(&mut staged, &mut shift, 3);
    let mut temp = vec![Complex::default(); n];
    for i in 0..m {
        fft15(&staged[15 * perm[i]..], &mut temp[i..], m);
    }
    downshift(&mut temp, &mut shift, 2);
    for row in temp.chunks_exact_mut(m) {
        let mut remaining = shift;
        downshift(row, &mut remaining, m.ilog2() as i32);
        let mut out = split_radix(row);
        let tail = remaining;
        downshift(&mut out, &mut remaining, tail);
        row.copy_from_slice(&out);
    }
    let mut output: Vec<_> = output_map.iter().map(|&index| temp[index]).collect();
    if inverse {
        output[1..].reverse();
    }
    output
}

pub(super) fn float(input: &[KissFftCpx], output: &mut [KissFftCpx], inverse: bool) -> bool {
    if !supported(input.len()) {
        return false;
    }
    let staged: Vec<_> = input.iter().map(|x| Complex::new(x.r, x.i)).collect();
    for (out, value) in output.iter_mut().zip(transform(&staged, 0, inverse)) {
        *out = KissFftCpx::new(value.r, value.i);
    }
    true
}

#[cfg(feature = "fixed_point")]
pub(super) fn fixed(
    input: &[FixedKissFftCpx],
    output: &mut [FixedKissFftCpx],
    shift: i32,
    inverse: bool,
) -> bool {
    if !supported(input.len()) {
        return false;
    }
    let staged: Vec<_> = input.iter().map(|x| Complex::new(x.r, x.i)).collect();
    for (out, value) in output.iter_mut().zip(transform(&staged, shift, inverse)) {
        *out = FixedKissFftCpx::new(value.r, value.i);
    }
    true
}
