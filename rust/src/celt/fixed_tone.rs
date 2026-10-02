//! Integer tone analysis from the pinned scalar CELT encoder.

use alloc::vec;

use super::fixed_ops::{mult16_16, mult32_32_q31, pshr32};
use super::math::celt_ilog2;
use super::math_fixed::{celt_sqrt, frac_div32_q29};

const ONE_Q29: i32 = 1 << 29;
// The first constant is formed from the C single-precision literal 1.999999f.
const LPC_LIMIT_Q29: i32 = 1_073_741_312;
const FOUR_Q29: i32 = 2_147_483_111;

/// Scales the analysis samples so the covariance sums fit in 32 bits.
pub(crate) fn normalize_tone_input(samples: &mut [i16]) {
    if samples.is_empty() {
        return;
    }
    let mut energy = samples.len() as i32;
    for &sample in samples.iter() {
        energy = energy.wrapping_add(mult16_16(sample, sample) >> 10);
    }
    let shift = 5 - (28 - celt_ilog2(energy)) / 2;
    if shift > 0 {
        for sample in samples {
            *sample = pshr32(i32::from(*sample), shift as u32) as i16;
        }
    }
}

/// Returns the CELT polynomial approximation to acos, from Q29 to Q13 radians.
pub(crate) fn acos_approx(x: i32) -> i32 {
    let flip = x < 0;
    let magnitude = x.wrapping_abs();
    let x14 = (magnitude >> 15) as i16;
    let mut value = (762 * i32::from(x14) >> 14) - 3308;
    value = (value * i32::from(x14) >> 14) + 25_726;
    value = value * celt_sqrt((1i32 << 30).wrapping_sub(magnitude.wrapping_shl(1)).max(0)) >> 16;
    if flip { 25_736 - value } else { value }
}

/// Fits the forward and backward two-tap predictor, returning Q29 coefficients.
/// An ill-conditioned covariance matrix has no usable predictor.
pub(crate) fn tone_lpc(samples: &[i16], delay: usize) -> Option<[i32; 2]> {
    let len = samples.len();
    assert!(delay > 0 && len > 2 * delay);
    let mut r00 = 0i32;
    let mut r01 = 0i32;
    let mut r02 = 0i32;
    for i in 0..len - 2 * delay {
        r00 = r00.wrapping_add(mult16_16(samples[i], samples[i]));
        r01 = r01.wrapping_add(mult16_16(samples[i], samples[i + delay]));
        r02 = r02.wrapping_add(mult16_16(samples[i], samples[i + 2 * delay]));
    }
    let mut edges = 0i32;
    for i in 0..delay {
        edges = edges.wrapping_add(
            mult16_16(samples[len + i - 2 * delay], samples[len + i - 2 * delay])
                .wrapping_sub(mult16_16(samples[i], samples[i])),
        );
    }
    let mut r11 = r00.wrapping_add(edges);
    edges = 0;
    for i in 0..delay {
        edges = edges.wrapping_add(
            mult16_16(samples[len + i - delay], samples[len + i - delay])
                .wrapping_sub(mult16_16(samples[i + delay], samples[i + delay])),
        );
    }
    let r22 = r11.wrapping_add(edges);
    edges = 0;
    for i in 0..delay {
        edges = edges.wrapping_add(
            mult16_16(samples[len + i - 2 * delay], samples[len + i - delay])
                .wrapping_sub(mult16_16(samples[i], samples[i + delay])),
        );
    }
    let mut r12 = r01.wrapping_add(edges);
    r00 = r00.wrapping_add(r22);
    r01 = r01.wrapping_add(r12);
    r11 = r11.wrapping_mul(2);
    r02 = r02.wrapping_mul(2);
    r12 = r01;
    let product = mult32_32_q31(r00, r11);
    let denominator = product.wrapping_sub(mult32_32_q31(r01, r01));
    if denominator <= product >> 10 {
        return None;
    }
    let numerator1 = mult32_32_q31(r02, r11).wrapping_sub(mult32_32_q31(r01, r12));
    let lpc1 = if numerator1 >= denominator {
        ONE_Q29
    } else if numerator1 <= -denominator {
        -ONE_Q29
    } else {
        frac_div32_q29(numerator1, denominator)
    };
    let numerator0 = mult32_32_q31(r00, r12).wrapping_sub(mult32_32_q31(r02, r01));
    let lpc0 = if numerator0 >> 1 >= denominator {
        LPC_LIMIT_Q29
    } else if numerator0 >> 1 <= -denominator {
        -LPC_LIMIT_Q29
    } else {
        frac_div32_q29(numerator0, denominator)
    };
    Some([lpc0, lpc1])
}

/// Detects a narrowband tone in planar Q12 pre-emphasized audio.
/// Returns (frequency in Q13 radians/sample, pole radius squared in Q29).
/// A frequency of -1 denotes that no stable tone was found.
pub(crate) fn tone_detect(
    input: &[i32],
    channels: usize,
    frame_size: usize,
    sample_rate: i32,
) -> (i16, i32) {
    assert!((channels == 1 || channels == 2) && input.len() >= channels * frame_size);
    assert!(frame_size > 2);
    let mut samples = vec![0i16; frame_size];
    for (i, sample) in samples.iter_mut().enumerate() {
        let value = if channels == 2 {
            (input[i] >> 1).wrapping_add(input[i + frame_size] >> 1)
        } else {
            input[i]
        };
        *sample = pshr32(value, 14) as i16;
    }
    normalize_tone_input(&mut samples);
    let mut delay = 1usize;
    let mut lpc = tone_lpc(&samples, delay);
    while delay <= (sample_rate.max(0) / 3000) as usize
        && lpc.is_none_or(|[a, b]| a > ONE_Q29 && b < 0)
    {
        delay *= 2;
        if frame_size <= 2 * delay {
            return (-1, 0);
        }
        lpc = tone_lpc(&samples, delay);
    }
    if let Some([a, b]) = lpc {
        if mult32_32_q31(a, a).wrapping_add(mult32_32_q31(FOUR_Q29, b)) < 0 {
            return (
                ((acos_approx(a >> 1) + delay as i32 / 2) / delay as i32) as i16,
                -b,
            );
        }
    }
    (-1, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn fixed_tone_matches_pinned_c_vectors() {
        let scripts = include_str!("../../tests/fixtures/reference/fixed-tone.script");
        let reference = include_str!("../../tests/fixtures/reference/fixed-tone.txt");
        assert_eq!(scripts.lines().count(), reference.lines().count());
        for (script, golden) in scripts.lines().zip(reference.lines()) {
            let mut fields = script.split_whitespace();
            let op = fields.next().unwrap();
            let name = fields.next().unwrap();
            let n = fields.next().unwrap().parse::<usize>().unwrap();
            let a = fields.next().unwrap().parse::<usize>().unwrap();
            let b = fields.next().unwrap().parse::<i32>().unwrap();
            let input: Vec<i32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(input.len(), n);
            let mut expected = golden.split_whitespace();
            assert_eq!(expected.next(), Some(name));
            let expected: Vec<i32> = expected.map(|v| v.parse().unwrap()).collect();
            let result = match op {
                "normalize" => {
                    let mut samples: Vec<i16> = input.iter().map(|&v| v as i16).collect();
                    normalize_tone_input(&mut samples);
                    samples.into_iter().map(i32::from).collect()
                }
                "acos" => input.into_iter().map(acos_approx).collect(),
                "lpc" => {
                    let samples: Vec<i16> = input.iter().map(|&v| v as i16).collect();
                    match tone_lpc(&samples, a) {
                        Some([a, b]) => vec![0, a, b],
                        None => vec![1, 0, 0],
                    }
                }
                "detect" => {
                    let (frequency, toneishness) = tone_detect(&input, a, n / a, b);
                    vec![i32::from(frequency), toneishness]
                }
                _ => panic!("unknown operation {op}"),
            };
            assert_eq!(result, expected, "{name}");
        }
    }
}
