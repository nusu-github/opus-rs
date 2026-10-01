//! Integer decision stages for the fixed-point CELT encoder.

use crate::celt::fixed_ops::{mult16_32_q15, pshr32};
use crate::celt::math_fixed::celt_sqrt;
use alloc::vec;

/// Returns transient status, Q14 time/frequency estimate, channel, and weak flag.
pub(crate) fn transient_analysis(
    input: &[i32],
    len: usize,
    channels: usize,
    allow_weak: bool,
    tone_frequency_q13: i16,
    toneishness_q29: i32,
) -> (bool, i16, usize, bool) {
    const INV: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];
    let peak = input[..channels * len]
        .iter()
        .map(|&sample| sample.saturating_abs())
        .max()
        .unwrap_or(0);
    let log2 = |value: i32| 31 - value.max(1).leading_zeros() as i32;
    let input_shift = (log2(peak.saturating_add(1)) - 14).max(0) as u32;
    let forward_shift = if allow_weak { 5 } else { 4 };
    let half = len / 2;
    let mut temporary = vec![0i16; len];
    let mut mask_metric = 0;
    let mut selected_channel = 0;
    for channel in 0..channels {
        let (mut memory0, mut memory1) = (0i32, 0i32);
        for index in 0..len {
            let x = input[channel * len + index] >> input_shift;
            let y = memory0.wrapping_add(x);
            memory0 = memory1.wrapping_add(y).wrapping_sub(x.wrapping_mul(2));
            memory1 = x.wrapping_sub(y >> 1);
            temporary[index] = pshr32(y, 2).clamp(-32767, 32767) as i16;
        }
        temporary[..len.min(12)].fill(0);
        let max_sample = temporary
            .iter()
            .map(|&sample| i32::from(sample).abs())
            .max()
            .unwrap_or(1)
            .max(1);
        let shift = (14 - log2(max_sample)) as u32;
        for sample in &mut temporary {
            *sample = sample.wrapping_shl(shift);
        }
        let mut mean = 0i32;
        memory0 = 0;
        for index in 0..half {
            let left = i32::from(temporary[2 * index]);
            let right = i32::from(temporary[2 * index + 1]);
            let energy = pshr32(left * left + right * right, 4);
            mean = mean.wrapping_add(pshr32(energy, 12));
            memory0 = memory0.wrapping_add(pshr32(energy.wrapping_sub(memory0), forward_shift));
            temporary[index] = pshr32(memory0, 12) as i16;
        }
        memory0 = 0;
        let mut max_energy = 0i16;
        for index in (0..half).rev() {
            memory0 = memory0.wrapping_add(pshr32(
                (i32::from(temporary[index]) << 4).wrapping_sub(memory0),
                3,
            ));
            temporary[index] = pshr32(memory0, 4) as i16;
            max_energy = max_energy.max(temporary[index]);
        }
        mean = celt_sqrt(mean) * celt_sqrt(i32::from(max_energy) * (half as i32 >> 1));
        let norm = ((half as i32) << 20) / (1 + (mean >> 1));
        let mut unmask = 0i32;
        for index in (12..half.saturating_sub(5)).step_by(4) {
            let id = mult16_32_q15(temporary[index].wrapping_add(1), norm).clamp(0, 127) as usize;
            unmask += i32::from(INV[id]);
        }
        unmask = 64 * unmask * 4 / (6 * (half as i32 - 17));
        if unmask > mask_metric {
            selected_channel = channel;
            mask_metric = unmask;
        }
    }
    let mut transient = mask_metric > 200;
    if toneishness_q29 > 526_133_504 && tone_frequency_q13 < 213 {
        transient = false;
        mask_metric = 0;
    }
    let weak = allow_weak && transient && mask_metric < 600;
    if weak {
        transient = false;
    }
    let maximum = (celt_sqrt(27 * mask_metric) - 42).clamp(0, 163);
    let estimate = celt_sqrt(((113 * maximum) << 14).saturating_sub(37_312_528).max(0));
    (transient, estimate as i16, selected_channel, weak)
}

use crate::celt::fixed_arch::DB_SHIFT;
use crate::celt::fixed_ops::mult16_16_q15;
use crate::celt::quant_bands::celt_log2_q10;
use crate::celt::types::{AnalysisInfo, OpusCustomMode};
use crate::celt::vq::celt_inner_prod_norm_shift;

pub(crate) fn stereo_analysis(mode: &OpusCustomMode<'_>, x: &[i32], lm: usize, n: usize) -> bool {
    let (mut lr, mut ms) = (1i32, 1i32);
    for index in 0..(mode.e_bands[13] as usize) << lm {
        let left = x[index] >> 10;
        let right = x[n + index] >> 10;
        lr += left.abs() + right.abs();
        ms += (left + right).abs() + (left - right).abs();
    }
    ms = mult16_32_q15(23170, ms);
    let thetas = if lm <= 1 { 5 } else { 13 };
    let base = i32::from(mode.e_bands[13]) << (lm + 1);
    mult16_32_q15((base + thetas) as i16, ms) > mult16_32_q15(base as i16, lr)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn alloc_trim_analysis(
    mode: &OpusCustomMode<'_>,
    x: &[i32],
    log_e: &[i32],
    end: usize,
    lm: usize,
    channels: usize,
    n: usize,
    analysis: &AnalysisInfo,
    stereo_saving: &mut i16,
    tf_estimate: i16,
    intensity: usize,
    surround_trim: i32,
    equiv_rate: i32,
) -> i32 {
    let mut trim = if equiv_rate < 64000 {
        4 << 8
    } else if equiv_rate < 80000 {
        (4 << 8) + 16 * ((equiv_rate - 64000) >> 10)
    } else {
        5 << 8
    };
    if channels == 2 {
        let correlation = |band: usize| {
            let first = (mode.e_bands[band] as usize) << lm;
            let last = (mode.e_bands[band + 1] as usize) << lm;
            (celt_inner_prod_norm_shift(&x[first..last], &x[n + first..n + last]) >> 18) as i16
        };
        let mut sum = 0i16;
        for band in 0..8 {
            sum = sum.wrapping_add(correlation(band));
        }
        sum = mult16_16_q15(4096, sum).wrapping_abs().min(1024);
        let mut minimum = sum;
        for band in 8..intensity {
            minimum = minimum.min(correlation(band).wrapping_abs());
        }
        minimum = minimum.wrapping_abs().min(1024);
        let log = celt_log2_q10(1049625 - i32::from(sum) * i32::from(sum));
        let log2 = (log >> 1).max(celt_log2_q10(
            1049625 - i32::from(minimum) * i32::from(minimum),
        ));
        let log = pshr32(i32::from(log) - 6144, 2) as i16;
        let log2 = pshr32(i32::from(log2) - 6144, 2) as i16;
        trim += i32::from(mult16_16_q15(24576, log).max(-1024));
        *stereo_saving = stereo_saving.wrapping_add(64).min(-(log2 >> 1));
    }
    let mut difference = 0i32;
    for channel in 0..channels {
        for band in 0..end - 1 {
            difference +=
                (log_e[channel * mode.num_ebands + band] >> 5) * (2 + 2 * band as i32 - end as i32);
        }
    }
    difference /= (channels * (end - 1)) as i32;
    trim -= (((difference + (1 << (DB_SHIFT - 5))) >> (DB_SHIFT - 13)) / 6).clamp(-512, 512);
    trim -= surround_trim >> (DB_SHIFT - 8);
    trim -= 2 * (i32::from(tf_estimate) >> 6);
    if analysis.valid {
        trim -= ((512.0 * (analysis.tonality_slope + 0.05)) as i32).clamp(-512, 512);
    }
    pshr32(trim, 8).clamp(0, 10)
}

use crate::celt::bands::haar1_fixed;
use crate::celt::celt::TF_SELECT_TABLE;
use core::cmp::max;
const MAX_TF_BANDS: usize = 25;
const MAX_TF_BAND_SIZE: usize = 208;

fn l1_metric(values: &[i32], count: usize, lm: i32, bias: i16) -> i32 {
    let sum = values[..count]
        .iter()
        .map(|&value| ((value >> 10) as i16).wrapping_abs() as i32)
        .sum::<i32>();
    sum + mult16_32_q15((lm * i32::from(bias)) as i16, sum)
}

pub(crate) fn tf_analysis(
    mode: &OpusCustomMode<'_>,
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[i32],
    n0: usize,
    lm: usize,
    tf_estimate: i16,
    tf_chan: usize,
    importance: &[i32],
) -> i32 {
    debug_assert!(lm < TF_SELECT_TABLE.len());
    debug_assert!(len <= tf_res.len());
    debug_assert!(len <= importance.len());
    debug_assert!(len < mode.e_bands.len());

    if len == 0 {
        return 0;
    }

    let bias = ((1311i32 * (8192 - i32::from(tf_estimate)).max(-4096)) >> 14) as i16;

    let mut max_band = 0usize;
    for band in 0..len {
        let start = mode.e_bands[band] as usize;
        let end = mode.e_bands[band + 1] as usize;
        let width = end.saturating_sub(start);
        max_band = max(max_band, width << lm);
    }

    debug_assert!(len <= MAX_TF_BANDS);
    debug_assert!(max_band <= MAX_TF_BAND_SIZE);

    let mut metric_storage = [0i32; MAX_TF_BANDS];
    let mut path0_storage = [0i32; MAX_TF_BANDS];
    let mut path1_storage = [0i32; MAX_TF_BANDS];
    let mut tmp_storage = [0i32; MAX_TF_BAND_SIZE];
    let mut tmp_alt_storage = [0i32; MAX_TF_BAND_SIZE];

    let metric = &mut metric_storage[..len];
    let path0 = &mut path0_storage[..len];
    let path1 = &mut path1_storage[..len];
    let tmp = &mut tmp_storage[..max_band.max(1)];
    let tmp_alt = &mut tmp_alt_storage[..max_band.max(1)];

    let lm_i32 = lm as i32;

    for band in 0..len {
        let start = mode.e_bands[band] as usize;
        let end = mode.e_bands[band + 1] as usize;
        let width = end.saturating_sub(start);
        let n = width << lm;
        if n == 0 {
            continue;
        }

        let offset = tf_chan * n0 + (start << lm);
        debug_assert!(offset + n <= x.len());
        tmp[..n].copy_from_slice(&x[offset..offset + n]);

        let narrow = width == 1;
        let mut best_level = 0i32;
        let mut best_l1 = l1_metric(&tmp[..n], n, if is_transient { lm_i32 } else { 0 }, bias);

        if is_transient && !narrow {
            tmp_alt[..n].copy_from_slice(&tmp[..n]);
            let blocks = n >> lm;
            if blocks > 0 {
                haar1_fixed(&mut tmp_alt[..n], blocks, 1 << lm);
                let l1 = l1_metric(&tmp_alt[..n], n, lm_i32 + 1, bias);
                if l1 < best_l1 {
                    best_l1 = l1;
                    best_level = -1;
                }
            }
        }

        let extra = if is_transient || narrow { 0 } else { 1 };
        for k in 0..(lm + extra) {
            let blocks = n >> k;
            if blocks == 0 {
                break;
            }

            haar1_fixed(&mut tmp[..n], blocks, 1 << k);
            let b = if is_transient {
                lm_i32 - k as i32 - 1
            } else {
                k as i32 + 1
            };

            let l1 = l1_metric(&tmp[..n], n, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }

        let mut value = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (value == 0 || value == -2 * lm_i32) {
            value -= 1;
        }
        metric[band] = value;
    }

    let table = &TF_SELECT_TABLE[lm];
    let base_index = if is_transient { 4 } else { 0 };
    let mut selcost = [0i32; 2];

    for sel in 0..2 {
        let idx0 = base_index + 2 * sel;
        let idx1 = idx0 + 1;
        let target0 = 2 * i32::from(table[idx0]);
        let target1 = 2 * i32::from(table[idx1]);

        let mut cost0 = importance[0] * (metric[0] - target0).abs();
        let mut cost1 = importance[0] * (metric[0] - target1).abs();
        if !is_transient {
            cost1 += lambda;
        }

        for band in 1..len {
            let from0 = cost0;
            let from1 = cost1 + lambda;
            let curr0;
            if from0 < from1 {
                curr0 = from0;
                path0[band] = 0;
            } else {
                curr0 = from1;
                path0[band] = 1;
            }

            let from0 = cost0 + lambda;
            let from1 = cost1;
            let curr1;
            if from0 < from1 {
                curr1 = from0;
                path1[band] = 0;
            } else {
                curr1 = from1;
                path1[band] = 1;
            }

            cost0 = curr0 + importance[band] * (metric[band] - target0).abs();
            cost1 = curr1 + importance[band] * (metric[band] - target1).abs();
        }

        selcost[sel] = cost0.min(cost1);
    }

    let mut tf_select = 0i32;
    if is_transient && selcost[1] < selcost[0] {
        tf_select = 1;
    }

    let idx0 = base_index + 2 * tf_select as usize;
    let idx1 = idx0 + 1;
    let target0 = 2 * i32::from(table[idx0]);
    let target1 = 2 * i32::from(table[idx1]);

    let mut cost0 = importance[0] * (metric[0] - target0).abs();
    let mut cost1 = importance[0] * (metric[0] - target1).abs();
    if !is_transient {
        cost1 += lambda;
    }

    for band in 1..len {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0;
        if from0 < from1 {
            curr0 = from0;
            path0[band] = 0;
        } else {
            curr0 = from1;
            path0[band] = 1;
        }

        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1;
        if from0 < from1 {
            curr1 = from0;
            path1[band] = 0;
        } else {
            curr1 = from1;
            path1[band] = 1;
        }

        cost0 = curr0 + importance[band] * (metric[band] - target0).abs();
        cost1 = curr1 + importance[band] * (metric[band] - target1).abs();
    }

    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    if len >= 2 {
        for band in (0..=(len - 2)).rev() {
            let next = tf_res[band + 1];
            tf_res[band] = if next == 1 {
                path1[band + 1]
            } else {
                path0[band + 1]
            };
        }
    }

    tf_select
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dynalloc_analysis(
    log_e: &[i32],
    log_e2: &[i32],
    old_e: &[i32],
    bands: usize,
    start: usize,
    end: usize,
    channels: usize,
    offsets: &mut [i32],
    depth: i32,
    log_n: &[i16],
    transient: bool,
    vbr: bool,
    constrained: bool,
    e_bands: &[i16],
    lm: i32,
    bytes: i32,
    total_boost: &mut i32,
    lfe: bool,
    surround: &[i32],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_frequency: i32,
    toneishness: i32,
) -> i32 {
    const ONE: i32 = 1 << DB_SHIFT;
    const MEANS: [i32; 25] = [
        103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60, 60,
        60, 60, 60,
    ];
    let mut follower = vec![0i32; channels * bands];
    let mut noise = vec![0i32; bands];
    let mut temporary = vec![0i32; bands];
    let mut max_depth = -535193184;
    offsets.fill(0);
    for band in 0..end {
        noise[band] = (ONE / 16) * i32::from(log_n[band]) + ONE / 2 + (9 - depth) * ONE
            - (MEANS[band] << (DB_SHIFT - 4))
            + 104019 * (band as i32 + 5) * (band as i32 + 5);
        for channel in 0..channels {
            max_depth = max_depth.max(log_e[channel * bands + band] - noise[band]);
        }
    }
    let mut mask = vec![0i32; bands];
    for band in 0..end {
        mask[band] = log_e[band] - noise[band];
        if channels == 2 {
            mask[band] = mask[band].max(log_e[bands + band] - noise[band]);
        }
    }
    let signal = mask.clone();
    for band in 1..end {
        mask[band] = mask[band].max(mask[band - 1] - 2 * ONE);
    }
    for band in (0..end - 1).rev() {
        mask[band] = mask[band].max(mask[band + 1] - 3 * ONE);
    }
    for band in 0..end {
        let smr = signal[band] - 0.max(max_depth - 12 * ONE).max(mask[band]);
        let shift = -pshr32(smr.clamp(-5 * ONE, 0), DB_SHIFT);
        spread_weight[band] = 32 >> shift;
    }
    let mut boost_sum = 0i32;
    if bytes >= 30 + 5 * lm && !lfe {
        let median = |values: &[i32]| {
            let mut sorted = [0i32; 5];
            sorted[..values.len()].copy_from_slice(values);
            sorted[..values.len()].sort_unstable();
            sorted[values.len() / 2]
        };
        let mut last = 0;
        for channel in 0..channels {
            let base = channel * bands;
            temporary[..end].copy_from_slice(&log_e2[base..base + end]);
            if lm == 0 {
                for band in 0..8.min(end) {
                    temporary[band] = temporary[band].max(old_e[base + band]);
                }
            }
            let f = &mut follower[base..base + bands];
            f[0] = temporary[0];
            for band in 1..end {
                if temporary[band] > temporary[band - 1] + ONE / 2 {
                    last = band;
                }
                f[band] = (f[band - 1] + 3 * ONE / 2).min(temporary[band]);
            }
            for band in (0..last).rev() {
                f[band] = f[band].min((f[band + 1] + 2 * ONE).min(temporary[band]));
            }
            for band in 2..end - 2 {
                f[band] = f[band].max(median(&temporary[band - 2..band + 3]) - ONE);
            }
            let low = median(&temporary[..3]) - ONE;
            f[0] = f[0].max(low);
            f[1] = f[1].max(low);
            let high = median(&temporary[end - 3..end]) - ONE;
            f[end - 2] = f[end - 2].max(high);
            f[end - 1] = f[end - 1].max(high);
            for band in 0..end {
                f[band] = f[band].max(noise[band]);
            }
        }
        for band in start..end {
            if channels == 2 {
                follower[bands + band] = follower[bands + band].max(follower[band] - 4 * ONE);
                follower[band] = follower[band].max(follower[bands + band] - 4 * ONE);
                follower[band] = ((log_e[band] - follower[band]).max(0)
                    + (log_e[bands + band] - follower[bands + band]).max(0))
                    >> 1;
            } else {
                follower[band] = (log_e[band] - follower[band]).max(0);
            }
            follower[band] = follower[band].max(surround[band]);
            importance[band] = pshr32(
                13 * crate::celt::bands::celt_exp2_db_fixed(follower[band].min(4 * ONE)),
                16,
            );
            if (!vbr || constrained) && !transient {
                follower[band] >>= 1;
            }
            if band < 8 {
                follower[band] *= 2;
            }
            if band >= 12 {
                follower[band] >>= 1;
            }
        }
        if toneishness > 526133504 {
            let bin = pshr32(tone_frequency * 19557, 22);
            for band in start..end {
                let first = i32::from(e_bands[band]);
                let last = i32::from(e_bands[band + 1]);
                if bin >= first && bin <= last {
                    follower[band] += 2 * ONE;
                }
                if bin >= first - 1 && bin <= last + 1 {
                    follower[band] += ONE;
                }
                if bin >= first - 2 && bin <= last + 2 {
                    follower[band] += ONE;
                }
                if bin >= first - 3 && bin <= last + 3 {
                    follower[band] += ONE / 2;
                }
            }
            if bin >= i32::from(e_bands[end]) {
                follower[end - 1] += 2 * ONE;
                follower[end - 2] += ONE;
            }
        }
        if analysis.valid {
            for band in start..end.min(analysis.leak_boost.len()) {
                follower[band] += ONE / 64 * i32::from(analysis.leak_boost[band]);
            }
        }
        if bytes > 320 {
            follower[0] += (3 * ONE / 2).min(16777 * (bytes - 320));
        }
        for band in start..end {
            let value = follower[band].min(4 * ONE) >> 8;
            let width = (channels as i32 * i32::from(e_bands[band + 1] - e_bands[band])) << lm;
            let (boost, bits) = if width < 6 {
                let boost = value >> (DB_SHIFT - 8);
                (boost, boost * width << 3)
            } else if width > 48 {
                let boost = value * 8 >> (DB_SHIFT - 8);
                (boost, (boost * width << 3) / 8)
            } else {
                let boost = (value * width / 6) >> (DB_SHIFT - 8);
                (boost, boost * 6 << 3)
            };
            if (!vbr || (constrained && !transient)) && ((boost_sum + bits) >> 6) > 2 * bytes / 3 {
                let cap = (2 * bytes / 3) << 6;
                offsets[band] = cap - boost_sum;
                boost_sum = cap;
                break;
            }
            offsets[band] = boost;
            boost_sum += bits;
        }
    } else {
        importance[start..end].fill(13);
    }
    *total_boost = boost_sum;
    max_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    fn random(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }
    #[test]
    fn decision_stages_match_pinned_fixed_c() {
        let fixture = include_str!("../../tests/fixtures/reference/fixed-encoder-analysis.txt");
        let mode = crate::celt::modes::opus_custom_mode_find_static(48000, 960).unwrap();
        for line in fixture.lines() {
            let mut fields = line.split_whitespace();
            let kind = fields.next().unwrap();
            let values = fields
                .map(|field| field.parse::<i32>().unwrap())
                .collect::<alloc::vec::Vec<_>>();
            let lm = values[0] as usize;
            let channels = values[1] as usize;
            let pattern = values[2];
            if kind == "T" {
                let count = (120 << lm) + 120;
                let mut seed = 12345 + 37 * pattern as u32 + channels as u32;
                let mut input = vec![0i32; count * channels];
                for (index, sample) in input.iter_mut().enumerate() {
                    let mut value = (random(&mut seed) >> 9) as i32 - 4194304;
                    if pattern == 0 {
                        value = 0;
                    }
                    if pattern == 1 && index % count < count / 2 {
                        value >>= 12;
                    }
                    if pattern == 2 {
                        value = if index % count == count / 2 {
                            100000000
                        } else {
                            0
                        };
                    }
                    if pattern == 3 {
                        value >>= 8;
                    }
                    *sample = value;
                }
                let (transient, estimate, channel, weak) = transient_analysis(
                    &input,
                    count,
                    channels,
                    values[3] != 0,
                    if pattern == 4 { 100 } else { 500 },
                    if pattern == 4 { 530000000 } else { 0 },
                );
                assert_eq!(
                    [
                        i32::from(transient),
                        i32::from(estimate),
                        channel as i32,
                        i32::from(weak)
                    ],
                    values[4..],
                    "{line}"
                );
            } else {
                let n = 120 << lm;
                let mut seed = 4321 + 57 * pattern as u32 + channels as u32;
                let mut x = vec![0i32; n * channels];
                for sample in &mut x {
                    *sample = (random(&mut seed) >> 9) as i32 - 4194304;
                }
                let mut energy = vec![0i32; 21 * channels];
                let mut alternate = energy.clone();
                let mut old = energy.clone();
                for index in 0..energy.len() {
                    energy[index] = (random(&mut seed) % 300000001) as i32 - 100000000;
                    alternate[index] =
                        energy[index] + (random(&mut seed) % 10000001) as i32 - 5000000;
                    old[index] = energy[index] + (random(&mut seed) % 10000001) as i32 - 5000000;
                }
                let mut surround = [0i32; 21];
                if pattern == 7 {
                    for (index, value) in surround.iter_mut().enumerate() {
                        *value = (1 << 23) * (index % 4) as i32;
                    }
                }
                let mut offsets = [0i32; 21];
                let mut importance = [0i32; 21];
                let mut spread = [0i32; 21];
                let mut tf = [0i32; 21];
                let mut boost = 0;
                let mut saving = 64;
                let analysis = AnalysisInfo::default();
                let depth = dynalloc_analysis(
                    &energy,
                    &alternate,
                    &old,
                    21,
                    0,
                    21,
                    channels,
                    &mut offsets,
                    16,
                    mode.log_n,
                    pattern & 1 != 0,
                    pattern & 2 != 0,
                    pattern & 4 != 0,
                    mode.e_bands,
                    lm as i32,
                    if pattern < 2 { 24 } else { 240 },
                    &mut boost,
                    false,
                    &surround,
                    &analysis,
                    &mut importance,
                    &mut spread,
                    if pattern == 6 { 8192 } else { 0 },
                    if pattern == 6 { 530000000 } else { 0 },
                );
                let select = tf_analysis(
                    &mode,
                    21,
                    pattern & 1 != 0,
                    &mut tf,
                    120,
                    &x,
                    n,
                    lm,
                    8192,
                    channels - 1,
                    &importance,
                );
                let trim = alloc_trim_analysis(
                    &mode,
                    &x,
                    &energy,
                    21,
                    lm,
                    channels,
                    n,
                    &analysis,
                    &mut saving,
                    8192,
                    17,
                    0,
                    64000 + pattern * 8000,
                );
                let stereo = if channels == 2 {
                    i32::from(stereo_analysis(&mode, &x, lm, n))
                } else {
                    -1
                };
                let patch = patch_transient_decision(&energy, &old, 21, 0, 21, channels);
                let vbr = compute_vbr(
                    &mode,
                    &analysis,
                    1024 + 128 * pattern,
                    lm as i32,
                    64000 + 8000 * pattern,
                    21,
                    channels,
                    17,
                    pattern & 4 != 0,
                    saving,
                    boost,
                    if pattern < 4 { 512 } else { 8192 },
                    false,
                    depth,
                    false,
                    pattern & 2 != 0,
                    -(1 << 23),
                    1 << 23,
                    false,
                );
                let mut actual = vec![
                    depth,
                    boost,
                    select,
                    trim,
                    i32::from(saving),
                    stereo,
                    i32::from(patch),
                    vbr,
                ];
                for band in 0..21 {
                    actual.extend_from_slice(&[
                        offsets[band],
                        importance[band],
                        spread[band],
                        tf[band],
                    ]);
                }
                assert_eq!(
                    actual,
                    values[3..],
                    "LM {lm}, channels {channels}, pattern {pattern}"
                );
            }
        }
    }
}

pub(crate) fn patch_transient_decision(
    energy: &[i32],
    old: &[i32],
    bands: usize,
    start: usize,
    end: usize,
    channels: usize,
) -> bool {
    let mut spread = [0i32; 26];
    for band in start..end {
        spread[band] = old[band];
        if channels == 2 {
            spread[band] = spread[band].max(old[bands + band]);
        }
        if band > start {
            spread[band] = spread[band].max(spread[band - 1] - (1 << DB_SHIFT));
        }
    }
    for band in (start..end - 1).rev() {
        spread[band] = spread[band].max(spread[band + 1] - (1 << DB_SHIFT));
    }
    let mut difference = 0i32;
    for channel in 0..channels {
        for band in start.max(2)..end - 1 {
            // The reference stores these intermediates in opus_val16, including
            // its narrowing conversion from the Q24 log-energy representation.
            let current = energy[channel * bands + band].max(0) as i16;
            let previous = spread[band].max(0) as i16;
            difference += (i32::from(current) - i32::from(previous)).max(0);
        }
    }
    difference / (channels * (end - 1 - start.max(2))) as i32 > 1 << DB_SHIFT
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_vbr(
    mode: &OpusCustomMode<'_>,
    analysis: &AnalysisInfo,
    base: i32,
    lm: i32,
    bitrate: i32,
    last_bands: i32,
    channels: usize,
    intensity: i32,
    constrained: bool,
    saving: i16,
    boost: i32,
    tf: i16,
    pitch_change: bool,
    depth: i32,
    lfe: bool,
    has_mask: bool,
    masking: i32,
    temporal: i32,
    qext: bool,
) -> i32 {
    let bands = if last_bands > 0 {
        last_bands as usize
    } else {
        mode.num_ebands
    };
    let mut bins = i32::from(mode.e_bands[bands]) << lm;
    if channels == 2 {
        bins += i32::from(mode.e_bands[(intensity as usize).min(bands)]) << lm;
    }
    let mut target = base;
    if analysis.valid && analysis.activity < 0.4 {
        target -= ((bins << 3) as f32 * (0.4 - analysis.activity)) as i32;
    }
    if channels == 2 {
        let stereo_bands = (intensity as usize).min(bands);
        let dof = (i32::from(mode.e_bands[stereo_bands]) << lm) - stereo_bands as i32;
        let fraction = (26214 * dof / bins) as i16;
        target -= mult16_32_q15(fraction, target)
            .min((i32::from(saving.min(256) - 26) * (dof << 3)) >> 8);
    }
    target += boost - (19 << lm);
    target += mult16_32_q15(tf - 721, target) << 1;
    if analysis.valid && !lfe {
        let tonal = (analysis.tonality - 0.15).max(0.0) - 0.12;
        target += ((bins << 3) as f32 * 1.2 * tonal) as i32;
        if pitch_change {
            target += ((bins << 3) as f32 * 0.8) as i32;
        }
    }
    if has_mask && !lfe {
        let delta = ((masking >> (DB_SHIFT - 10)) * (bins << 3)) >> 10;
        target = (target / 4).max(target + delta);
    }
    let floor_bins = if qext {
        mode.short_mdct_size as i32
    } else {
        i32::from(mode.e_bands[mode.num_ebands - 2])
    } << lm;
    let floor = (mult16_32_q15((channels as i32 * floor_bins << 3) as i16, depth)
        >> (DB_SHIFT - 15))
        .max(target >> 2);
    target = target.min(floor);
    if (!has_mask || lfe) && constrained {
        target = base + mult16_32_q15(21955, target - base);
    }
    if !has_mask && tf < 3277 {
        let amount = mult16_16_q15(3329, (96000 - bitrate).clamp(0, 32000) as i16);
        let factor = (((temporal >> (DB_SHIFT - 10)) * i32::from(amount)) >> 10) as i16;
        target += mult16_32_q15(factor, target);
    }
    target.min(2 * base)
}

pub(crate) fn surround_analysis(
    mode: &OpusCustomMode<'_>,
    mask: &[i32],
    channels: usize,
    end: usize,
    dynalloc: &mut [i32],
) -> (i32, i32) {
    let mut average = 0i32;
    let mut difference = 0i32;
    let mut count = 0i32;
    for channel in 0..channels {
        for band in 0..end {
            let mut value =
                mask[channel * mode.num_ebands + band].clamp(-(2 << DB_SHIFT), 1 << (DB_SHIFT - 2));
            if value > 0 {
                value >>= 1;
            }
            let mask16 = (value >> (DB_SHIFT - 10)) as i16;
            let width = i32::from(mode.e_bands[band + 1] - mode.e_bands[band]);
            average += i32::from(mask16) * width;
            count += width;
            difference += i32::from(mask16) * (1 + 2 * band as i32 - end as i32);
        }
    }
    average = ((average / count) << (DB_SHIFT - 10)) + 3355443;
    difference =
        (difference * 6 / (channels * (end - 1) * (end + 1) * end) as i32) << (DB_SHIFT - 10);
    difference = (difference >> 1).clamp(-520094, 520094);
    let mut mid = 0;
    while mode.e_bands[mid + 1] < mode.e_bands[end] / 2 {
        mid += 1;
    }
    let mut boosted = 0;
    for band in 0..end {
        let linear = average + difference * (band as i32 - mid as i32);
        let mut unmask = mask[band];
        if channels == 2 {
            unmask = unmask.max(mask[mode.num_ebands + band]);
        }
        unmask = unmask.min(0) - linear;
        if unmask > 1 << (DB_SHIFT - 2) {
            dynalloc[band] = unmask - (1 << (DB_SHIFT - 2));
            boosted += 1;
        }
    }
    if boosted >= 3 {
        average += 1 << (DB_SHIFT - 2);
        if average > 0 {
            average = 0;
            difference = 0;
            dynalloc[..end].fill(0);
        } else {
            for value in &mut dynalloc[..end] {
                *value = (*value - (1 << (DB_SHIFT - 2))).max(0);
            }
        }
    }
    (average + 3355443, 64 * difference)
}
