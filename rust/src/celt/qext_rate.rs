//! Extra band allocation for the quality extension (`celt/rate.c`).

use super::super::entcode::{BITRES, ec_tell_frac};
use super::super::entdec::EcDec;
use super::super::entenc::EcEnc;
use super::super::quant_bands::E_MEANS;
use super::super::types::OpusCustomMode;
use alloc::vec;

const LAST_ZERO: [u8; 3] = [64, 50, 0];
const LAST_CAP: [u8; 3] = [110, 60, 0];
const LAST_OTHER: [u8; 4] = [120, 112, 70, 0];

fn encode_depth(enc: &mut EcEnc<'_>, depth: i32, cap: i32, last: &mut i32) {
    let symbol = if depth == 0 {
        0
    } else if depth == cap {
        1
    } else if depth == *last {
        2
    } else {
        3
    };
    if *last == 0 {
        enc.enc_icdf(symbol.min(2), &LAST_ZERO, 7);
    } else if *last == cap {
        enc.enc_icdf(symbol.min(2), &LAST_CAP, 7);
    } else {
        enc.enc_icdf(symbol, &LAST_OTHER, 7);
    }
    if symbol == 3 {
        enc.enc_uint((depth - 1) as u32, cap as u32);
    }
    *last = depth;
}

fn decode_depth(dec: &mut EcDec<'_>, cap: i32, last: &mut i32) -> i32 {
    let symbol = if *last == 0 {
        let symbol = dec.dec_icdf(&LAST_ZERO, 7);
        if symbol == 2 { 3 } else { symbol }
    } else if *last == cap {
        let symbol = dec.dec_icdf(&LAST_CAP, 7);
        if symbol == 2 { 3 } else { symbol }
    } else {
        dec.dec_icdf(&LAST_OTHER, 7)
    };
    let depth = match symbol {
        0 => 0,
        1 => cap,
        2 => *last,
        _ => 1 + dec.dec_uint(cap as u32) as i32,
    };
    *last = depth;
    depth
}

fn median_of_five<T: Copy + PartialOrd>(x: &[T]) -> T {
    let t2 = x[2];
    let (mut t0, mut t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let (mut t3, mut t4) = if x[3] > x[4] {
        (x[4], x[3])
    } else {
        (x[3], x[4])
    };
    if t0 > t3 {
        core::mem::swap(&mut t0, &mut t3);
        core::mem::swap(&mut t1, &mut t4);
    }
    let minimum = |a: T, b: T| if a < b { a } else { b };
    if t2 > t1 {
        if t1 < t3 {
            minimum(t2, t3)
        } else {
            minimum(t4, t1)
        }
    } else if t2 < t3 {
        minimum(t1, t3)
    } else {
        minimum(t2, t4)
    }
}

#[allow(clippy::too_many_arguments)]
fn code_depths(
    mode: &OpusCustomMode<'_>,
    qext_mode: Option<&OpusCustomMode<'_>>,
    start: usize,
    end: usize,
    qext_end: usize,
    depths: &mut [i32],
    extra_pulses: &mut [i32],
    extra_equant: &mut [i32],
    channels: i32,
    lm: i32,
    mut encoder: Option<&mut EcEnc<'_>>,
    mut decoder: Option<&mut EcDec<'_>>,
) {
    let mut last = 0;
    for depth in &mut depths[start..end + qext_end] {
        if let Some(enc) = encoder.as_deref_mut() {
            if ec_tell_frac(enc.ctx()) + 80 < (enc.ctx().storage * 8) << BITRES {
                encode_depth(enc, *depth, 56, &mut last);
            } else {
                *depth = 0;
            }
        } else if let Some(dec) = decoder.as_deref_mut() {
            *depth = if ec_tell_frac(dec.ctx()) + 80 < (dec.ctx().storage * 8) << BITRES {
                decode_depth(dec, 56, &mut last)
            } else {
                0
            };
        }
    }
    for i in start..end {
        extra_equant[i] = (depths[i] + 3) >> 2;
        let width = i32::from(mode.e_bands[i + 1] - mode.e_bands[i]);
        extra_pulses[i] = ((((width << lm) - 1) * channels * depths[i] * (1 << BITRES)) + 2) >> 2;
    }
    if let Some(qext) = qext_mode {
        for i in 0..qext_end {
            extra_equant[end + i] = (depths[end + i] + 3) >> 2;
            let width = i32::from(qext.e_bands[i + 1] - qext.e_bands[i]);
            extra_pulses[end + i] =
                ((((width << lm) - 1) * channels * depths[end + i] * (1 << BITRES)) + 2) >> 2;
        }
    }
}

/// Allocate extension pulse and energy bits for the floating-point profile.
#[allow(clippy::too_many_arguments)]
pub(crate) fn clt_compute_extra_allocation(
    mode: &OpusCustomMode<'_>,
    qext_mode: Option<&OpusCustomMode<'_>>,
    start: usize,
    end: usize,
    qext_end: usize,
    band_log_e: &[f32],
    qext_band_log_e: &[f32],
    mut total: i32,
    extra_pulses: &mut [i32],
    extra_equant: &mut [i32],
    channels: i32,
    lm: i32,
    encoder: Option<&mut EcEnc<'_>>,
    decoder: Option<&mut EcDec<'_>>,
    tone_freq: f32,
    toneishness: f32,
) {
    let bands = end + qext_end;
    assert!(end >= start && end <= mode.num_ebands);
    assert!(extra_pulses.len() >= bands && extra_equant.len() >= bands);
    let samples = if let Some(qext) = qext_mode {
        assert_eq!(end, mode.num_ebands);
        (i32::from(qext.e_bands[qext_end] - mode.e_bands[start]) * channels) << lm
    } else {
        (i32::from(mode.e_bands[end] - mode.e_bands[start]) * channels) << lm
    };
    if total <= 0 {
        extra_pulses[start..mode.num_ebands + qext_end].fill(0);
        extra_equant[start..mode.num_ebands + qext_end].fill(0);
        return;
    }
    let mut depth = vec![0; bands];
    if encoder.is_some() {
        let mut flat = vec![0.0f32; bands];
        let mut minimum = vec![0.0f32; bands];
        let mut count = vec![0i32; bands];
        for i in start..end {
            count[i] = (i32::from(mode.e_bands[i + 1] - mode.e_bands[i]) * channels) << lm;
            let baseline = |value: f32| {
                value - 0.0625 * f32::from(mode.log_n[i]) + E_MEANS[i]
                    - 0.0062 * (i + 5) as f32 * (i + 5) as f32
            };
            flat[i] = baseline(band_log_e[i]);
            if channels == 2 {
                flat[i] = flat[i].max(baseline(band_log_e[mode.num_ebands + i]));
            }
        }
        if let Some(qext) = qext_mode {
            let min_depth = if total
                >= (3 * channels * i32::from(qext.e_bands[qext_end] - qext.e_bands[0]))
                    << lm
                    << BITRES
                && (toneishness < 0.98 || tone_freq > 1.33)
            {
                1.0
            } else {
                0.0
            };
            for i in 0..qext_end {
                count[end + i] =
                    (i32::from(qext.e_bands[i + 1] - qext.e_bands[i]) * channels) << lm;
                minimum[end + i] = min_depth;
                let baseline = |value: f32| {
                    value - 0.0625 * f32::from(qext.log_n[i]) + E_MEANS[i]
                        - 0.0062 * (end + i + 5) as f32 * (end + i + 5) as f32
                };
                flat[end + i] = baseline(qext_band_log_e[i]);
                if channels == 2 {
                    flat[end + i] =
                        flat[end + i].max(baseline(qext_band_log_e[qext.num_ebands + i]));
                }
            }
        }
        let mut follower = vec![0.0f32; bands];
        if bands - start >= 5 {
            for i in start + 2..bands - 2 {
                follower[i] = median_of_five(&flat[i - 2..i + 3]);
            }
            follower[start] = follower[start + 2];
            follower[start + 1] = follower[start + 2];
            follower[bands - 1] = follower[bands - 3];
            follower[bands - 2] = follower[bands - 3];
        } else {
            follower[start..bands].copy_from_slice(&flat[start..bands]);
        }
        for i in start + 1..bands {
            follower[i] = follower[i].max(follower[i - 1] - 1.0);
        }
        for i in (start..bands - 1).rev() {
            follower[i] = follower[i].max(follower[i + 1] - 1.0);
        }
        if qext_mode.is_some() {
            for i in 0..qext_end {
                flat[end + i] = flat[end + i] + 4.0 + 0.3 * i as f32;
                follower[end + i] = follower[end + i] + 5.0 + 0.6 * i as f32;
            }
        }
        for (offset, boost) in [0.25, 0.5, 1.2, 2.0].into_iter().enumerate() {
            flat[end - 4 + offset] += boost;
            follower[end - 4 + offset] += boost;
        }
        let mut cap: alloc::vec::Vec<f32> = flat
            .iter()
            .map(|value| (value + 9.0).clamp(0.0, 14.0))
            .collect();
        let mut sum = 0.0f32;
        for i in start..bands {
            sum += count[i] as f32 * cap[i];
        }
        total >>= BITRES;
        if sum <= total as f32 {
            let mut dynamic_samples = 0;
            for i in start..bands {
                if cap[i] > 0.0 {
                    dynamic_samples += count[i];
                }
            }
            let overfill = (total as f32 - sum) / dynamic_samples.max(1) as f32;
            for i in start..bands {
                if cap[i] > 0.0 {
                    cap[i] = (cap[i] + overfill).min(14.0);
                }
            }
            for i in start..bands {
                depth[i] = libm::floor(0.5 + f64::from(4.0 * cap[i])) as i32;
            }
        } else {
            for i in start..bands {
                flat[i] -= (1.0 - toneishness) * follower[i];
            }
            sum = 0.0;
            for i in start..bands {
                sum += count[i] as f32 * flat[i];
            }
            let mut fill = (total as f32 + sum) / samples as f32;
            for _ in 0..20 {
                sum = 0.0;
                for i in start..bands {
                    sum += count[i] as f32 * cap[i].min(minimum[i].max(flat[i] - fill));
                }
                fill -= (total as f32 - sum) / samples as f32;
            }
            for i in start..bands {
                depth[i] =
                    libm::floor(0.5 + f64::from(4.0 * cap[i].min(minimum[i].max(flat[i] - fill))))
                        as i32;
            }
        }
    }
    code_depths(
        mode,
        qext_mode,
        start,
        end,
        qext_end,
        &mut depth,
        extra_pulses,
        extra_equant,
        channels,
        lm,
        encoder,
        decoder,
    );
}

/// Allocate extension bits with the reference Q24 energies and Q10 fill levels.
#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn clt_compute_extra_allocation_fixed(
    mode: &OpusCustomMode<'_>,
    qext_mode: Option<&OpusCustomMode<'_>>,
    start: usize,
    end: usize,
    qext_end: usize,
    band_log_e: &[i32],
    qext_band_log_e: &[i32],
    mut total: i32,
    extra_pulses: &mut [i32],
    extra_equant: &mut [i32],
    channels: i32,
    lm: i32,
    encoder: Option<&mut EcEnc<'_>>,
    decoder: Option<&mut EcDec<'_>>,
    tone_freq: i16,
    toneishness: i32,
) {
    use super::super::fixed_ops::{pshr32, qconst32};
    let bands = end + qext_end;
    assert!(end >= start && end <= mode.num_ebands);
    assert!(extra_pulses.len() >= bands && extra_equant.len() >= bands);
    let samples = if let Some(qext) = qext_mode {
        assert_eq!(end, mode.num_ebands);
        (i32::from(qext.e_bands[qext_end] - mode.e_bands[start]) * channels) << lm
    } else {
        (i32::from(mode.e_bands[end] - mode.e_bands[start]) * channels) << lm
    };
    if total <= 0 {
        extra_pulses[start..mode.num_ebands + qext_end].fill(0);
        extra_equant[start..mode.num_ebands + qext_end].fill(0);
        return;
    }
    let mut depth = vec![0; bands];
    if encoder.is_some() {
        let mut flat = vec![0i16; bands];
        let mut minimum = vec![0i16; bands];
        let mut count = vec![0i32; bands];
        let slope = qconst32(f64::from(0.0062_f32), 24);
        for i in start..end {
            count[i] = (i32::from(mode.e_bands[i + 1] - mode.e_bands[i]) * channels) << lm;
            let baseline = |value: i32| {
                pshr32(
                    value
                        .wrapping_sub((1 << 20) * i32::from(mode.log_n[i]))
                        .wrapping_add(qconst32(f64::from(E_MEANS[i]), 24))
                        .wrapping_sub(slope * (i + 5) as i32 * (i + 5) as i32),
                    14,
                )
            };
            flat[i] = baseline(band_log_e[i]) as i16;
            if channels == 2 {
                flat[i] = i32::from(flat[i]).max(baseline(band_log_e[mode.num_ebands + i])) as i16;
            }
        }
        if let Some(qext) = qext_mode {
            let min_depth = if total
                >= (3 * channels * i32::from(qext.e_bands[qext_end] - qext.e_bands[0]))
                    << lm
                    << BITRES
                && (toneishness < qconst32(f64::from(0.98_f32), 29)
                    || i32::from(tone_freq) > qconst32(f64::from(1.33_f32), 13))
            {
                1024
            } else {
                0
            };
            for i in 0..qext_end {
                count[end + i] =
                    (i32::from(qext.e_bands[i + 1] - qext.e_bands[i]) * channels) << lm;
                minimum[end + i] = min_depth;
                let baseline = |value: i32| {
                    pshr32(
                        value
                            .wrapping_sub((1 << 20) * i32::from(qext.log_n[i]))
                            .wrapping_add(qconst32(f64::from(E_MEANS[i]), 24))
                            .wrapping_sub(slope * (end + i + 5) as i32 * (end + i + 5) as i32),
                        14,
                    )
                };
                flat[end + i] = baseline(qext_band_log_e[i]) as i16;
                if channels == 2 {
                    flat[end + i] = i32::from(flat[end + i])
                        .max(baseline(qext_band_log_e[qext.num_ebands + i]))
                        as i16;
                }
            }
        }
        let mut follower = vec![0i16; bands];
        if bands - start >= 5 {
            for i in start + 2..bands - 2 {
                follower[i] = median_of_five(&flat[i - 2..i + 3]);
            }
            follower[start] = follower[start + 2];
            follower[start + 1] = follower[start + 2];
            follower[bands - 1] = follower[bands - 3];
            follower[bands - 2] = follower[bands - 3];
        } else {
            follower[start..bands].copy_from_slice(&flat[start..bands]);
        }
        for i in start + 1..bands {
            follower[i] = i32::from(follower[i]).max(i32::from(follower[i - 1]) - 1024) as i16;
        }
        for i in (start..bands - 1).rev() {
            follower[i] = i32::from(follower[i]).max(i32::from(follower[i + 1]) - 1024) as i16;
        }
        if qext_mode.is_some() {
            for i in 0..qext_end {
                flat[end + i] = (i32::from(flat[end + i]) + 4096 + 307 * i as i32) as i16;
                follower[end + i] = (i32::from(follower[end + i]) + 5120 + 614 * i as i32) as i16;
            }
        }
        for (offset, boost) in [256i16, 512, 1229, 2048].into_iter().enumerate() {
            flat[end - 4 + offset] = flat[end - 4 + offset].wrapping_add(boost);
            follower[end - 4 + offset] = follower[end - 4 + offset].wrapping_add(boost);
        }
        let mut cap: alloc::vec::Vec<i16> = flat
            .iter()
            .map(|&value| (i32::from(value) + 9216).clamp(0, 14336) as i16)
            .collect();
        let mut sum = 0i32;
        for i in start..bands {
            sum += count[i] * i32::from(cap[i]);
        }
        total >>= BITRES;
        if sum <= total << 10 {
            let mut dynamic_samples = 0;
            for i in start..bands {
                if cap[i] > 0 {
                    dynamic_samples += count[i];
                }
            }
            let overfill = ((total << 10) - sum) / dynamic_samples.max(1);
            for i in start..bands {
                if cap[i] > 0 {
                    cap[i] = (i32::from(cap[i]) + overfill).min(14336) as i16;
                }
            }
            for i in start..bands {
                depth[i] = pshr32(i32::from(cap[i]), 8);
            }
        } else {
            let factor = (32767 - pshr32(toneishness, 14)) as i16;
            for i in start..bands {
                flat[i] = (i32::from(flat[i])
                    - ((i32::from(factor) * i32::from(follower[i])) >> 15))
                    as i16;
            }
            sum = 0;
            for i in start..bands {
                sum += count[i] * i32::from(flat[i]);
            }
            let mut fill = ((total << 10) + sum) / samples;
            for _ in 0..20 {
                sum = 0;
                for i in start..bands {
                    sum += count[i]
                        * i32::from(cap[i])
                            .min(i32::from(minimum[i]).max(i32::from(flat[i]) - fill));
                }
                fill -= ((total << 10) - sum) / samples;
            }
            for i in start..bands {
                depth[i] = pshr32(
                    i32::from(cap[i]).min(i32::from(minimum[i]).max(i32::from(flat[i]) - fill)),
                    8,
                );
            }
        }
    }
    code_depths(
        mode,
        qext_mode,
        start,
        end,
        qext_end,
        &mut depth,
        extra_pulses,
        extra_equant,
        channels,
        lm,
        encoder,
        decoder,
    );
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "fixed_point"))]
    use super::clt_compute_extra_allocation as allocate;
    #[cfg(feature = "fixed_point")]
    use super::clt_compute_extra_allocation_fixed as allocate;
    use super::*;
    use crate::celt::modes::{compute_qext_mode, opus_custom_mode_find_static_ref};
    #[cfg(feature = "fixed_point")]
    use crate::celt::quant_bands::{
        quant_fine_energy_fixed_with_previous as fine,
        unquant_fine_energy_fixed_with_previous as unfine,
    };
    #[cfg(not(feature = "fixed_point"))]
    use crate::celt::quant_bands::{
        quant_fine_energy_with_previous as fine, unquant_fine_energy_with_previous as unfine,
    };
    use alloc::vec::Vec;
    #[cfg(feature = "fixed_point")]
    type Energy = i32;
    #[cfg(not(feature = "fixed_point"))]
    type Energy = f32;

    fn energy(value: i32, divisor: i32) -> Energy {
        #[cfg(feature = "fixed_point")]
        {
            value * (1 << 24) / divisor
        }
        #[cfg(not(feature = "fixed_point"))]
        {
            value as f32 / divisor as f32
        }
    }
    fn expected_energy(line: &str) -> Vec<Energy> {
        line.split_whitespace()
            .map(|x| {
                #[cfg(feature = "fixed_point")]
                {
                    x.parse().unwrap()
                }
                #[cfg(not(feature = "fixed_point"))]
                {
                    f32::from_bits(u32::from_str_radix(x, 16).unwrap())
                }
            })
            .collect()
    }
    fn equal_energy(actual: &[Energy], expected: &[Energy]) {
        #[cfg(feature = "fixed_point")]
        assert_eq!(actual, expected);
        #[cfg(not(feature = "fixed_point"))]
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    fn hex(line: &str) -> Vec<u8> {
        line.as_bytes()
            .chunks_exact(2)
            .map(|x| u8::from_str_radix(core::str::from_utf8(x).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn allocation_and_fine_energy_match_c() {
        #[cfg(feature = "fixed_point")]
        let fixture = include_str!("../../tests/fixtures/reference/fixed-qext-allocation.tsv");
        #[cfg(not(feature = "fixed_point"))]
        let fixture = include_str!("../../tests/fixtures/reference/qext-allocation.tsv");
        let mut lines = fixture.lines();
        let mut count = 0;
        while let Some(header) = lines.next() {
            let parts: Vec<_> = header.split_whitespace().collect();
            let v = |i: usize| parts[i].parse::<i32>().unwrap();
            if parts[0] == "A" {
                let rate = v(1);
                let channels = v(2);
                let lm = v(3);
                let qend = v(4) as usize;
                let start = v(5) as usize;
                let total = v(6);
                let pattern = v(7);
                let mode = opus_custom_mode_find_static_ref(rate, (rate / 50) as usize).unwrap();
                let qmode = compute_qext_mode(mode);
                let extension = if qend > 0 { Some(&qmode) } else { None };
                let end = mode.num_ebands;
                let expected_p: Vec<i32> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let expected_q: Vec<i32> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let packet = hex(lines.next().unwrap());
                let bands: Vec<_> = (0..2 * end)
                    .map(|i| energy((i as i32 * 13 + pattern * 7) % 97 - 55, 4))
                    .collect();
                let qb: Vec<_> = (0..28)
                    .map(|i| energy((i * 11 + pattern * 19) % 79 - 45, 4))
                    .collect();
                #[cfg(not(feature = "fixed_point"))]
                let (frequency, tone) = if pattern == 1 {
                    (1.7, 0.99)
                } else {
                    (0.4, 0.25)
                };
                #[cfg(feature = "fixed_point")]
                let (frequency, tone) = {
                    use crate::celt::fixed_ops::qconst32;
                    if pattern == 1 {
                        (
                            qconst32(f64::from(1.7f32), 13) as i16,
                            qconst32(f64::from(0.99f32), 29),
                        )
                    } else {
                        (
                            qconst32(f64::from(0.4f32), 13) as i16,
                            qconst32(f64::from(0.25f32), 29),
                        )
                    }
                };
                let mut pulses = vec![0; end + qend];
                let mut quant = vec![0; end + qend];
                let mut storage = vec![0; packet.len()];
                let mut enc = EcEnc::new(&mut storage);
                allocate(
                    mode,
                    extension,
                    start,
                    end,
                    qend,
                    &bands,
                    &qb,
                    total,
                    &mut pulses,
                    &mut quant,
                    channels,
                    lm,
                    Some(&mut enc),
                    None,
                    frequency,
                    tone,
                );
                assert_eq!(pulses, expected_p, "{header}");
                assert_eq!(quant, expected_q, "{header}");
                assert_eq!(enc.ctx().rng, parts[8].parse::<u32>().unwrap(), "{header}");
                assert_eq!(
                    ec_tell_frac(enc.ctx()),
                    parts[9].parse::<u32>().unwrap(),
                    "{header}"
                );
                enc.enc_done();
                drop(enc);
                assert_eq!(storage, packet, "{header}");
                pulses.fill(0);
                quant.fill(0);
                let mut dec = EcDec::new(&packet);
                allocate(
                    mode,
                    extension,
                    start,
                    end,
                    qend,
                    &[],
                    &[],
                    total,
                    &mut pulses,
                    &mut quant,
                    channels,
                    lm,
                    None,
                    Some(&mut dec),
                    frequency,
                    tone,
                );
                assert_eq!(pulses, expected_p, "{header}");
                assert_eq!(quant, expected_q, "{header}");
            } else {
                let channels = v(1) as usize;
                let previous = v(2);
                let extra = v(3);
                let size = v(4) as usize;
                let expected_old = expected_energy(lines.next().unwrap());
                let expected_error = expected_energy(lines.next().unwrap());
                let packet = hex(lines.next().unwrap());
                let mode = opus_custom_mode_find_static_ref(48000, 960).unwrap();
                let n = mode.num_ebands;
                let mut old: Vec<_> = (0..channels * n)
                    .map(|i| energy((i % 17) as i32 - 8, 4))
                    .collect();
                let initial = old.clone();
                let mut error: Vec<_> = (0..channels * n)
                    .map(|i| energy((i % 13) as i32 - 6, 16))
                    .collect();
                let prev = vec![previous; n];
                let bits = vec![extra; n];
                let mut storage = vec![0; size];
                let mut enc = EcEnc::new(&mut storage);
                fine(
                    mode,
                    0,
                    n,
                    &mut old,
                    &mut error,
                    Some(&prev),
                    &bits,
                    &mut enc,
                    channels,
                );
                equal_energy(&old, &expected_old);
                equal_energy(&error, &expected_error);
                assert_eq!(enc.ctx().rng, parts[5].parse::<u32>().unwrap());
                assert_eq!(ec_tell_frac(enc.ctx()), parts[6].parse::<u32>().unwrap());
                enc.enc_done();
                drop(enc);
                assert_eq!(storage, packet);
                let mut decoded = initial;
                let mut dec = EcDec::new(&packet);
                unfine(
                    mode,
                    0,
                    n,
                    &mut decoded,
                    Some(&prev),
                    &bits,
                    &mut dec,
                    channels,
                );
                equal_energy(&decoded, &expected_old);
            }
            count += 1;
        }
        assert_eq!(count, 432);
    }
}
