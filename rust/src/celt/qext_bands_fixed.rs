//! CELT band quantisation with the scalable quality extension.
use super::*;
use crate::celt::qext_vq;

use crate::celt::vq::celt_inner_prod_norm_shift;
const NORM_SCALING: i32 = 1 << 24;
const MIN_STEREO_ENERGY: i32 = 2;

#[derive(Debug, Clone)]
pub(crate) struct BandCtx<'mode, 'band> {
    /// Whether the caller is encoding (`true`) or decoding (`false`).
    pub encode: bool,
    /// When `true`, the quantiser should resynthesise the canonical unit vector.
    pub resynth: bool,
    /// Active CELT mode driving the band configuration.
    pub mode: &'mode OpusCustomMode<'mode>,
    /// Index of the band currently being processed.
    pub band: usize,
    /// First band where intensity stereo becomes active.
    pub intensity: usize,
    /// Spreading decision selected for the frame.
    pub spread: i32,
    /// Time/frequency resolution change applied to the band.
    pub tf_change: i32,
    /// Remaining fractional bits available to the band quantiser.
    pub remaining_bits: i32,
    /// Per-band energy targets.
    pub band_e: &'band [i32],
    /// Random seed used for collapse prevention.
    pub seed: u32,
    /// Architecture hint for platform-specific optimisations.
    pub arch: i32,
    /// Theta rounding mode used by the stereo splitting logic.
    pub theta_round: i32,
    /// Whether inverse signalling is disabled for this band.
    pub disable_inv: bool,
    /// Forces deterministic synthesis when splitting noise.
    pub avoid_split_noise: bool,
    /// Prefix of the detached lowband workspace written during this band.
    pub scratch_written: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SplitCtx {
    pub inv: bool,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub itheta_q30: i32,
    pub qalloc: i32,
}

fn compute_theta(
    ctx: &mut BandCtx<'_, '_>,
    sctx: &mut SplitCtx,
    x: &mut [i32],
    y: &mut [i32],
    n: usize,
    b: &mut i32,
    b_current: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
    coder: &mut BandCodingState<'_, '_>,
    qext: &mut QextState<'_, '_, '_, '_>,
    ext_b: &mut i32,
) {
    debug_assert!(n > 0, "band size must be positive");
    debug_assert!(x.len() >= n, "mid buffer shorter than band length");
    debug_assert!(y.len() >= n, "side buffer shorter than band length");

    let encode = ctx.encode;
    let mode = ctx.mode;
    let band = ctx.band;
    let intensity = ctx.intensity;
    let band_e = ctx.band_e;
    let log_n = i32::from(mode.log_n[band]);
    let pulse_cap = log_n + lm * (1_i32 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };

    let mut qn = compute_qn(n as i32, *b, offset, pulse_cap, stereo);
    if stereo && band >= intensity {
        qn = 1;
    }

    let mut itheta_q30 = if encode {
        qext_vq::stereo_itheta(x, y, stereo)
    } else {
        0
    };
    let mut itheta = itheta_q30 >> 16;
    let tell_before = coder.tell_frac() as i32;
    let mut inv = false;
    let imid;
    let iside;
    let mut delta;

    if qn != 1 {
        if encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = ((itheta * qn) + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = celt_udiv((itheta * 16_384) as u32, qn as u32) as i32;
                    let mid = i32::from(bitexact_cos(unquantized as i16));
                    let side = i32::from(bitexact_cos((16_384 - unquantized) as i16));
                    let log_ratio = bitexact_log2tan(side, mid);
                    let scale = ((n as i32 - 1) << 7).max(0);
                    delta = frac_mul16(scale, log_ratio);
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32_767 / qn
                } else {
                    -32_767 / qn
                };
                let mut down = ((itheta * qn) + bias) >> 14;
                down = down.clamp(0, qn - 1);
                itheta = if ctx.theta_round < 0 { down } else { down + 1 };
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let mut x_val = itheta;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if encode {
                let (fl, fh) = if x_val <= x0 {
                    (p0 * x_val, p0 * (x_val + 1))
                } else {
                    let base = (x0 + 1) * p0;
                    (base + (x_val - 1 - x0), base + (x_val - x0))
                };
                coder.encode_range(fl as u32, fh as u32, ft as u32);
            } else {
                let fs = coder.decode_range(ft as u32) as i32;
                x_val = if fs < (x0 + 1) * p0 {
                    fs / p0
                } else {
                    x0 + 1 + (fs - (x0 + 1) * p0)
                };
                let (fl, fh) = if x_val <= x0 {
                    (p0 * x_val, p0 * (x_val + 1))
                } else {
                    let base = (x0 + 1) * p0;
                    (base + (x_val - 1 - x0), base + (x_val - x0))
                };
                coder.update_range(fl as u32, fh as u32, ft as u32);
                itheta = x_val;
            }
        } else if b0 > 1 || stereo {
            if encode {
                coder.encode_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = coder.decode_uint((qn + 1) as u32) as i32;
            }
        } else {
            let half_qn = qn >> 1;
            let ft = (half_qn + 1) * (half_qn + 1);
            if encode {
                let (fl, fs) = if itheta <= half_qn {
                    let fl = (itheta * (itheta + 1)) >> 1;
                    (fl, itheta + 1)
                } else {
                    let fs = qn + 1 - itheta;
                    let fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    (fl, fs)
                };
                coder.encode_range(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = coder.decode_range(ft as u32) as i32;
                let threshold = (half_qn * (half_qn + 1)) >> 1;
                let (fl, fs);
                if fm < threshold {
                    let root = isqrt32((8 * fm + 1) as u32) as i32;
                    itheta = (root - 1) >> 1;
                    fl = (itheta * (itheta + 1)) >> 1;
                    fs = itheta + 1;
                } else {
                    let root = isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32;
                    itheta = (2 * (qn + 1) - root) >> 1;
                    fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    fs = qn + 1 - itheta;
                }
                coder.update_range(fl as u32, (fl + fs) as u32, ft as u32);
            }
        }

        debug_assert!(itheta >= 0);
        if qn > 0 {
            itheta = celt_udiv((itheta * 16_384) as u32, qn as u32) as i32;
            *ext_b = (*ext_b).min(qext.total_bits - qext.coder.tell_frac() as i32);
            if *ext_b >= ((2 * n as i32) << BITRES)
                && qext.total_bits - qext.coder.tell_frac() as i32 - 1 > (2 << BITRES)
            {
                let bits = ((*ext_b / ((2 * n as i32 - 1) << BITRES)).max(2)).min(14);
                let before = qext.coder.tell_frac() as i32;
                if encode {
                    let delta = i64::from(itheta_q30 - (itheta << 16));
                    itheta_q30 = ((delta * i64::from(qn) * i64::from((1 << bits) - 1) + (1 << 29))
                        >> 30) as i32;
                    itheta_q30 += (1 << (bits - 1)) - 1;
                    itheta_q30 = itheta_q30.clamp(0, (1 << bits) - 2);
                    qext.coder
                        .encode_uint(itheta_q30 as u32, ((1 << bits) - 1) as u32);
                } else {
                    itheta_q30 = qext.coder.decode_uint(((1 << bits) - 1) as u32) as i32;
                }
                itheta_q30 -= (1 << (bits - 1)) - 1;
                itheta_q30 = ((i64::from(itheta) << 16)
                    + i64::from(itheta_q30) * (1 << 30) / i64::from(qn * ((1 << bits) - 1)))
                .clamp(0, 1 << 30) as i32;
                *ext_b -= qext.coder.tell_frac() as i32 - before;
            } else {
                itheta_q30 = itheta << 16;
            }
        }
        if encode && stereo {
            if itheta == 0 {
                intensity_stereo(mode, x, y, band_e, band, n);
            } else {
                let x_band = &mut x[..n];
                let y_band = &mut y[..n];
                stereo_split(x_band, y_band);
            }
        }
    } else if stereo {
        if encode {
            inv = itheta > 8_192 && !ctx.disable_inv;
            if inv {
                for sample in y.iter_mut().take(n) {
                    *sample = -*sample;
                }
            }
            intensity_stereo(mode, x, y, band_e, band, n);
        }

        let threshold = 2 << BITRES;
        if *b > threshold && ctx.remaining_bits > threshold {
            if encode {
                coder.encode_bit_logp(if inv { 1 } else { 0 }, 2);
            } else {
                inv = coder.decode_bit_logp(2) != 0;
            }
        } else {
            inv = false;
        }

        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
        itheta_q30 = 0;
    }

    let tell_after = coder.tell_frac() as i32;
    let qalloc = tell_after - tell_before;
    *b -= qalloc;

    let b_mask = mask_from_bits(b_current);
    let band_scale = ((n as i32 - 1) << 7).max(0);

    if itheta == 0 {
        imid = 32_767;
        iside = 0;
        *fill &= b_mask;
        delta = -16_384;
    } else if itheta == 16_384 {
        imid = 0;
        iside = 32_767;
        let shifted = if b_current <= 0 {
            0
        } else if b_current >= 32 {
            u32::MAX
        } else {
            b_mask << (b_current as u32)
        };
        *fill &= shifted;
        delta = 16_384;
    } else {
        imid = i32::from(bitexact_cos(itheta as i16));
        iside = i32::from(bitexact_cos((16_384 - itheta) as i16));
        delta = frac_mul16(band_scale, bitexact_log2tan(iside, imid));
    }

    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.delta = delta;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
    sctx.itheta_q30 = itheta_q30;
}

pub(crate) fn quant_band_n1(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [i32],
    y: Option<&mut [i32]>,
    lowband_out: Option<&mut [i32]>,
    coder: &mut BandCodingState<'_, '_>,
) -> usize {
    debug_assert_eq!(ctx.encode, coder.is_encoder());

    quant_band_n1_channel(ctx, x, coder);
    if let Some(y_samples) = y {
        quant_band_n1_channel(ctx, y_samples, coder);
    }

    if let Some(lowband) = lowband_out.filter(|lowband| !lowband.is_empty()) {
        lowband[0] = x[0] >> 4;
    }

    1
}

fn quant_partition(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [i32],
    n: usize,
    mut b: i32,
    mut b_blocks: i32,
    lowband: Option<&mut [i32]>,
    mut lm: i32,
    gain: i32,
    mut fill: u32,
    coder: &mut BandCodingState<'_, '_>,
    qext: &mut QextState<'_, '_, '_, '_>,
    mut ext_b: i32,
) -> u32 {
    debug_assert!(n > 0, "partition length must be positive");
    debug_assert!(
        x.len() >= n,
        "partition slice shorter than requested length"
    );
    if let Some(ref slice) = lowband {
        debug_assert!(slice.len() >= n, "lowband slice shorter than partition");
    }

    let mode = ctx.mode;
    let band = ctx.band;
    let encode = ctx.encode;
    let spread = ctx.spread;

    let cache_index = i32::from(mode.cache.index[((lm + 1) as usize) * mode.num_ebands + band]);
    let cache_slice = if cache_index >= 0 {
        &mode.cache.bits[cache_index as usize..]
    } else {
        &[]
    };

    let mut cm = 0u32;
    let original_b = b_blocks;

    if lm != -1 && n > 2 && !cache_slice.is_empty() {
        let hi_index = cache_slice[0] as usize;
        if hi_index < cache_slice.len() {
            let threshold = i32::from(cache_slice[hi_index]) + 12;
            if b > threshold {
                let half = n >> 1;
                let (x_left, x_right) = x.split_at_mut(half);

                let (lowband_left, lowband_right) = match lowband {
                    Some(slice) => {
                        let (left, right) = slice.split_at_mut(half);
                        (Some(left), Some(right))
                    }
                    None => (None, None),
                };

                lm -= 1;
                if b_blocks == 1 {
                    fill = (fill & 1) | (fill << 1);
                }
                b_blocks = (b_blocks + 1) >> 1;

                let mut split = SplitCtx::default();
                compute_theta(
                    ctx, &mut split, x_left, x_right, half, &mut b, b_blocks, original_b, lm,
                    false, &mut fill, coder, qext, &mut ext_b,
                );

                let imid = crate::celt::math_fixed::celt_cos_norm32(split.itheta_q30);
                let iside = crate::celt::math_fixed::celt_cos_norm32((1 << 30) - split.itheta_q30);
                let mut delta = split.delta;
                let itheta = split.itheta;
                let qalloc = split.qalloc;

                if original_b > 1 && (itheta & 0x3fff) != 0 {
                    if itheta > 8192 {
                        let shift = (4 - lm) as u32;
                        delta -= delta >> shift;
                    } else {
                        let shift = (5 - lm) as u32;
                        let adjust = ((half as i32) << BITRES) >> shift;
                        delta = (delta + adjust).min(0);
                    }
                }

                ctx.remaining_bits -= qalloc;

                let mut mbits = (b - delta) / 2;
                mbits = mbits.clamp(0, b);
                let mut sbits = b - mbits;

                let mut rebalance = ctx.remaining_bits;

                if mbits >= sbits {
                    cm = quant_partition(
                        ctx,
                        x_left,
                        half,
                        mbits,
                        b_blocks,
                        lowband_left,
                        lm,
                        mult32_32_q31(gain, imid),
                        fill,
                        coder,
                        qext,
                        ext_b / 2,
                    );
                    let used = rebalance - ctx.remaining_bits;
                    rebalance = mbits - used;
                    if rebalance > (3 << BITRES) && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    let cm_right = quant_partition(
                        ctx,
                        x_right,
                        half,
                        sbits,
                        b_blocks,
                        lowband_right,
                        lm,
                        mult32_32_q31(gain, iside),
                        fill >> (b_blocks as u32),
                        coder,
                        qext,
                        ext_b / 2,
                    );
                    cm |= cm_right << ((original_b >> 1) as u32);
                } else {
                    let cm_right = quant_partition(
                        ctx,
                        x_right,
                        half,
                        sbits,
                        b_blocks,
                        lowband_right,
                        lm,
                        mult32_32_q31(gain, iside),
                        fill >> (b_blocks as u32),
                        coder,
                        qext,
                        ext_b / 2,
                    );
                    cm = cm_right << ((original_b >> 1) as u32);
                    let used = rebalance - ctx.remaining_bits;
                    rebalance = sbits - used;
                    if rebalance > (3 << BITRES) && itheta != 16_384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    cm |= quant_partition(
                        ctx,
                        x_left,
                        half,
                        mbits,
                        b_blocks,
                        lowband_left,
                        lm,
                        mult32_32_q31(gain, imid),
                        fill,
                        coder,
                        qext,
                        ext_b / 2,
                    );
                }

                return cm;
            }
        }
    }

    let mut extra_bits = (ext_b / (n as i32 - 1)) >> BITRES;
    let remaining = qext.total_bits - qext.coder.tell_frac() as i32;
    if remaining < (((extra_bits + 1) * (n as i32 - 1) + n as i32) << BITRES) {
        extra_bits = (((remaining - ((n as i32) << BITRES)) / (n as i32 - 1)) >> BITRES) - 1;
        extra_bits = extra_bits.max(0);
    }
    extra_bits = extra_bits.min(14);
    let mut q = bits2pulses(mode, band, lm, b);
    let mut curr_bits = pulses2bits(mode, band, lm, q);
    ctx.remaining_bits -= curr_bits;

    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(mode, band, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        let block_count = b_blocks.max(1) as usize;
        if encode {
            cm = qext_vq::alg_quant(
                &mut x[..n],
                k,
                spread,
                block_count,
                coder.encoder_mut(),
                gain,
                ctx.resynth,
                qext.coder.encoder_mut(),
                extra_bits,
                ctx.arch,
            );
        } else {
            cm = qext_vq::alg_unquant(
                &mut x[..n],
                k,
                spread,
                block_count,
                coder.decoder_mut(),
                gain,
                qext.coder.decoder_mut(),
                extra_bits,
            );
        }
    } else if ext_b > ((2 * n as i32) << BITRES) {
        // TF recombination can leave zero blocks; cubic pulse parity and
        // collapse masks must retain that value.
        cm = if encode {
            qext_vq::cubic_quant(
                &mut x[..n],
                extra_bits,
                b_blocks as usize,
                qext.coder.encoder_mut(),
                gain,
                ctx.resynth,
            )
        } else {
            qext_vq::cubic_unquant(
                &mut x[..n],
                extra_bits,
                b_blocks as usize,
                qext.coder.decoder_mut(),
                gain,
            )
        };
    } else if ctx.resynth {
        let cm_mask = mask_from_bits(b_blocks);
        fill &= cm_mask;
        if fill == 0 {
            x[..n].fill(0);
        } else if let Some(lowband_slice) = lowband {
            let tmp = 4096;
            for (dst, src) in x.iter_mut().zip(lowband_slice.iter()) {
                ctx.seed = celt_lcg_rand(ctx.seed);
                let noise = if (ctx.seed & 0x8000) != 0 { tmp } else { -tmp };
                *dst = *src + noise;
            }
            cm = fill;
            renormalise_vector_fixed(x, n, gain, ctx.arch);
        } else {
            for sample in &mut x[..n] {
                ctx.seed = celt_lcg_rand(ctx.seed);
                let value = ((ctx.seed as i32) >> 20).wrapping_shl(10);
                *sample = value;
            }
            cm = cm_mask;
            renormalise_vector_fixed(x, n, gain, ctx.arch);
        }
    }

    cm
}

fn quant_band(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [i32],
    n: usize,
    b: i32,
    mut b_blocks: i32,
    lowband_input: Option<&mut [i32]>,
    lm: i32,
    mut lowband_out: Option<&mut [i32]>,
    gain: i32,
    mut lowband_scratch: Option<&mut [i32]>,
    mut fill: u32,
    coder: &mut BandCodingState<'_, '_>,
    qext: &mut QextState<'_, '_, '_, '_>,
    ext_b: i32,
) -> u32 {
    debug_assert!(x.len() >= n, "quant_band expects at least n coefficients");
    if let Some(ref slice) = lowband_input {
        debug_assert!(slice.len() >= n, "lowband slice shorter than partition");
    }
    if let Some(ref slice) = lowband_out {
        debug_assert!(slice.len() >= n, "lowband_out slice shorter than partition");
    }

    let encode = ctx.encode;
    let mut tf_change = ctx.tf_change;

    let n0 = n;
    let mut n_b = n;
    let mut b0 = b_blocks;
    let mut time_divide = 0;
    let mut recombine = 0;
    let long_blocks = b0 == 1;

    if b_blocks > 0 {
        n_b = celt_udiv(n_b as u32, b_blocks as u32) as usize;
    }

    if n == 1 {
        return quant_band_n1(ctx, &mut x[..1], None, lowband_out, coder) as u32;
    }

    if tf_change > 0 {
        recombine = tf_change;
    }

    let mut lowband_stack_storage = [0; MAX_CELT_BAND_SIZE];
    let copy_lowband =
        lowband_input.is_some() && (recombine > 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1);
    let mut lowband_view: Option<&mut [i32]> = None;

    if let Some(slice) = lowband_input {
        let len = n.min(slice.len());
        if copy_lowband {
            if let Some(scratch) = lowband_scratch.as_mut() {
                assert!(
                    scratch.len() >= len,
                    "lowband scratch shorter than current band width"
                );
                scratch[..len].copy_from_slice(&slice[..len]);
                ctx.scratch_written = ctx.scratch_written.max(len);
                let (head, _) = scratch.split_at_mut(len);
                lowband_view = Some(head);
            } else {
                assert!(
                    len <= lowband_stack_storage.len(),
                    "band width exceeds fixed lowband stack workspace"
                );
                lowband_stack_storage[..len].copy_from_slice(&slice[..len]);
                let (head, _) = lowband_stack_storage.split_at_mut(len);
                lowband_view = Some(head);
            }
        } else {
            let (head, _) = slice.split_at_mut(len);
            lowband_view = Some(head);
        }
    }

    for k in 0..recombine {
        const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
        if encode {
            haar1_fixed(x, n >> k, 1usize << k);
        }
        if let Some(ref mut lowband_slice) = lowband_view {
            haar1_fixed(lowband_slice, n >> k, 1usize << k);
        }
        let low = (fill & 0xF) as usize;
        let high = ((fill >> 4) & 0xF) as usize;
        let mapped =
            u32::from(BIT_INTERLEAVE_TABLE[low]) | (u32::from(BIT_INTERLEAVE_TABLE[high]) << 2);
        fill = mapped;
    }
    b_blocks >>= recombine;
    n_b <<= recombine;

    while (n_b & 1) == 0 && tf_change < 0 {
        if encode {
            haar1_fixed(x, n_b, b_blocks.max(1) as usize);
        }
        if let Some(ref mut lowband_slice) = lowband_view {
            haar1_fixed(lowband_slice, n_b, b_blocks.max(1) as usize);
        }
        let shift = b_blocks.max(1) as u32;
        fill |= fill << shift;
        b_blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }

    b0 = b_blocks;
    let n_b0 = n_b;

    if b0 > 1 {
        if encode {
            deinterleave_hadamard_fixed(
                x,
                n_b >> recombine,
                (b0 << recombine) as usize,
                long_blocks,
            );
        }
        if let Some(ref mut lowband_slice) = lowband_view {
            deinterleave_hadamard_fixed(
                lowband_slice,
                n_b >> recombine,
                (b0 << recombine) as usize,
                long_blocks,
            );
        }
    }

    let mut cm = if qext.extra_bands
        && b > ((3 * n as i32) << BITRES) + (i32::from(ctx.mode.log_n[ctx.band]) + 8 + 8 * lm)
    {
        cubic_quant_partition(ctx, &mut x[..n], b, b_blocks, lm, gain, coder)
    } else {
        quant_partition(
            ctx,
            x,
            n,
            b,
            b_blocks,
            lowband_view,
            lm,
            gain,
            fill,
            coder,
            qext,
            ext_b,
        )
    };
    if ctx.resynth {
        if b0 > 1 {
            interleave_hadamard_fixed(x, n_b >> recombine, (b0 << recombine) as usize, long_blocks);
        }

        n_b = n_b0;
        b_blocks = b0;
        for _ in 0..time_divide {
            b_blocks >>= 1;
            n_b <<= 1;
            if b_blocks > 0 {
                cm |= cm >> (b_blocks as u32);
            }
            haar1_fixed(x, n_b, b_blocks.max(1) as usize);
        }

        for k in 0..recombine {
            const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
                0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
                0xFC, 0xFF,
            ];
            cm = u32::from(BIT_DEINTERLEAVE_TABLE[cm as usize & 0xF]);
            haar1_fixed(x, n0 >> k, 1usize << k);
        }
        b_blocks <<= recombine;

        if let Some(ref mut out) = lowband_out {
            let scale = celt_sqrt_fixed((n0 as i32) << 22) as i16;
            for (dst, src) in out.iter_mut().zip(x.iter()) {
                *dst = mult16_32_q15(scale, *src);
            }
        }

        cm &= mask_from_bits(b_blocks);
    }

    cm
}

fn quant_band_stereo(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [i32],
    y: &mut [i32],
    n: usize,
    mut b: i32,
    b_blocks: i32,
    mut lowband_input: Option<&mut [i32]>,
    lm: i32,
    mut lowband_out: Option<&mut [i32]>,
    mut lowband_scratch: Option<&mut [i32]>,
    fill: u32,
    coder: &mut BandCodingState<'_, '_>,
    qext: &mut QextState<'_, '_, '_, '_>,
    mut ext_b: i32,
) -> u32 {
    debug_assert!(
        x.len() >= n && y.len() >= n,
        "stereo bands require at least n samples"
    );

    let encode = ctx.encode;

    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out, coder) as u32;
    }

    let mut fill_local = fill;
    let orig_fill = fill;

    if encode {
        let band = ctx.band;
        let mode = ctx.mode;
        let stride = mode.num_ebands;
        let left = ctx.band_e[band];
        let right = ctx.band_e[band + stride];
        if left < MIN_STEREO_ENERGY || right < MIN_STEREO_ENERGY {
            if left > right {
                y[..n].copy_from_slice(&x[..n]);
            } else {
                x[..n].copy_from_slice(&y[..n]);
            }
        }
    }

    let mut split = SplitCtx::default();
    compute_theta(
        ctx,
        &mut split,
        x,
        y,
        n,
        &mut b,
        b_blocks,
        b_blocks,
        lm,
        true,
        &mut fill_local,
        coder,
        qext,
        &mut ext_b,
    );

    let mut cm;
    let inv = split.inv;
    let delta = split.delta;
    let itheta = split.itheta;
    let qalloc = split.qalloc;
    let mid = crate::celt::math_fixed::celt_cos_norm32(split.itheta_q30);
    let side = crate::celt::math_fixed::celt_cos_norm32((1 << 30) - split.itheta_q30);

    if n == 2 {
        let mut mbits = b;
        let mut sbits = 0;
        if itheta != 0 && itheta != 16_384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let use_side = itheta > 8_192;
        ctx.remaining_bits -= qalloc + sbits;

        let mut sign = 0i32;
        {
            let (x2, y2): (&mut [i32], &mut [i32]) = if use_side {
                (&mut y[..n], &mut x[..n])
            } else {
                (&mut x[..n], &mut y[..n])
            };

            if sbits != 0 {
                if encode {
                    sign = if mult32_32_q31(x2[0], y2[1]) - mult32_32_q31(x2[1], y2[0]) < 0 {
                        1
                    } else {
                        0
                    };
                    coder.encode_bits(sign as u32, 1);
                } else {
                    sign = coder.decode_bits(1) as i32;
                }
            }
            let sign_val = 1 - 2 * sign;

            cm = quant_band(
                ctx,
                x2,
                n,
                mbits,
                b_blocks,
                lowband_input.take(),
                lm,
                lowband_out.take(),
                Q31_ONE,
                lowband_scratch.as_deref_mut(),
                orig_fill,
                coder,
                qext,
                ext_b,
            );

            y2[0] = -(sign_val as i32) * x2[1];
            y2[1] = (sign_val as i32) * x2[0];
        }

        if ctx.resynth {
            x[0] = mult32_32_q31(mid, x[0]);
            x[1] = mult32_32_q31(mid, x[1]);
            y[0] = mult32_32_q31(side, y[0]);
            y[1] = mult32_32_q31(side, y[1]);
            let tmp0 = x[0];
            x[0] = tmp0 - y[0];
            y[0] += tmp0;
            let tmp1 = x[1];
            x[1] = tmp1 - y[1];
            y[1] += tmp1;
        }
    } else {
        let mut mbits = (b - delta) / 2;
        mbits = mbits.clamp(0, b);
        let mut sbits = b - mbits;

        ctx.remaining_bits -= qalloc;
        let mut rebalance = ctx.remaining_bits;

        if mbits >= sbits {
            let qext_extra = if !qext.cap.is_empty() && ext_b != 0 {
                (mbits - qext.cap[ctx.band] / 2).min(ext_b / 2).max(0)
            } else {
                0
            };

            cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                b_blocks,
                lowband_input.take(),
                lm,
                lowband_out.take(),
                Q31_ONE,
                lowband_scratch.as_deref_mut(),
                fill_local,
                coder,
                qext,
                ext_b / 2 + qext_extra,
            );
            let used = rebalance - ctx.remaining_bits;
            rebalance = mbits - used;
            if rebalance > (3 << BITRES) && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            if qext.extra_bands {
                sbits = sbits.min(ctx.remaining_bits);
            }
            cm |= quant_band(
                ctx,
                y,
                n,
                sbits,
                b_blocks,
                None,
                lm,
                None,
                side,
                None,
                fill_local >> (b_blocks as u32),
                coder,
                qext,
                ext_b / 2 - qext_extra,
            );
        } else {
            let qext_extra = if !qext.cap.is_empty() && ext_b != 0 {
                (sbits - qext.cap[ctx.band] / 2).min(ext_b / 2).max(0)
            } else {
                0
            };
            cm = quant_band(
                ctx,
                y,
                n,
                sbits,
                b_blocks,
                None,
                lm,
                None,
                side,
                None,
                fill_local >> (b_blocks as u32),
                coder,
                qext,
                ext_b / 2 + qext_extra,
            );
            let used = rebalance - ctx.remaining_bits;
            rebalance = sbits - used;
            if rebalance > (3 << BITRES) && itheta != 16_384 {
                mbits += rebalance - (3 << BITRES);
            }
            if qext.extra_bands {
                mbits = mbits.min(ctx.remaining_bits);
            }
            cm |= quant_band(
                ctx,
                x,
                n,
                mbits,
                b_blocks,
                lowband_input,
                lm,
                lowband_out,
                Q31_ONE,
                lowband_scratch,
                fill_local,
                coder,
                qext,
                ext_b / 2 - qext_extra,
            );
        }
    }

    if ctx.resynth {
        if n != 2 {
            stereo_merge_fixed(x, y, mid);
        }
        if inv {
            for sample in &mut y[..n] {
                *sample = -*sample;
            }
        }
    }

    cm
}

pub(crate) fn quant_all_bands(
    encode: bool,
    mode: &OpusCustomMode<'_>,
    start: usize,
    end: usize,
    x: &mut [i32],
    mut y: Option<&mut [i32]>,
    collapse_masks: &mut [u8],
    band_e: &[i32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    mut dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    coder: &mut BandCodingState<'_, '_>,
    lm: i32,
    coded_bands: usize,
    seed: &mut u32,
    complexity: i32,
    arch: i32,
    disable_inv: bool,
    ext_coder: &mut BandCodingState<'_, '_>,
    extra_pulses: &[i32],
    extra_bands: bool,
    cap: &[i32],
) {
    let total_ext_bits = (coder_storage(ext_coder) as i32 * 8) << BITRES;
    let mut qext_state = QextState {
        coder: ext_coder,
        total_bits: total_ext_bits,
        extra_bands,
        cap,
    };
    let qext = &mut qext_state;
    let mut ext_balance = 0i32;
    let mut ext_tell = 0i32;
    if start >= end || end > mode.num_ebands {
        return;
    }

    let channels = if y.is_some() { 2 } else { 1 };
    let m = 1usize << (lm as usize);
    let b_blocks_base = if short_blocks { m as i32 } else { 1 };

    if mode.num_ebands == 0 {
        return;
    }

    let norm_offset = m * (mode.e_bands[start] as usize);
    let last_band_start = if mode.num_ebands > 0 {
        m * (mode.e_bands[mode.num_ebands - 1] as usize)
    } else {
        0
    };
    let norm_len = last_band_start.saturating_sub(norm_offset);

    let mut norm_storage = vec![0; channels * norm_len];
    let (norm_slice, norm2_slice) = norm_storage.split_at_mut(norm_len);
    let norm = norm_slice;
    let mut norm2 = if channels == 2 {
        Some(norm2_slice)
    } else {
        None
    };

    // Enable resynthesis when decoding or when the encoder evaluates the
    // two-pass theta RDO path.
    let theta_rdo = encode
        && y.is_some()
        && !dual_stereo
        && complexity >= 8
        && !extra_bands
        && total_ext_bits == 0;
    let resynth = !encode || (encode && y.is_some() && !dual_stereo && complexity >= 8);

    // Custom extension modes can have narrower final bands. Reserve the
    // largest band so folding scratch also covers every earlier band.
    let resynth_alloc = if resynth {
        mode.e_bands[..=mode.num_ebands]
            .windows(2)
            .map(|band| m * (band[1] - band[0]) as usize)
            .max()
            .unwrap_or(0)
    } else {
        0
    };
    let mut lowband_scratch_storage = if resynth_alloc > 0 {
        Some(vec![0; resynth_alloc])
    } else {
        None
    };

    let mut ctx = BandCtx {
        encode,
        resynth,
        mode,
        band: start,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: total_bits,
        band_e,
        seed: *seed,
        arch,
        theta_round: 0,
        disable_inv,
        avoid_split_noise: b_blocks_base > 1,
        scratch_written: 0,
    };

    let first_band_start = norm_offset;
    let mut lowband_offset: Option<usize> = None;
    let mut update_lowband = true;

    for band in start..end {
        ctx.band = band;
        ctx.scratch_written = 0;

        let last = band + 1 == end;
        let band_start = m * (mode.e_bands[band] as usize);
        let band_end = m * (mode.e_bands[band + 1] as usize);
        let n = band_end.saturating_sub(band_start);
        if n == 0 {
            continue;
        }

        let tell = coder.tell_frac() as i32;
        if band != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        if band != start {
            ext_balance += extra_pulses[band - 1] + ext_tell;
        }
        ext_tell = qext.coder.tell_frac() as i32;
        if band != start {
            ext_balance -= ext_tell;
        }
        let ext_b = if band < coded_bands {
            let current = ext_balance / (coded_bands - band).min(3) as i32;
            (total_ext_bits - ext_tell)
                .min(extra_pulses[band] + current)
                .clamp(0, 16383)
        } else {
            0
        };

        let mut b_allocation = 0i32;
        if band < coded_bands {
            let remaining_coded = (coded_bands - band).min(3) as i32;
            let curr_balance = celt_sudiv(balance, remaining_coded);
            let pulse_target = pulses.get(band).copied().unwrap_or(0) + curr_balance;
            let max_target = (remaining_bits + 1).min(pulse_target);
            b_allocation = max_target.clamp(0, 16_383);
        }
        if resynth
            && (band_start >= first_band_start.saturating_add(n) || band == start + 1)
            && (update_lowband || lowband_offset.is_none())
        {
            lowband_offset = Some(band);
        }

        if band == start + 1 {
            special_hybrid_folding_fixed(
                mode,
                &mut *norm,
                norm2.as_deref_mut(),
                start,
                m,
                dual_stereo,
            );
        }

        let tf_change = tf_res.get(band).copied().unwrap_or(0);
        ctx.tf_change = tf_change;

        if band >= mode.effective_ebands {
            lowband_scratch_storage = None;
        }
        if last {
            lowband_scratch_storage = None;
        }

        // The C decoder reuses the final left-channel band as folding
        // workspace. Custom extension modes can make this window cross into
        // the right-channel prefix, whose writes are observable at synthesis.
        // Synchronize a detached window to preserve those aliases safely.
        let scratch_offset = m * mode.e_bands[mode.effective_ebands - 1] as usize;
        if !encode && let Some(scratch) = lowband_scratch_storage.as_mut() {
            for (index, value) in scratch.iter_mut().enumerate() {
                let position = scratch_offset + index;
                if let Some(source) = x.get(position) {
                    *value = *source;
                } else if let Some(source) =
                    y.as_deref().and_then(|right| right.get(position - x.len()))
                {
                    *value = *source;
                }
            }
        }

        let x_band = &mut x[band_start..band_end];
        let mut y_band = y.as_mut().map(|slice| &mut slice[band_start..band_end]);

        let mut effective_lowband = None;
        let mut x_cm = 0u32;
        let mut y_cm = 0u32;

        if let Some(lowband_idx) = lowband_offset
            && (spread != SPREAD_AGGRESSIVE || b_blocks_base > 1 || tf_change < 0)
        {
            let lowband_start = m * (mode.e_bands[lowband_idx] as usize);
            let effective = lowband_start.saturating_sub(norm_offset).saturating_sub(n);
            effective_lowband = Some(effective);

            let threshold = effective.saturating_add(norm_offset).saturating_add(n);

            let mut fold_start = lowband_idx;
            while fold_start > 0 {
                fold_start -= 1;
                if m * (mode.e_bands[fold_start] as usize) <= effective + norm_offset {
                    break;
                }
            }

            let mut fold_end = lowband_idx.saturating_sub(1);
            while {
                fold_end = fold_end.saturating_add(1);
                fold_end < band && m * (mode.e_bands[fold_end] as usize) < threshold
            } {}

            for fold in fold_start..fold_end {
                let base = fold * channels;
                x_cm |= u32::from(collapse_masks.get(base).copied().unwrap_or(0));
                let right_index = base + channels - 1;
                y_cm |= u32::from(collapse_masks.get(right_index).copied().unwrap_or(0));
            }
        }

        if effective_lowband.is_none() {
            let mask = if b_blocks_base >= 32 {
                u32::MAX
            } else {
                (1u32 << b_blocks_base) - 1
            };
            x_cm = mask;
            y_cm = mask;
        }

        if dual_stereo && band == intensity {
            dual_stereo = false;
            if resynth && let Some(norm2_slice) = norm2.as_mut() {
                for (dst, src) in norm.iter_mut().zip(norm2_slice.iter()) {
                    *dst = (*dst + *src) >> 1;
                }
            }
        }

        let mut lowband_out_offset = None;
        if !last {
            lowband_out_offset = Some(band_start.saturating_sub(norm_offset));
        }

        if dual_stereo {
            let lowband_input_offset = effective_lowband;
            {
                let mut lowband_alias_scratch = Vec::new();
                let (x_lowband_input, x_lowband_out) = lowband_in_out_mut(
                    norm,
                    lowband_input_offset,
                    lowband_out_offset,
                    n,
                    &mut lowband_alias_scratch,
                );
                x_cm = quant_band(
                    &mut ctx,
                    x_band,
                    n,
                    b_allocation / 2,
                    b_blocks_base,
                    x_lowband_input,
                    lm,
                    x_lowband_out,
                    Q31_ONE,
                    lowband_scratch_storage.as_deref_mut(),
                    x_cm,
                    coder,
                    qext,
                    ext_b / 2,
                );
            }

            if let Some(y_band_slice_ref) = y_band.as_mut()
                && let Some(norm2_buf) = norm2.as_mut()
            {
                let y_band_slice = &mut **y_band_slice_ref;
                let mut lowband_alias_scratch = Vec::new();
                let (y_lowband_input, y_lowband_out) = lowband_in_out_mut(
                    norm2_buf,
                    lowband_input_offset,
                    lowband_out_offset,
                    n,
                    &mut lowband_alias_scratch,
                );
                y_cm = quant_band(
                    &mut ctx,
                    y_band_slice,
                    n,
                    b_allocation / 2,
                    b_blocks_base,
                    y_lowband_input,
                    lm,
                    y_lowband_out,
                    Q31_ONE,
                    lowband_scratch_storage.as_deref_mut(),
                    y_cm,
                    coder,
                    qext,
                    ext_b / 2,
                );
            }
        } else if let Some(y_band_slice_ref) = y_band.as_mut() {
            let y_band_slice = &mut **y_band_slice_ref;
            let lowband_input_offset = effective_lowband;
            let initial_fill = x_cm | y_cm;
            let rdo_active = theta_rdo && band < intensity && coder.is_encoder();

            if rdo_active {
                let weights = compute_channel_weights(band_e[band], band_e[band + mode.num_ebands]);

                let ctx_initial = ctx.clone();
                let coder_initial = coder.encoder_snapshot();
                let mut x_before = vec![0; n];
                x_before.copy_from_slice(&x_band[..n]);
                let mut y_before = vec![0; n];
                y_before.copy_from_slice(&y_band_slice[..n]);

                let lowband_initial_left = lowband_out_offset.and_then(|offset| {
                    if offset + n <= norm.len() {
                        Some((offset, norm[offset..offset + n].to_vec()))
                    } else {
                        None
                    }
                });
                let lowband_initial_right = lowband_out_offset.and_then(|offset| {
                    norm2.as_ref().and_then(|norm2_buf| {
                        if offset + n <= norm2_buf.len() {
                            Some((offset, norm2_buf[offset..offset + n].to_vec()))
                        } else {
                            None
                        }
                    })
                });

                ctx.theta_round = -1;
                let cm_first = {
                    let mut lowband_alias_scratch = Vec::new();
                    let (x_lowband_input_slice, x_lowband_out_slice) = lowband_in_out_mut(
                        norm,
                        lowband_input_offset,
                        lowband_out_offset,
                        n,
                        &mut lowband_alias_scratch,
                    );
                    quant_band_stereo(
                        &mut ctx,
                        x_band,
                        y_band_slice,
                        n,
                        b_allocation,
                        b_blocks_base,
                        x_lowband_input_slice,
                        lm,
                        x_lowband_out_slice,
                        lowband_scratch_storage.as_deref_mut(),
                        initial_fill,
                        coder,
                        qext,
                        ext_b,
                    )
                };
                let dist0 = mult16_32_q15(
                    weights[0],
                    celt_inner_prod_norm_shift(&x_before[..n], &x_band[..n]),
                ) + mult16_32_q15(
                    weights[1],
                    celt_inner_prod_norm_shift(&y_before[..n], &y_band_slice[..n]),
                );

                let coder_after_first = coder.encoder_snapshot();
                let ctx_after_first = ctx.clone();
                let mut x_after_first = vec![0; n];
                x_after_first.copy_from_slice(&x_band[..n]);
                let mut y_after_first = vec![0; n];
                y_after_first.copy_from_slice(&y_band_slice[..n]);
                let lowband_after_first_left = lowband_out_offset.and_then(|offset| {
                    if offset + n <= norm.len() {
                        Some((offset, norm[offset..offset + n].to_vec()))
                    } else {
                        None
                    }
                });
                let lowband_after_first_right = lowband_out_offset.and_then(|offset| {
                    norm2.as_ref().and_then(|norm2_buf| {
                        if offset + n <= norm2_buf.len() {
                            Some((offset, norm2_buf[offset..offset + n].to_vec()))
                        } else {
                            None
                        }
                    })
                });

                coder.restore_encoder_snapshot(&coder_initial);
                ctx = ctx_initial.clone();
                x_band[..n].copy_from_slice(&x_before[..n]);
                y_band_slice[..n].copy_from_slice(&y_before[..n]);
                if let Some((offset, data)) = lowband_initial_left.as_ref() {
                    let len = data.len().min(norm.len().saturating_sub(*offset));
                    norm[*offset..*offset + len].copy_from_slice(&data[..len]);
                }
                if let Some((offset, data)) = lowband_initial_right.as_ref()
                    && let Some(norm2_buf) = norm2.as_mut()
                {
                    let len = data.len().min(norm2_buf.len().saturating_sub(*offset));
                    norm2_buf[*offset..*offset + len].copy_from_slice(&data[..len]);
                }
                ctx.theta_round = 1;
                let cm_second = {
                    let mut lowband_alias_scratch = Vec::new();
                    let (x_lowband_input_slice, x_lowband_out_slice) = lowband_in_out_mut(
                        norm,
                        lowband_input_offset,
                        lowband_out_offset,
                        n,
                        &mut lowband_alias_scratch,
                    );
                    quant_band_stereo(
                        &mut ctx,
                        x_band,
                        y_band_slice,
                        n,
                        b_allocation,
                        b_blocks_base,
                        x_lowband_input_slice,
                        lm,
                        x_lowband_out_slice,
                        lowband_scratch_storage.as_deref_mut(),
                        initial_fill,
                        coder,
                        qext,
                        ext_b,
                    )
                };
                let dist1 = mult16_32_q15(
                    weights[0],
                    celt_inner_prod_norm_shift(&x_before[..n], &x_band[..n]),
                ) + mult16_32_q15(
                    weights[1],
                    celt_inner_prod_norm_shift(&y_before[..n], &y_band_slice[..n]),
                );

                if dist0 >= dist1 {
                    coder.restore_encoder_snapshot(&coder_after_first);
                    ctx = ctx_after_first;
                    x_band[..n].copy_from_slice(&x_after_first[..n]);
                    y_band_slice[..n].copy_from_slice(&y_after_first[..n]);
                    if let Some((offset, data)) = lowband_after_first_left {
                        let len = data.len().min(norm.len().saturating_sub(offset));
                        norm[offset..offset + len].copy_from_slice(&data[..len]);
                    }
                    if let Some((offset, data)) = lowband_after_first_right
                        && let Some(norm2_buf) = norm2.as_mut()
                    {
                        let len = data.len().min(norm2_buf.len().saturating_sub(offset));
                        norm2_buf[offset..offset + len].copy_from_slice(&data[..len]);
                    }
                    x_cm = cm_first;
                } else {
                    x_cm = cm_second;
                }
                y_cm = x_cm;
                ctx.theta_round = 0;
            } else {
                let mut lowband_alias_scratch = Vec::new();
                let (x_lowband_input, x_lowband_out) = lowband_in_out_mut(
                    norm,
                    lowband_input_offset,
                    lowband_out_offset,
                    n,
                    &mut lowband_alias_scratch,
                );
                x_cm = quant_band_stereo(
                    &mut ctx,
                    x_band,
                    y_band_slice,
                    n,
                    b_allocation,
                    b_blocks_base,
                    x_lowband_input,
                    lm,
                    x_lowband_out,
                    lowband_scratch_storage.as_deref_mut(),
                    initial_fill,
                    coder,
                    qext,
                    ext_b,
                );
                y_cm = x_cm;
            }
        } else {
            let lowband_input_offset = effective_lowband;
            let mut lowband_alias_scratch = Vec::new();
            let (x_lowband_input, x_lowband_out) = lowband_in_out_mut(
                norm,
                lowband_input_offset,
                lowband_out_offset,
                n,
                &mut lowband_alias_scratch,
            );
            x_cm = quant_band(
                &mut ctx,
                x_band,
                n,
                b_allocation,
                b_blocks_base,
                x_lowband_input,
                lm,
                x_lowband_out,
                Q31_ONE,
                lowband_scratch_storage.as_deref_mut(),
                x_cm | y_cm,
                coder,
                qext,
                ext_b,
            );
            y_cm = x_cm;
        }

        if !encode && let Some(scratch) = lowband_scratch_storage.as_ref() {
            let channel_len = x.len();
            // Untouched entries can overlap coefficients decoded in this
            // band, so only commit the prefix actually used as workspace.
            for (index, &value) in scratch[..ctx.scratch_written].iter().enumerate() {
                let position = scratch_offset + index;
                if let Some(destination) = x.get_mut(position) {
                    *destination = value;
                } else if let Some(destination) = y
                    .as_deref_mut()
                    .and_then(|right| right.get_mut(position - channel_len))
                {
                    *destination = value;
                }
            }
        }

        if let Some(mask) = collapse_masks.get_mut(band * channels) {
            *mask = x_cm as u8;
        }
        if let Some(mask) = collapse_masks.get_mut(band * channels + channels - 1) {
            *mask = y_cm as u8;
        }

        balance += pulses.get(band).copied().unwrap_or(0) + tell;
        let n_bits = (n as i32) << BITRES;
        update_lowband = b_allocation > n_bits;
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}

struct QextState<'a, 'b, 'c, 'd> {
    coder: &'c mut BandCodingState<'a, 'b>,
    total_bits: i32,
    extra_bands: bool,
    cap: &'d [i32],
}

fn quant_band_n1_channel(
    ctx: &mut BandCtx<'_, '_>,
    samples: &mut [i32],
    coder: &mut BandCodingState<'_, '_>,
) {
    assert!(
        !samples.is_empty(),
        "quant_band_n1 expects non-empty coefficient slices",
    );

    let mut sign = 0;
    let bit_budget = 1_i32 << BITRES;
    if ctx.remaining_bits >= bit_budget {
        if ctx.encode {
            debug_assert!(coder.is_encoder());
            sign = i32::from(samples[0] < 0);
            coder.encode_bits(sign as u32, 1);
        } else {
            debug_assert!(!coder.is_encoder());
            sign = coder.decode_bits(1) as i32;
        }
        ctx.remaining_bits -= bit_budget;
    }

    if ctx.resynth {
        samples[0] = if sign != 0 {
            -NORM_SCALING
        } else {
            NORM_SCALING
        };
    }
}

fn cubic_quant_partition(
    ctx: &mut BandCtx<'_, '_>,
    x: &mut [i32],
    mut b: i32,
    blocks: i32,
    lm: i32,
    gain: i32,
    coder: &mut BandCodingState<'_, '_>,
) -> u32 {
    let n = x.len();
    ctx.remaining_bits = ((coder_storage(coder) as i32 * 8) << BITRES) - coder.tell_frac() as i32;
    b = b.min(ctx.remaining_bits);
    if lm == 0 || b <= ((2 * n as i32) << BITRES) {
        b = (b + (((n as i32 - 1) << BITRES) / 2)).min(ctx.remaining_bits);
        let res =
            (((b - (1 << BITRES) - i32::from(ctx.mode.log_n[ctx.band]) - (lm << BITRES) - 1)
                / (n as i32 - 1))
                >> BITRES)
                .clamp(0, 14);
        let ret = if ctx.encode {
            qext_vq::cubic_quant(
                x,
                res,
                blocks as usize,
                coder.encoder_mut(),
                gain,
                ctx.resynth,
            )
        } else {
            qext_vq::cubic_unquant(x, res, blocks as usize, coder.decoder_mut(), gain)
        };
        ctx.remaining_bits =
            ((coder_storage(coder) as i32 * 8) << BITRES) - coder.tell_frac() as i32;
        ret
    } else {
        let (left, right) = x.split_at_mut(n / 2);
        let bits = 16.min((b >> BITRES) / (n as i32 - 1) + 1);
        let qtheta = if ctx.encode {
            let angle = qext_vq::stereo_itheta(left, right, false);
            let q = (angle + (1 << (29 - bits))) >> (30 - bits);
            coder.encode_uint(q as u32, ((1 << bits) + 1) as u32);
            q
        } else {
            coder.decode_uint(((1 << bits) + 1) as u32) as i32
        };
        let angle = qtheta << (30 - bits);
        b -= bits << BITRES;
        let delta = (n as i32 - 1) * 23 * ((angle >> 16) - 8192) >> (17 - BITRES);
        let g1 = crate::celt::math_fixed::celt_cos_norm32(angle);
        let g2 = crate::celt::math_fixed::celt_cos_norm32((1 << 30) - angle);
        let (b1, b2) = if angle == 0 {
            (b, 0)
        } else if angle == (1 << 30) {
            (0, b)
        } else {
            let b1 = ((b - delta) / 2).clamp(0, b);
            (b1, b - b1)
        };
        let cm = cubic_quant_partition(
            ctx,
            left,
            b1,
            (blocks + 1) / 2,
            lm - 1,
            mult32_32_q31(gain, g1),
            coder,
        );
        cm | cubic_quant_partition(
            ctx,
            right,
            b2,
            (blocks + 1) / 2,
            lm - 1,
            mult32_32_q31(gain, g2),
            coder,
        )
    }
}

fn compute_channel_weights(mut left: i32, mut right: i32) -> [i16; 2] {
    let minimum = left.min(right);
    left = left.wrapping_add(minimum / 3);
    right = right.wrapping_add(minimum / 3);
    let shift = celt_ilog2(1 + left.max(right)) - 14;
    [vshr32(left, shift) as i16, vshr32(right, shift) as i16]
}

fn intensity_stereo(
    mode: &OpusCustomMode<'_>,
    x: &mut [i32],
    y: &[i32],
    energy: &[i32],
    band: usize,
    n: usize,
) {
    let shift = celt_zlog2(energy[band].max(energy[band + mode.num_ebands])) - 13;
    let mut left = vshr32(energy[band], shift) as i16;
    let mut right = vshr32(energy[band + mode.num_ebands], shift) as i16;
    let norm = (1 + celt_sqrt_fixed(1 + mult16_16(left, left) + mult16_16(right, right))) as i16;
    left = left.min(norm - 1);
    right = right.min(norm - 1);
    let a1 = ((i32::from(left) << 15) / i32::from(norm)) as i16;
    let a2 = ((i32::from(right) << 15) / i32::from(norm)) as i16;
    for i in 0..n {
        x[i] = mult16_32_q15(a1, x[i]).wrapping_add(mult16_32_q15(a2, y[i]));
    }
}

fn stereo_split(x: &mut [i32], y: &mut [i32]) {
    for (left, right) in x.iter_mut().zip(y) {
        let l = mult32_32_q31(1518500224, *left);
        let r = mult32_32_q31(1518500224, *right);
        *left = l.wrapping_add(r);
        *right = r.wrapping_sub(l);
    }
}

#[cfg(test)]
#[path = "qext_bands_test.rs"]
mod reference_tests;
