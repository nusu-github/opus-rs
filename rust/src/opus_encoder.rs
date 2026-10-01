//! Top-level Opus encoder front-end.
//!
//! Combines SILK and CELT, selects modes and bandwidth, and preserves the
//! precision of integer, 24-bit, and floating-point PCM input.

#[cfg(feature = "dred")]
use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(test)]
extern crate std;

use crate::analysis::{
    TonalityAnalysisState, run_analysis, tonality_analysis_init, tonality_analysis_reset,
    tonality_get_info,
};
use crate::celt::AnalysisInfo;
use crate::celt::{
    CELT_SIG_SCALE, CeltCoef, CeltEncodeError, CeltEncoderCtlError, CeltEncoderInitError,
    EncoderCtlRequest as CeltEncoderCtlRequest, OpusRes, OwnedCeltEncoder, SilkInfo,
    canonical_mode, celt_encode_with_ec, celt_exp2, celt_sqrt, frac_div32,
    opus_custom_encoder_create, opus_custom_encoder_ctl, opus_select_arch,
};
use crate::opus_multistream::{OPUS_AUTO, OPUS_BITRATE_MAX};
use crate::packet::Bandwidth;
use crate::range::RangeEncoder;
#[cfg(feature = "dred")]
use crate::repacketizer::{OpusExtensionData, opus_packet_pad_with_extensions};
use crate::repacketizer::{OpusRepacketizer, RepacketizerError, opus_packet_pad};
use crate::silk::EncControl as SilkEncControl;
#[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
use crate::silk::enc_api::silk_encode;
#[cfg(not(feature = "fixed_point"))]
use crate::silk::enc_api::silk_encode_float;
use crate::silk::enc_api::{PrefillMode, silk_init_encoder};
use crate::silk::errors::SilkError;
use crate::silk::lin2log::lin2log;
use crate::silk::log2lin::log2lin;
use crate::silk::tuning_parameters::MAX_CONSECUTIVE_DTX;
use crate::silk::tuning_parameters::{
    NB_SPEECH_FRAMES_BEFORE_DTX, VARIABLE_HP_MIN_CUTOFF_HZ, VARIABLE_HP_SMTH_COEF2,
};
#[cfg(feature = "dred")]
use crate::{
    dred_constants::{DRED_MAX_DATA_SIZE, DRED_MIN_BYTES},
    dred_encoder::{
        DredEnc, dred_compute_latents, dred_encode_silk_frame, dred_encoder_init,
        dred_encoder_load_model, dred_encoder_reset,
    },
};

const MAX_CHANNELS: usize = 2;
const MAX_ENCODER_BUFFER: usize = if cfg!(feature = "enable_qext") {
    960
} else {
    480
};
const DELAY_BUFFER_SAMPLES: usize = MAX_ENCODER_BUFFER * 2;
const MAX_PACKET_BYTES: i32 = if cfg!(feature = "enable_qext") {
    3825 * 6
} else {
    1276 * 6
};
const MAX_REPACKETIZER_BYTES: usize = MAX_PACKET_BYTES as usize + 6;
const MAX_FRAME_SAMPLES: usize = if cfg!(feature = "enable_qext") {
    11520
} else {
    5760
};
const MAX_DELAY_COMPENSATION_SAMPLES: usize = if cfg!(feature = "enable_qext") {
    384
} else {
    192
};
const MAX_PCM_BUF_SAMPLES: usize =
    (MAX_FRAME_SAMPLES + MAX_DELAY_COMPENSATION_SAMPLES) * MAX_CHANNELS;
const MAX_TMP_PREFILL_SAMPLES: usize = (MAX_ENCODER_BUFFER / 4) * MAX_CHANNELS;

#[cfg(test)]
mod opus_mode_trace {
    extern crate std;

    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::env;
    use std::sync::OnceLock;

    use crate::celt::AnalysisInfo;

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn begin_frame() -> Option<usize> {
        if config().is_some() {
            Some(FRAME_INDEX.fetch_add(1, Ordering::Relaxed))
        } else {
            None
        }
    }

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = match env::var("OPUS_TRACE_MODE") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                if !enabled {
                    return None;
                }
                let frame = env::var("OPUS_TRACE_MODE_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                Some(TraceConfig { frame })
            })
            .as_ref()
    }

    pub(crate) fn dump_if_match(
        frame_idx: usize,
        mode: i32,
        prev_mode: i32,
        equiv_rate: i32,
        bandwidth_int: i32,
        stream_channels: i32,
        voice_ratio: i32,
        is_silence: bool,
        analysis: &AnalysisInfo,
    ) {
        let Some(cfg) = config() else {
            return;
        };
        if let Some(frame) = cfg.frame {
            if frame != frame_idx {
                return;
            }
        }
        crate::test_trace::trace_println!("opus_mode[{frame_idx}].mode={mode}");
        crate::test_trace::trace_println!("opus_mode[{frame_idx}].prev_mode={prev_mode}");
        crate::test_trace::trace_println!("opus_mode[{frame_idx}].equiv_rate={equiv_rate}");
        crate::test_trace::trace_println!("opus_mode[{frame_idx}].bandwidth={bandwidth_int}");
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].stream_channels={stream_channels}"
        );
        crate::test_trace::trace_println!("opus_mode[{frame_idx}].voice_ratio={voice_ratio}");
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].is_silence={}",
            if is_silence { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.valid={}",
            if analysis.valid { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.bandwidth={}",
            analysis.bandwidth
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.activity={:.9e}",
            analysis.activity as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.tonality={:.9e}",
            analysis.tonality as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.tonality_slope={:.9e}",
            analysis.tonality_slope as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.noisiness={:.9e}",
            analysis.noisiness as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.music_prob={:.9e}",
            analysis.music_prob as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.activity_probability={:.9e}",
            analysis.activity_probability as f64
        );
        crate::test_trace::trace_println!(
            "opus_mode[{frame_idx}].analysis.max_pitch_ratio={:.9e}",
            analysis.max_pitch_ratio as f64
        );
    }
}

#[cfg(test)]
mod opus_pcm_trace {
    extern crate std;

    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::env;
    use std::sync::OnceLock;

    use crate::celt::OpusRes;

    #[derive(Clone, Copy)]
    pub(crate) struct TraceConfig {
        pub frame: Option<usize>,
        pub want_bits: bool,
        pub start: usize,
        pub count: usize,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);
    static CURRENT_FRAME: AtomicUsize = AtomicUsize::new(usize::MAX);

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = match env::var("OPUS_TRACE_PCM_DUMP") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                if !enabled {
                    return None;
                }
                let frame = env::var("OPUS_TRACE_PCM_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let want_bits = match env::var("OPUS_TRACE_PCM_BITS") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                let start = env::var("OPUS_TRACE_PCM_START")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                let count = env::var("OPUS_TRACE_PCM_COUNT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(64);
                Some(TraceConfig {
                    frame,
                    want_bits,
                    start,
                    count,
                })
            })
            .as_ref()
    }

    pub(crate) fn begin_frame() -> Option<usize> {
        if config().is_some() {
            let idx = FRAME_INDEX.fetch_add(1, Ordering::Relaxed);
            CURRENT_FRAME.store(idx, Ordering::Relaxed);
            Some(idx)
        } else {
            None
        }
    }

    pub(crate) fn current_frame() -> Option<usize> {
        if config().is_some() {
            let idx = CURRENT_FRAME.load(Ordering::Relaxed);
            if idx == usize::MAX { None } else { Some(idx) }
        } else {
            None
        }
    }

    pub(crate) fn config_copy() -> Option<TraceConfig> {
        config().copied()
    }

    fn sample_bits(value: OpusRes) -> u32 {
        value.to_bits()
    }

    fn sample_value(value: OpusRes) -> f64 {
        value as f64
    }

    pub(crate) fn dump(tag: &str, frame_idx: usize, pcm: &[OpusRes], channels: usize, len: usize) {
        let cfg = match config() {
            Some(cfg) => cfg,
            None => return,
        };
        if channels == 0 {
            return;
        }
        if let Some(frame) = cfg.frame {
            if frame != frame_idx {
                return;
            }
        }
        let max_len = pcm.len() / channels;
        let len = len.min(max_len);
        let start = cfg.start.min(len);
        let end = start.saturating_add(cfg.count).min(len);
        crate::test_trace::trace_println!("opus_pcm[{frame_idx}].{tag}.len={len}");
        for ch in 0..channels {
            for i in start..end {
                let idx = i * channels + ch;
                if idx >= pcm.len() {
                    continue;
                }
                let value = pcm[idx];
                crate::test_trace::trace_println!(
                    "opus_pcm[{frame_idx}].{tag}.ch[{ch}].sample[{i}]={:.9e}",
                    sample_value(value)
                );
                if cfg.want_bits {
                    crate::test_trace::trace_println!(
                        "opus_pcm[{frame_idx}].{tag}.ch[{ch}].sample_bits[{i}]=0x{:08x}",
                        sample_bits(value)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod opus_celt_budget_trace {
    extern crate std;

    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::env;
    use std::sync::OnceLock;

    pub(crate) struct TraceConfig {
        frame: Option<usize>,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();
    static FRAME_INDEX: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn begin_frame() -> Option<usize> {
        if config().is_some() {
            Some(FRAME_INDEX.fetch_add(1, Ordering::Relaxed))
        } else {
            None
        }
    }

    fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = match env::var("OPUS_TRACE_CELT_BUDGET") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                if !enabled {
                    return None;
                }
                let frame = env::var("OPUS_TRACE_CELT_BUDGET_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                Some(TraceConfig { frame })
            })
            .as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dump_if_match(
        frame_idx: usize,
        mode: i32,
        max_data_bytes: i32,
        max_frame_bytes: i32,
        max_payload_bytes: i32,
        frame_size: i32,
        frame_rate: i32,
        equiv_rate: i32,
        bitrate_bps: i32,
        use_vbr: bool,
        vbr_constraint: bool,
        channels: i32,
        stream_channels: i32,
        redundancy: bool,
        celt_to_silk: bool,
        to_celt: bool,
        redundancy_bytes: i32,
        nb_compr_bytes: i32,
        tell_pre: i32,
        tell_frac_pre: i32,
        tell_post: i32,
        tell_frac_post: i32,
    ) {
        let Some(cfg) = config() else {
            return;
        };
        if let Some(frame) = cfg.frame {
            if frame != frame_idx {
                return;
            }
        }
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].mode={mode}");
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].max_data_bytes={max_data_bytes}"
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].max_frame_bytes={max_frame_bytes}"
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].max_payload_bytes={max_payload_bytes}"
        );
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].frame_size={frame_size}");
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].frame_rate={frame_rate}");
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].equiv_rate={equiv_rate}");
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].bitrate_bps={bitrate_bps}"
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].use_vbr={}",
            if use_vbr { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].vbr_constraint={}",
            if vbr_constraint { 1 } else { 0 }
        );
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].channels={channels}");
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].stream_channels={stream_channels}"
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].redundancy={}",
            if redundancy { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].celt_to_silk={}",
            if celt_to_silk { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].to_celt={}",
            if to_celt { 1 } else { 0 }
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].redundancy_bytes={redundancy_bytes}"
        );
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].nb_compr_bytes={nb_compr_bytes}"
        );
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].tell_pre={tell_pre}");
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].tell_frac_pre={tell_frac_pre}"
        );
        crate::test_trace::trace_println!("opus_celt_budget[{frame_idx}].tell_post={tell_post}");
        crate::test_trace::trace_println!(
            "opus_celt_budget[{frame_idx}].tell_frac_post={tell_frac_post}"
        );
    }
}

#[cfg(test)]
mod celt_pcm_trace {
    extern crate std;

    use std::env;
    use std::sync::OnceLock;

    #[derive(Clone, Copy)]
    pub(crate) struct TraceConfig {
        pub frame: Option<usize>,
        pub want_bits: bool,
        pub start: usize,
        pub count: usize,
    }

    static TRACE_CONFIG: OnceLock<Option<TraceConfig>> = OnceLock::new();

    pub(crate) fn config() -> Option<&'static TraceConfig> {
        TRACE_CONFIG
            .get_or_init(|| {
                let enabled = match env::var("OPUS_TRACE_PCM_DUMP") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                if !enabled {
                    return None;
                }
                let frame = env::var("CELT_TRACE_PCM_FRAME")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok());
                let want_bits = match env::var("CELT_TRACE_PCM_BITS") {
                    Ok(value) => !value.is_empty() && value != "0",
                    Err(_) => false,
                };
                let start = env::var("CELT_TRACE_PCM_START")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                let count = env::var("CELT_TRACE_PCM_COUNT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(64);
                Some(TraceConfig {
                    frame,
                    want_bits,
                    start,
                    count,
                })
            })
            .as_ref()
    }

    pub(crate) fn config_copy() -> Option<TraceConfig> {
        config().copied()
    }

    pub(crate) fn dump(
        tag: &str,
        pcm: &[i16],
        frame_size: usize,
        channels: usize,
        start: usize,
        count: usize,
        want_bits: bool,
        frame_idx: usize,
    ) {
        if channels == 0 {
            return;
        }
        let len = frame_size;
        let start = start.min(len);
        let end = start.saturating_add(count).min(len);
        crate::test_trace::trace_println!("celt_pcm[{}].{}.len={}", frame_idx, tag, len);
        for ch in 0..channels {
            for i in start..end {
                let idx = i * channels + ch;
                if idx >= pcm.len() {
                    continue;
                }
                let value = pcm[idx];
                crate::test_trace::trace_println!(
                    "celt_pcm[{}].{}.ch[{}].sample[{}]={}",
                    frame_idx,
                    tag,
                    ch,
                    i,
                    value
                );
                if want_bits {
                    let bits = value as u16;
                    crate::test_trace::trace_println!(
                        "celt_pcm[{}].{}.ch[{}].sample_bits[{}]=0x{:04x}",
                        frame_idx,
                        tag,
                        ch,
                        i,
                        bits
                    );
                }
            }
        }
    }
}

pub(crate) const MODE_SILK_ONLY: i32 = 1000;
#[allow(dead_code)]
pub(crate) const MODE_HYBRID: i32 = 1001;
pub(crate) const MODE_CELT_ONLY: i32 = 1002;
pub(crate) const OPUS_SIGNAL_VOICE: i32 = 3001;
pub(crate) const OPUS_SIGNAL_MUSIC: i32 = 3002;
pub(crate) const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub(crate) const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub(crate) const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub(crate) const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub(crate) const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;
pub(crate) const OPUS_FRAMESIZE_ARG: i32 = 5000;
pub(crate) const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
pub(crate) const OPUS_FRAMESIZE_5_MS: i32 = 5002;
pub(crate) const OPUS_FRAMESIZE_10_MS: i32 = 5003;
pub(crate) const OPUS_FRAMESIZE_20_MS: i32 = 5004;
pub(crate) const OPUS_FRAMESIZE_40_MS: i32 = 5005;
pub(crate) const OPUS_FRAMESIZE_60_MS: i32 = 5006;
pub(crate) const OPUS_FRAMESIZE_80_MS: i32 = 5007;
pub(crate) const OPUS_FRAMESIZE_100_MS: i32 = 5008;
pub(crate) const OPUS_FRAMESIZE_120_MS: i32 = 5009;

const DRED_MAX_FRAMES: i32 = 104;
#[cfg(feature = "dred")]
const DRED_ACTIVITY_MEM_LEN: usize = DRED_MAX_FRAMES as usize * 4;
#[cfg(feature = "dred")]
const DRED_NUM_REDUNDANCY_FRAMES: i32 = DRED_MAX_FRAMES / 2;
#[cfg(feature = "dred")]
const DRED_EXPERIMENTAL_BYTES: i32 = 2;
#[cfg(feature = "dred")]
const DRED_EXTENSION_ID: u8 = 126;
#[cfg(feature = "dred")]
const DRED_EXPERIMENTAL_VERSION: u8 = 12;
#[cfg(feature = "dred")]
const DRED_BITS_TABLE: [f32; 16] = [
    73.2, 68.1, 62.5, 57.0, 51.5, 45.7, 39.9, 32.4, 26.4, 20.4, 16.3, 13.0, 9.3, 8.2, 7.2, 6.4,
];
const FEC_THRESHOLDS: [i32; 10] = [
    12_000, 1_000, 14_000, 1_000, 16_000, 1_000, 20_000, 1_000, 22_000, 1_000,
];
const FEC_RATE_SCALE_Q16: i32 = 655;
#[allow(dead_code)]
const SILK_RATE_TABLE: [[i32; 5]; 7] = [
    [0, 0, 0, 0, 0],
    [12_000, 10_000, 10_000, 11_000, 11_000],
    [16_000, 13_500, 13_500, 15_000, 15_000],
    [20_000, 16_000, 16_000, 18_000, 18_000],
    [24_000, 18_000, 18_000, 21_000, 21_000],
    [32_000, 22_000, 22_000, 28_000, 28_000],
    [64_000, 38_000, 38_000, 50_000, 50_000],
];
const MONO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9_000, 700, 9_000, 700, 13_500, 1_000, 14_000, 2_000];
const MONO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9_000, 700, 9_000, 700, 11_000, 1_000, 12_000, 2_000];
const STEREO_VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9_000, 700, 9_000, 700, 13_500, 1_000, 14_000, 2_000];
const STEREO_MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] =
    [9_000, 700, 9_000, 700, 11_000, 1_000, 12_000, 2_000];
const STEREO_VOICE_THRESHOLD: i32 = 19_000;
const STEREO_MUSIC_THRESHOLD: i32 = 17_000;
const MODE_THRESHOLDS: [[i32; 2]; 2] = [[64_000, 10_000], [44_000, 10_000]];
const Q15_ONE: i32 = 32767;
const VARIABLE_HP_SMTH_COEF2_Q16: i32 = (VARIABLE_HP_SMTH_COEF2 * ((1 << 16) as f32) + 0.5) as i32;
const HP_CUTOFF_COEF_Q19: i32 =
    (1.5 * core::f32::consts::PI / 1000.0 * ((1 << 19) as f32) + 0.5) as i32;
const HP_CUTOFF_R_COEF_Q9: i32 = (0.92 * ((1 << 9) as f32) + 0.5) as i32;
const VERY_SMALL: f32 = 1.0e-30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    Voip,
    Audio,
    RestrictedLowDelay,
}

impl OpusApplication {
    #[inline]
    fn from_opus_int(value: i32) -> Option<Self> {
        match value {
            2048 => Some(Self::Voip),
            2049 => Some(Self::Audio),
            2051 => Some(Self::RestrictedLowDelay),
            _ => None,
        }
    }

    #[inline]
    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::RestrictedLowDelay => 2051,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusEncoderInitError {
    BadArgument,
    SilkInit,
    CeltInit,
}

impl OpusEncoderInitError {
    #[inline]
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::SilkInit | Self::CeltInit => -3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusEncoderCtlError {
    BadArgument,
    Unimplemented,
    Silk(SilkError),
    InternalError,
}

impl OpusEncoderCtlError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::Unimplemented => -5,
            Self::Silk(_) | Self::InternalError => -3,
        }
    }
}

impl From<SilkError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: SilkError) -> Self {
        Self::Silk(value)
    }
}

impl From<CeltEncoderInitError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: CeltEncoderInitError) -> Self {
        let _ = value;
        Self::InternalError
    }
}

impl From<CeltEncoderCtlError> for OpusEncoderCtlError {
    #[inline]
    fn from(value: CeltEncoderCtlError) -> Self {
        let _ = value;
        Self::InternalError
    }
}

pub enum OpusEncoderCtlRequest<'req> {
    SetApplication(i32),
    GetApplication(&'req mut i32),
    SetBitrate(i32),
    GetBitrate(&'req mut i32),
    SetForceChannels(i32),
    GetForceChannels(&'req mut i32),
    SetMaxBandwidth(i32),
    GetMaxBandwidth(&'req mut i32),
    SetBandwidth(i32),
    GetBandwidth(&'req mut i32),
    SetVbr(bool),
    GetVbr(&'req mut bool),
    SetVbrConstraint(bool),
    GetVbrConstraint(&'req mut bool),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    SetSignal(i32),
    GetSignal(&'req mut i32),
    SetVoiceRatio(i32),
    GetVoiceRatio(&'req mut i32),
    SetPacketLossPerc(i32),
    GetPacketLossPerc(&'req mut i32),
    SetInbandFec(bool),
    GetInbandFec(&'req mut bool),
    #[cfg(feature = "enable_qext")]
    SetQext(bool),
    #[cfg(feature = "enable_qext")]
    GetQext(&'req mut bool),
    SetDtx(bool),
    GetDtx(&'req mut bool),
    GetInDtx(&'req mut bool),
    SetLsbDepth(i32),
    GetLsbDepth(&'req mut i32),
    SetExpertFrameDuration(i32),
    GetExpertFrameDuration(&'req mut i32),
    SetPredictionDisabled(bool),
    GetPredictionDisabled(&'req mut bool),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
    SetDredDuration(i32),
    GetDredDuration(&'req mut i32),
    SetDnnBlob(&'req [u8]),
    SetForceMode(i32),
    GetSampleRate(&'req mut i32),
    GetLookahead(&'req mut i32),
    GetFinalRange(&'req mut u32),
    ResetState,
    SetLfe(bool),
    GetLfe(&'req mut bool),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OpusEncodeOptions<'a> {
    pub energy_masking: Option<&'a [f32]>,
}

/// Original channels used for analysis when a multistream encoder mixes PCM.
pub(crate) struct OpusAnalysisInput<'a> {
    pub pcm: &'a [f32],
    pub channels: i32,
    pub c1: i32,
    pub c2: i32,
    pub depth: i32,
}

pub(crate) fn opus_encode_float_with_analysis(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
    analysis: OpusAnalysisInput<'_>,
) -> Result<usize, OpusEncodeError> {
    opus_encode_native_with_options(encoder, pcm, frame_size, data, options, Some(analysis))
}

pub(crate) fn opus_encode24_with_analysis(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
    analysis: OpusAnalysisInput<'_>,
) -> Result<usize, OpusEncodeError> {
    opus_encode_native_with_options(encoder, pcm, frame_size, data, options, Some(analysis))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusEncodeError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    Unimplemented,
    Silk(SilkError),
}

impl OpusEncodeError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError | Self::Silk(_) => -3,
            Self::Unimplemented => -5,
        }
    }
}

impl From<SilkError> for OpusEncodeError {
    #[inline]
    fn from(value: SilkError) -> Self {
        Self::Silk(value)
    }
}

#[inline]
fn align(value: usize) -> usize {
    #[repr(C)]
    struct AlignProbe {
        _tag: u8,
        _union: AlignUnion,
    }

    #[repr(C)]
    union AlignUnion {
        _ptr: *const (),
        _i32: i32,
        _f32: f32,
    }

    let alignment = core::mem::align_of::<AlignProbe>();
    value.div_ceil(alignment) * alignment
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct StereoWidthState {
    xx: f32,
    xy: f32,
    yy: f32,
    smoothed_width: f32,
    max_follower: f32,
    #[cfg(feature = "fixed_point")]
    fixed: [i32; 5],
}

#[repr(C)]
struct OpusEncoderLayout {
    celt_enc_offset: i32,
    silk_enc_offset: i32,
    silk_mode: SilkEncControlLayout,
    #[cfg(feature = "dred")]
    dred_encoder: DredEnc,
    application: i32,
    channels: i32,
    delay_compensation: i32,
    force_channels: i32,
    signal_type: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    fs: i32,
    use_vbr: i32,
    vbr_constraint: i32,
    variable_duration: i32,
    bitrate_bps: i32,
    user_bitrate_bps: i32,
    lsb_depth: i32,
    encoder_buffer: i32,
    lfe: i32,
    arch: i32,
    use_dtx: i32,
    #[cfg(feature = "enable_qext")]
    enable_qext: bool,
    fec_config: i32,
    analysis: TonalityAnalysisState,
    stream_channels: i32,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: f32,
    hp_mem: [f32; 4],
    #[cfg(feature = "fixed_point")]
    fixed_hp_mem: [i32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: i32,
    auto_bandwidth: i32,
    silk_bw_switch: i32,
    first: i32,
    width_mem: StereoWidthState,
    delay_buffer: [OpusRes; DELAY_BUFFER_SAMPLES],
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    native_delay_buffer: [i32; DELAY_BUFFER_SAMPLES],
    detected_bandwidth: i32,
    nb_no_activity_ms_q1: i32,
    peak_signal_energy: f32,
    #[cfg(feature = "fixed_point")]
    fixed_peak_signal_energy: i32,
    #[cfg(feature = "dred")]
    dred_duration: i32,
    #[cfg(feature = "dred")]
    dred_q0: i32,
    #[cfg(feature = "dred")]
    dred_dq: i32,
    #[cfg(feature = "dred")]
    dred_qmax: i32,
    #[cfg(feature = "dred")]
    dred_target_chunks: i32,
    #[cfg(feature = "dred")]
    activity_mem: [u8; DRED_ACTIVITY_MEM_LEN],
    nonfinal_frame: i32,
    range_final: u32,
}

#[repr(C)]
struct SilkEncControlLayout {
    n_channels_api: i32,
    n_channels_internal: i32,
    api_sample_rate: i32,
    max_internal_sample_rate: i32,
    min_internal_sample_rate: i32,
    desired_internal_sample_rate: i32,
    payload_size_ms: i32,
    bit_rate: i32,
    packet_loss_percentage: i32,
    complexity: i32,
    use_in_band_fec: i32,
    use_dred: i32,
    lbrr_coded: i32,
    use_dtx: i32,
    use_cbr: i32,
    max_bits: i32,
    to_mono: bool,
    opus_can_switch: bool,
    reduced_dependency: bool,
    internal_sample_rate: i32,
    allow_bandwidth_switch: bool,
    in_wb_mode_without_variable_lp: bool,
    stereo_width_q14: i32,
    switch_ready: bool,
    signal_type: i32,
    offset: i32,
}

#[must_use]
pub fn opus_encoder_get_size(channels: usize) -> Option<usize> {
    if channels == 0 || channels > MAX_CHANNELS {
        return None;
    }

    let mut silk_size = 0usize;
    crate::silk::get_encoder_size::get_encoder_size(&mut silk_size).ok()?;
    let silk_size = align(silk_size);

    let celt_size = crate::celt::celt_encoder_get_size(channels)?;
    let header_size = align(core::mem::size_of::<OpusEncoderLayout>());

    Some(header_size + silk_size + celt_size)
}

#[derive(Debug)]
pub struct OpusEncoder<'mode> {
    celt: OwnedCeltEncoder<'mode>,
    silk: crate::silk::encoder::state::Encoder,
    silk_mode: SilkEncControl,
    #[cfg(feature = "dred")]
    dred_encoder: Box<DredEnc>,
    analysis: TonalityAnalysisState,
    analysis_info: AnalysisInfo,
    application: OpusApplication,
    channels: i32,
    stream_channels: i32,
    fs: i32,
    arch: i32,
    use_vbr: bool,
    vbr_constraint: bool,
    user_bitrate_bps: i32,
    bitrate_bps: i32,
    packet_loss_perc: i32,
    complexity: i32,
    inband_fec: bool,
    use_dtx: bool,
    #[cfg(feature = "enable_qext")]
    enable_qext: bool,
    fec_config: i32,
    force_channels: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    signal_type: i32,
    user_forced_mode: i32,
    voice_ratio: i32,
    delay_compensation: i32,
    encoder_buffer: i32,
    lsb_depth: i32,
    variable_duration: i32,
    prediction_disabled: bool,
    hybrid_stereo_width_q14: i16,
    variable_hp_smth2_q15: i32,
    prev_hb_gain: f32,
    hp_mem: [f32; 4],
    #[cfg(feature = "fixed_point")]
    fixed_hp_mem: [i32; 4],
    mode: i32,
    prev_mode: i32,
    prev_channels: i32,
    prev_framesize: i32,
    bandwidth: Bandwidth,
    auto_bandwidth: i32,
    silk_bw_switch: bool,
    first: bool,
    width_mem: StereoWidthState,
    delay_buffer: [OpusRes; DELAY_BUFFER_SAMPLES],
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    native_delay_buffer: [i32; DELAY_BUFFER_SAMPLES],
    detected_bandwidth: i32,
    nb_no_activity_ms_q1: i32,
    peak_signal_energy: f32,
    #[cfg(feature = "fixed_point")]
    fixed_peak_signal_energy: i32,
    nonfinal_frame: bool,
    range_final: u32,
    dred_duration: i32,
    #[cfg(feature = "dred")]
    dred_loaded: bool,
    #[cfg(feature = "dred")]
    dred_latents_buffer_fill: i32,
    #[cfg(feature = "dred")]
    dred_q0: i32,
    #[cfg(feature = "dred")]
    dred_dq: i32,
    #[cfg(feature = "dred")]
    dred_qmax: i32,
    #[cfg(feature = "dred")]
    dred_target_chunks: i32,
    #[cfg(feature = "dred")]
    dred_activity_mem: [u8; DRED_ACTIVITY_MEM_LEN],
    lfe: bool,
}

impl<'mode> OpusEncoder<'mode> {
    #[cfg(feature = "fixed_point")]
    pub(crate) fn set_fixed_energy_mask(&mut self, mask: Option<Vec<i32>>) {
        self.celt.encoder().owned_fixed_energy_mask = mask;
    }

    pub fn init(
        &mut self,
        fs: i32,
        channels: i32,
        application: i32,
    ) -> Result<(), OpusEncoderInitError> {
        if !(matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000)
            || cfg!(feature = "enable_qext") && fs == 96_000)
            || !matches!(channels, 1 | 2)
        {
            return Err(OpusEncoderInitError::BadArgument);
        }
        let application =
            OpusApplication::from_opus_int(application).ok_or(OpusEncoderInitError::BadArgument)?;

        let mode = {
            #[cfg(feature = "enable_qext")]
            if fs == 96_000 {
                crate::celt::canonical_mode_96k()
            } else {
                canonical_mode()
            }
            #[cfg(not(feature = "enable_qext"))]
            canonical_mode()
        }
        .ok_or(OpusEncoderInitError::CeltInit)?;
        self.celt = opus_custom_encoder_create(mode, fs, channels as usize, 0)
            .map_err(|_| OpusEncoderInitError::CeltInit)?;

        self.arch = opus_select_arch();
        self.channels = channels;
        self.stream_channels = channels;
        self.fs = fs;
        self.application = application;

        // Reset SILK encoder.
        silk_init_encoder(&mut self.silk, self.arch, &mut self.silk_mode)
            .map_err(|_| OpusEncoderInitError::SilkInit)?;

        // Default SILK parameters from `opus_encoder_init`.
        self.silk_mode.n_channels_api = channels;
        self.silk_mode.n_channels_internal = channels;
        self.silk_mode.api_sample_rate = fs;
        self.silk_mode.max_internal_sample_rate = 16_000;
        self.silk_mode.min_internal_sample_rate = 8_000;
        self.silk_mode.desired_internal_sample_rate = 16_000;
        self.silk_mode.payload_size_ms = 20;
        self.silk_mode.bit_rate = 25_000;
        self.silk_mode.packet_loss_percentage = 0;
        self.silk_mode.complexity = 9;
        self.silk_mode.use_in_band_fec = 0;
        self.silk_mode.use_dred = 0;
        self.silk_mode.use_dtx = 0;
        self.silk_mode.use_cbr = 0;
        self.silk_mode.reduced_dependency = false;
        self.dred_duration = 0;
        #[cfg(feature = "dred")]
        {
            self.dred_loaded = false;
            self.dred_latents_buffer_fill = 0;
            self.dred_q0 = 0;
            self.dred_dq = 0;
            self.dred_qmax = 0;
            self.dred_target_chunks = 0;
            self.dred_activity_mem = [0; DRED_ACTIVITY_MEM_LEN];
        }

        // Keep CELT's signalling disabled for later frame packing.
        opus_custom_encoder_ctl(self.celt.encoder(), CeltEncoderCtlRequest::SetSignalling(0))
            .map_err(|_| OpusEncoderInitError::CeltInit)?;
        opus_custom_encoder_ctl(
            self.celt.encoder(),
            CeltEncoderCtlRequest::SetComplexity(self.silk_mode.complexity),
        )
        .map_err(|_| OpusEncoderInitError::CeltInit)?;

        #[cfg(feature = "dred")]
        {
            dred_encoder_init(&mut self.dred_encoder, fs, channels);
            self.dred_loaded = self.dred_encoder.loaded;
        }

        #[cfg(feature = "enable_qext")]
        {
            self.enable_qext = false;
        }
        self.use_vbr = true;
        self.vbr_constraint = true;
        self.user_bitrate_bps = OPUS_AUTO;
        self.bitrate_bps = 3000 + fs * channels;
        self.packet_loss_perc = 0;
        self.complexity = self.silk_mode.complexity;
        self.inband_fec = false;
        self.use_dtx = false;
        self.fec_config = 0;
        self.force_channels = OPUS_AUTO;
        self.user_bandwidth = OPUS_AUTO;
        self.max_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.signal_type = OPUS_AUTO;
        self.user_forced_mode = OPUS_AUTO;
        self.voice_ratio = -1;
        self.encoder_buffer = fs / 100;
        self.lsb_depth = 24;
        self.variable_duration = OPUS_FRAMESIZE_ARG;
        self.delay_compensation = fs / 250;
        self.prediction_disabled = false;

        self.hybrid_stereo_width_q14 = 1_i16 << 14;
        self.variable_hp_smth2_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        self.prev_hb_gain = 1.0;
        self.hp_mem = [0.0; 4];
        #[cfg(feature = "fixed_point")]
        {
            self.fixed_hp_mem = [0; 4];
        }
        self.mode = MODE_HYBRID;
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.bandwidth = Bandwidth::Full;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = false;
        self.first = true;
        self.width_mem = StereoWidthState::default();
        self.delay_buffer = [OpusRes::default(); DELAY_BUFFER_SAMPLES];
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        {
            self.native_delay_buffer.fill(0);
        }
        {
            self.detected_bandwidth = 0;
            self.nb_no_activity_ms_q1 = 0;
            self.peak_signal_energy = 0.0;
            #[cfg(feature = "fixed_point")]
            {
                self.fixed_peak_signal_energy = 0;
            }
        }
        self.nonfinal_frame = false;
        self.range_final = 0;
        self.dred_duration = 0;
        self.lfe = false;

        {
            tonality_analysis_init(&mut self.analysis, fs);
        }
        self.analysis_info = AnalysisInfo::default();

        Ok(())
    }

    fn reset_state(&mut self) -> Result<(), OpusEncoderCtlError> {
        let mut dummy_control = SilkEncControl::default();
        silk_init_encoder(&mut self.silk, self.arch, &mut dummy_control)?;
        opus_custom_encoder_ctl(self.celt.encoder(), CeltEncoderCtlRequest::ResetState)?;
        #[cfg(feature = "dred")]
        {
            dred_encoder_reset(&mut self.dred_encoder);
            self.dred_loaded = self.dred_encoder.loaded;
        }
        {
            tonality_analysis_reset(&mut self.analysis);
        }
        self.analysis_info = AnalysisInfo::default();
        #[cfg(feature = "dred")]
        {
            self.dred_latents_buffer_fill = 0;
            self.dred_q0 = 0;
            self.dred_dq = 0;
            self.dred_qmax = 0;
            self.dred_target_chunks = 0;
            self.dred_activity_mem = [0; DRED_ACTIVITY_MEM_LEN];
        }
        self.stream_channels = self.channels;
        self.hybrid_stereo_width_q14 = 1_i16 << 14;
        self.prev_hb_gain = 1.0;
        self.first = true;
        self.mode = MODE_HYBRID;
        self.bandwidth = Bandwidth::Full;
        self.variable_hp_smth2_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        self.hp_mem = [0.0; 4];
        #[cfg(feature = "fixed_point")]
        {
            self.fixed_hp_mem = [0; 4];
        }
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = false;
        self.width_mem = StereoWidthState::default();
        self.delay_buffer = [OpusRes::default(); DELAY_BUFFER_SAMPLES];
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        {
            self.native_delay_buffer.fill(0);
        }
        {
            self.detected_bandwidth = 0;
            self.nb_no_activity_ms_q1 = 0;
            self.peak_signal_energy = 0.0;
            #[cfg(feature = "fixed_point")]
            {
                self.fixed_peak_signal_energy = 0;
            }
        }
        self.nonfinal_frame = false;
        self.range_final = 0;
        Ok(())
    }

    fn configure_silk_control(&mut self, frame_size: i32, max_data_bytes: usize) {
        // Frame size in milliseconds is derived from the API-level sampling rate.
        let payload_size_ms = (1000i64 * i64::from(frame_size) / i64::from(self.fs)) as i32;
        self.silk_mode.payload_size_ms = payload_size_ms;
        self.silk_mode.n_channels_api = self.channels;
        self.silk_mode.n_channels_internal = self.stream_channels;
        self.silk_mode.api_sample_rate = self.fs;
        self.silk_mode.max_internal_sample_rate = 16_000;
        self.silk_mode.min_internal_sample_rate = 8_000;
        self.silk_mode.desired_internal_sample_rate = 16_000;
        self.silk_mode.packet_loss_percentage = self.packet_loss_perc;
        self.silk_mode.complexity = self.complexity;
        self.silk_mode.use_in_band_fec = i32::from(self.inband_fec);
        self.silk_mode.use_dtx = i32::from(self.use_dtx);
        self.silk_mode.use_cbr = i32::from(!self.use_vbr);
        self.silk_mode.reduced_dependency = self.prediction_disabled;
        self.silk_mode.opus_can_switch = false;
        self.silk_mode.max_bits = (max_data_bytes.saturating_mul(8)).min(i32::MAX as usize) as i32;

        let max_internal =
            max_internal_sample_rate_for_bandwidth(self.user_bandwidth, self.max_bandwidth);
        self.silk_mode.max_internal_sample_rate = max_internal;
        self.silk_mode.desired_internal_sample_rate = max_internal;

        let target_bits = (i64::from(self.bitrate_bps) * i64::from(frame_size)
            / i64::from(self.fs))
        .min((max_data_bytes as i64 + 1) * 8)
            - 8;
        self.silk_mode.bit_rate = (target_bits * i64::from(self.fs) / i64::from(frame_size)) as i32;
    }

    fn bandwidth_from_silk_control(control: &SilkEncControl) -> Bandwidth {
        match control.internal_sample_rate {
            8_000 => Bandwidth::Narrow,
            12_000 => Bandwidth::Medium,
            _ => Bandwidth::Wide,
        }
    }
}

fn gen_toc(mode: i32, framerate: i32, bandwidth: Bandwidth, channels: i32) -> u8 {
    let mut framerate = framerate;
    let mut period = 0i32;
    while framerate < 400 {
        framerate <<= 1;
        period += 1;
    }

    let bw_int = bandwidth.to_opus_int();
    let mut toc = if mode == MODE_SILK_ONLY {
        let bw_index = (bw_int - Bandwidth::Narrow.to_opus_int()).clamp(0, 3);
        let period_index = (period - 2).clamp(0, 3);
        ((bw_index as u8) << 5) | ((period_index as u8) << 3)
    } else if mode == MODE_CELT_ONLY {
        let mut tmp = bw_int - Bandwidth::Medium.to_opus_int();
        if tmp < 0 {
            tmp = 0;
        }
        let period_index = period.clamp(0, 3);
        0x80 | ((tmp as u8) << 5) | ((period_index as u8) << 3)
    } else {
        // Hybrid
        let bw_flag = if bandwidth == Bandwidth::Full { 1 } else { 0 };
        let period_index = (period - 2).clamp(0, 3);
        0x60 | ((bw_flag as u8) << 4) | ((period_index as u8) << 3)
    };

    if channels == 2 {
        toc |= 0x04;
    }
    toc
}

pub(crate) fn frame_size_select(frame_size: i32, variable_duration: i32, fs: i32) -> Option<i32> {
    if frame_size < fs / 400 {
        return None;
    }

    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if (OPUS_FRAMESIZE_2_5_MS..=OPUS_FRAMESIZE_120_MS).contains(&variable_duration) {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (fs / 400) << (variable_duration - OPUS_FRAMESIZE_2_5_MS)
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * fs / 50
        }
    } else {
        return None;
    };

    if new_size > frame_size {
        return None;
    }

    let valid = 400 * new_size == fs
        || 200 * new_size == fs
        || 100 * new_size == fs
        || 50 * new_size == fs
        || 25 * new_size == fs
        || 50 * new_size == 3 * fs
        || 50 * new_size == 4 * fs
        || 50 * new_size == 5 * fs
        || 50 * new_size == 6 * fs;

    valid.then_some(new_size)
}

fn user_bitrate_to_bitrate(
    encoder: &OpusEncoder<'_>,
    mut frame_size: i32,
    max_data_bytes: i32,
) -> i32 {
    if frame_size == 0 {
        frame_size = encoder.fs / 400;
    }
    if encoder.user_bitrate_bps == OPUS_AUTO {
        let base = i64::from(60 * encoder.fs / frame_size);
        let channels = i64::from(encoder.channels * encoder.fs);
        (base + channels).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    } else if encoder.user_bitrate_bps == OPUS_BITRATE_MAX {
        let bitrate = i64::from(max_data_bytes) * 8 * i64::from(encoder.fs) / i64::from(frame_size);
        bitrate.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    } else {
        encoder.user_bitrate_bps
    }
}

/// PCM precision is preserved until the individual codec needs quantization.
trait PcmSample: Copy {
    #[cfg(feature = "fixed_point")]
    fn fixed_res(self) -> crate::celt::FixedOpusRes {
        crate::celt::float2res(self.normalized())
    }
    const DEPTH: i32;
    fn normalized(self) -> f32;
}

impl PcmSample for i16 {
    const DEPTH: i32 = 16;
    fn normalized(self) -> f32 {
        f32::from(self) * (1.0 / 32768.0)
    }
}

impl PcmSample for f32 {
    const DEPTH: i32 = 24;
    fn normalized(self) -> f32 {
        self
    }
}

impl PcmSample for i32 {
    #[cfg(feature = "fixed_point")]
    fn fixed_res(self) -> crate::celt::FixedOpusRes {
        crate::celt::int24tores(self)
    }
    const DEPTH: i32 = 24;
    fn normalized(self) -> f32 {
        self as f32 * (1.0 / 8388608.0)
    }
}

#[cfg(not(feature = "fixed_point"))]
fn compute_frame_energy<P: PcmSample>(pcm: &[P]) -> f32 {
    let mut energy = 0.0f32;
    for sample in pcm {
        let value = sample.normalized();
        energy += value * value;
    }
    energy / pcm.len() as f32
}

#[cfg(feature = "fixed_point")]
fn compute_frame_energy_fixed<P: PcmSample>(pcm: &[P]) -> i32 {
    let maximum = pcm
        .iter()
        .map(|&sample| i32::from(crate::celt::res2int16(sample.fixed_res())).abs())
        .max()
        .unwrap_or(0);
    let log = |value: i32| 31 - value.max(1).leading_zeros() as i32;
    let shift = (2 * log(1 + maximum) + log(pcm.len() as i32) - 28).max(0);
    let energy = pcm
        .iter()
        .map(|&sample| {
            let value = i32::from(crate::celt::res2int16(sample.fixed_res()));
            (value * value) >> shift
        })
        .sum::<i32>();
    (energy / pcm.len() as i32) << shift
}

#[cfg(any(test, not(feature = "fixed_point")))]
fn compute_stereo_width<P: PcmSample>(
    pcm: &[P],
    frame_size: usize,
    fs: i32,
    mem: &mut StereoWidthState,
) -> f32 {
    let frame_rate = fs / frame_size as i32;
    if frame_rate <= 0 {
        return 0.0;
    }
    let short_alpha = 25.0 / (frame_rate.max(50) as f32);
    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    let mut yy = 0.0f32;

    for i in (0..frame_size.saturating_sub(3)).step_by(4) {
        let mut pxx = 0.0f32;
        let mut pxy = 0.0f32;
        let mut pyy = 0.0f32;

        let x0 = pcm[2 * i].normalized();
        let y0 = pcm[2 * i + 1].normalized();
        pxx += x0 * x0;
        pxy += x0 * y0;
        pyy += y0 * y0;

        let x1 = pcm[2 * i + 2].normalized();
        let y1 = pcm[2 * i + 3].normalized();
        pxx += x1 * x1;
        pxy += x1 * y1;
        pyy += y1 * y1;

        let x2 = pcm[2 * i + 4].normalized();
        let y2 = pcm[2 * i + 5].normalized();
        pxx += x2 * x2;
        pxy += x2 * y2;
        pyy += y2 * y2;

        let x3 = pcm[2 * i + 6].normalized();
        let y3 = pcm[2 * i + 7].normalized();
        pxx += x3 * x3;
        pxy += x3 * y3;
        pyy += y3 * y3;

        xx += pxx;
        xy += pxy;
        yy += pyy;
    }

    if xx >= 1.0e9 || xx.is_nan() || yy >= 1.0e9 || yy.is_nan() {
        xx = 0.0;
        xy = 0.0;
        yy = 0.0;
    }

    mem.xx += short_alpha * (xx - mem.xx);
    mem.xy = (1.0 - short_alpha) * mem.xy + short_alpha * xy;
    mem.yy += short_alpha * (yy - mem.yy);
    mem.xx = mem.xx.max(0.0);
    mem.xy = mem.xy.max(0.0);
    mem.yy = mem.yy.max(0.0);

    const WIDTH_THRESHOLD: f32 = 8.0e-4;
    const EPSILON: f32 = 1.0e-15;
    if mem.xx.max(mem.yy) > WIDTH_THRESHOLD {
        let sqrt_xx = celt_sqrt(mem.xx);
        let sqrt_yy = celt_sqrt(mem.yy);
        let qrrt_xx = celt_sqrt(sqrt_xx);
        let qrrt_yy = celt_sqrt(sqrt_yy);
        mem.xy = mem.xy.min(sqrt_xx * sqrt_yy);
        let corr = frac_div32(mem.xy, EPSILON + sqrt_xx * sqrt_yy);
        let ldiff = (qrrt_xx - qrrt_yy).abs() / (EPSILON + qrrt_xx + qrrt_yy);
        let width = celt_sqrt((1.0 - corr * corr).max(0.0)).min(1.0) * ldiff;
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        mem.max_follower = (mem.max_follower - 0.02 / frame_rate as f32).max(mem.smoothed_width);
    }

    (20.0 * mem.max_follower).min(1.0)
}

/// Integer stereo-width detector, retaining the reference Q15 state between frames.
#[cfg(feature = "fixed_point")]
fn compute_stereo_width_fixed<P: PcmSample>(
    pcm: &[P],
    frame_size: usize,
    fs: i32,
    mem: &mut StereoWidthState,
) -> i32 {
    use crate::celt::math_fixed::{celt_sqrt, frac_div32};
    let frame_rate = fs / frame_size as i32;
    let shift = (31 - (frame_size as u32).leading_zeros()) - 2;
    let alpha = 25 * 32767 / frame_rate.max(50);
    let mult = |a: i32, b: i32| ((i64::from(a) * i64::from(b)) >> 15) as i32;
    let mut energies = [0i32; 3];
    for i in (0..frame_size.saturating_sub(3)).step_by(4) {
        let mut partial = [0i32; 3];
        for j in 0..4 {
            let x = i32::from(crate::celt::res2val16(pcm[2 * (i + j)].fixed_res()));
            let y = i32::from(crate::celt::res2val16(pcm[2 * (i + j) + 1].fixed_res()));
            partial[0] += (x * x) >> 2;
            partial[1] += (x * y) >> 2;
            partial[2] += (y * y) >> 2;
        }
        for j in 0..3 {
            energies[j] += partial[j] >> shift;
        }
    }
    let state = &mut mem.fixed;
    state[0] += mult(alpha, energies[0] - state[0]);
    state[1] = mult(32767 - alpha, state[1]) + mult(alpha, energies[1]);
    state[2] += mult(alpha, energies[2] - state[2]);
    for value in &mut state[..3] {
        *value = (*value).max(0);
    }
    if state[0].max(state[2]) > 210 {
        let sqrt_xx = celt_sqrt(state[0]);
        let sqrt_yy = celt_sqrt(state[2]);
        let qrrt_xx = celt_sqrt(sqrt_xx);
        let qrrt_yy = celt_sqrt(sqrt_yy);
        state[1] = state[1].min(sqrt_xx * sqrt_yy);
        let corr = frac_div32(state[1], 1 + sqrt_xx * sqrt_yy) >> 16;
        let difference = 32767 * (qrrt_xx - qrrt_yy).abs() / (1 + qrrt_xx + qrrt_yy);
        let width = mult(celt_sqrt((1 << 30) - corr * corr).min(32767), difference);
        state[3] += (width - state[3]) / frame_rate;
        state[4] = (state[4] - 655 / frame_rate).max(state[3]);
    }
    (20 * state[4]).min(32767)
}

fn decide_fec(
    use_in_band_fec: bool,
    packet_loss_perc: i32,
    last_fec: bool,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> bool {
    if !use_in_band_fec || packet_loss_perc == 0 || mode == MODE_CELT_ONLY {
        return false;
    }

    let orig_bandwidth = *bandwidth;
    loop {
        let idx = usize::try_from(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND)).unwrap_or(0);
        let mut lbrr_rate_thres_bps = *FEC_THRESHOLDS.get(idx).unwrap_or(&0);
        let hysteresis = *FEC_THRESHOLDS.get(idx + 1).unwrap_or(&0);
        if last_fec {
            lbrr_rate_thres_bps -= hysteresis;
        }
        if !last_fec {
            lbrr_rate_thres_bps += hysteresis;
        }

        let loss_scale = 125 - packet_loss_perc.min(25);
        let scaled = (i64::from(lbrr_rate_thres_bps)
            * i64::from(loss_scale)
            * i64::from(FEC_RATE_SCALE_Q16))
            >> 16;
        lbrr_rate_thres_bps = scaled.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;

        if rate > lbrr_rate_thres_bps {
            return true;
        }
        if packet_loss_perc <= 5 {
            return false;
        }
        if *bandwidth > OPUS_BANDWIDTH_NARROWBAND {
            *bandwidth -= 1;
        } else {
            break;
        }
    }

    *bandwidth = orig_bandwidth;
    false
}

#[allow(dead_code)]
fn compute_silk_rate_for_hybrid(
    mut rate: i32,
    bandwidth: i32,
    frame_20ms: bool,
    vbr: bool,
    fec: bool,
    channels: i32,
) -> i32 {
    rate /= channels;
    let entry = 1 + i32::from(frame_20ms) + 2 * i32::from(fec);

    let mut idx = 1;
    while idx < SILK_RATE_TABLE.len() && SILK_RATE_TABLE[idx][0] <= rate {
        idx += 1;
    }

    let mut silk_rate = if idx == SILK_RATE_TABLE.len() {
        let base = SILK_RATE_TABLE[idx - 1][entry as usize];
        base + (rate - SILK_RATE_TABLE[idx - 1][0]) / 2
    } else {
        let lo = SILK_RATE_TABLE[idx - 1][entry as usize];
        let hi = SILK_RATE_TABLE[idx][entry as usize];
        let x0 = SILK_RATE_TABLE[idx - 1][0];
        let x1 = SILK_RATE_TABLE[idx][0];
        let num = i64::from(lo) * i64::from(x1 - rate) + i64::from(hi) * i64::from(rate - x0);
        (num / i64::from(x1 - x0)) as i32
    };

    if !vbr {
        silk_rate += 100;
    }
    if bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND {
        silk_rate += 300;
    }
    silk_rate *= channels;
    if channels == 2 && rate >= 12_000 {
        silk_rate -= 1_000;
    }
    silk_rate
}

/// Computes the surround masking rate offset for SILK bitrate adjustment.
///
/// This adjusts the SILK bitrate based on the psychoacoustic masking depth
/// computed from the energy_masking array. The masking array has 21 bands per channel.
///
/// Returns the rate offset in bits per second (can be negative).
fn compute_surround_masking_rate_offset(
    energy_masking: &[f32],
    bandwidth: i32,
    channels: i32,
) -> i32 {
    let (end, srate): (usize, i32) = match bandwidth {
        OPUS_BANDWIDTH_NARROWBAND => (13, 8000),
        OPUS_BANDWIDTH_MEDIUMBAND => (15, 12000),
        _ => (17, 16000),
    };

    let channels_usize = channels as usize;
    let mut mask_sum: f32 = 0.0;

    for c in 0..channels_usize {
        for i in 0..end {
            let idx = 21 * c + i;
            if idx < energy_masking.len() {
                let mut mask = energy_masking[idx].clamp(-2.0, 0.5);
                if mask > 0.0 {
                    mask *= 0.5;
                }
                mask_sum += mask;
            }
        }
    }

    let masking_depth = mask_sum / end as f32 * channels_usize as f32 + 0.2;
    (srate as f32 * masking_depth) as i32
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: bool,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = i64::from(bitrate);
    if frame_rate > 50 {
        equiv -= i64::from((40 * channels + 20) * (frame_rate - 50));
    }
    if !vbr {
        equiv -= equiv / 12;
    }
    equiv = equiv * i64::from(90 + complexity) / 100;
    if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * i64::from(loss) / i64::from(6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        equiv -= equiv * i64::from(loss) / i64::from(12 * loss + 20);
    }
    equiv.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[cfg(feature = "dred")]
fn ec_ilog(value: i32) -> i32 {
    if value <= 0 {
        0
    } else {
        32 - (value as u32).leading_zeros() as i32
    }
}

#[cfg(feature = "dred")]
fn compute_quantizer(q0: i32, d_q: i32, qmax: i32, index: i32) -> i32 {
    const DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    let d_q = d_q as usize;
    debug_assert!(d_q < DQ_TABLE.len());
    let step = DQ_TABLE[d_q];
    let quant = q0 + (step * index + 8) / 16;
    quant.min(qmax)
}

#[cfg(feature = "dred")]
fn estimate_dred_bitrate(
    q0: i32,
    d_q: i32,
    qmax: i32,
    duration: i32,
    target_bits: i32,
    target_chunks: &mut i32,
) -> i32 {
    let mut bits = 8.0 * (3 + DRED_EXPERIMENTAL_BYTES) as f32;
    bits += 50.0 + DRED_BITS_TABLE[q0 as usize];

    let dred_chunks = ((duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES / 2);
    *target_chunks = 0;
    for i in 0..dred_chunks {
        let q = compute_quantizer(q0, d_q, qmax, i);
        bits += DRED_BITS_TABLE[q as usize];
        if bits < target_bits as f32 {
            *target_chunks = i + 1;
        }
    }

    libm::floorf(bits + 0.5) as i32
}

#[cfg(feature = "dred")]
fn compute_dred_bitrate(encoder: &mut OpusEncoder<'_>, bitrate_bps: i32, frame_size: i32) -> i32 {
    let (mut dred_frac, bitrate_offset) = if encoder.inband_fec {
        (3.0 * encoder.packet_loss_perc as f32 / 100.0, 20_000)
    } else if encoder.packet_loss_perc > 5 {
        (0.55 + encoder.packet_loss_perc as f32 / 100.0, 12_000)
    } else {
        (12.0 * encoder.packet_loss_perc as f32 / 100.0, 12_000)
    };
    dred_frac = dred_frac.min(if encoder.inband_fec { 0.7 } else { 0.8 });

    dred_frac = dred_frac
        / (dred_frac + (1.0 - dred_frac) * (frame_size as f32 * 50.0) / encoder.fs as f32);

    let rate_delta = bitrate_bps - bitrate_offset;
    let q0 = (51 - 3 * ec_ilog(rate_delta.max(1))).clamp(4, 15);
    let d_q = if rate_delta > 36_000 { 3 } else { 5 };
    let qmax = 15;

    let target_dred_bitrate = (dred_frac * rate_delta as f32) as i32;
    let target_dred_bitrate = target_dred_bitrate.max(0);
    let mut target_chunks = 0;
    let max_dred_bits = if encoder.dred_duration > 0 {
        let target_bits =
            (i64::from(target_dred_bitrate) * i64::from(frame_size) / i64::from(encoder.fs)) as i32;
        estimate_dred_bitrate(
            q0,
            d_q,
            qmax,
            encoder.dred_duration,
            target_bits,
            &mut target_chunks,
        )
    } else {
        target_chunks = 0;
        0
    };

    let mut dred_bitrate = target_dred_bitrate.min(
        (i64::from(max_dred_bits) * i64::from(encoder.fs) / i64::from(frame_size.max(1))) as i32,
    );
    if target_chunks < 2 {
        dred_bitrate = 0;
    }

    encoder.dred_q0 = q0;
    encoder.dred_dq = d_q;
    encoder.dred_qmax = qmax;
    encoder.dred_target_chunks = target_chunks;

    dred_bitrate
}

#[cfg(feature = "dred")]
fn adjust_nb_compr_bytes_for_dred(
    nb_compr_bytes: usize,
    range_tell_bits: i32,
    frame_rate: i32,
    dred_bitrate_bps: i32,
) -> usize {
    if dred_bitrate_bps <= 0 || frame_rate <= 0 {
        return nb_compr_bytes;
    }
    let dred_bytes = dred_bitrate_bps / (frame_rate * 8);
    let min_celt_bytes = ((range_tell_bits + 7) / 8) + 5;
    let max_celt_bytes = (nb_compr_bytes as i32 - dred_bytes * 3 / 4).max(min_celt_bytes);
    nb_compr_bytes.min(max_celt_bytes.max(0) as usize)
}

#[cfg(feature = "dred")]
#[allow(dead_code)]
fn update_dred_activity_history(encoder: &mut OpusEncoder<'_>, activity: i32, frame_size: usize) {
    if encoder.dred_duration > 0 && encoder.dred_loaded {
        let frame_size_400hz = ((frame_size as i32 * 400) / encoder.fs)
            .clamp(0, DRED_ACTIVITY_MEM_LEN as i32) as usize;
        let mem_len = encoder.dred_activity_mem.len();
        if frame_size_400hz < mem_len {
            encoder
                .dred_activity_mem
                .copy_within(0..(mem_len - frame_size_400hz), frame_size_400hz);
        }
        for value in encoder.dred_activity_mem[..frame_size_400hz.min(mem_len)].iter_mut() {
            *value = activity as u8;
        }
    } else {
        encoder.dred_latents_buffer_fill = 0;
        encoder.dred_activity_mem.fill(0);
    }
}

fn is_digital_silence<P: PcmSample>(
    pcm: &[P],
    frame_size: usize,
    channels: usize,
    lsb_depth: i32,
) -> bool {
    let total = frame_size.saturating_mul(channels);
    if pcm.len() < total || lsb_depth <= 0 {
        return false;
    }
    let mut sample_max = 0.0f32;
    for &sample in &pcm[..total] {
        sample_max = sample_max.max(sample.normalized().abs());
    }
    #[cfg(not(feature = "fixed_point"))]
    {
        sample_max <= 1.0 / ((1u32 << lsb_depth.min(30)) as f32)
    }
    #[cfg(feature = "fixed_point")]
    {
        pcm[..total].iter().all(|&sample| sample.fixed_res() == 0)
    }
}

#[allow(dead_code)]
fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    if frame_rate <= 0 {
        return 0;
    }
    let base_bits = i64::from(40 * channels + 20);
    let mut redundancy_rate = i64::from(bitrate_bps) + base_bits * i64::from(200 - frame_rate);
    redundancy_rate = 3 * redundancy_rate / 2;
    let mut redundancy_bytes = redundancy_rate / 1_600;

    let available_bits = i64::from(max_data_bytes) * 8 - 2 * base_bits;
    let denom = i64::from(240 + 48_000 / frame_rate);
    let redundancy_bytes_cap = (available_bits * 240 / denom + base_bits) / 8;
    redundancy_bytes = redundancy_bytes.min(redundancy_bytes_cap);

    if redundancy_bytes > i64::from(4 + 8 * channels) {
        redundancy_bytes = redundancy_bytes.min(257);
    } else {
        redundancy_bytes = 0;
    }

    redundancy_bytes.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn finish_encode(encoder: &mut OpusEncoder<'_>, mode: i32, to_celt: bool, frame_size: i32) {
    encoder.mode = mode;
    encoder.prev_mode = if to_celt { MODE_CELT_ONLY } else { mode };
    encoder.prev_channels = encoder.stream_channels;
    encoder.prev_framesize = frame_size;
    encoder.first = false;
}

fn prepare_pcm_buffer<P: PcmSample>(
    encoder: &OpusEncoder<'_>,
    pcm: &[P],
    frame_size: usize,
    channels: usize,
    scratch: &mut [OpusRes],
) -> Result<usize, OpusEncodeError> {
    let total_buffer = if matches!(encoder.application, OpusApplication::RestrictedLowDelay) {
        0
    } else {
        encoder.delay_compensation
    };
    let total_buffer = usize::try_from(total_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
    let needed_per_channel = total_buffer
        .checked_add(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    let needed = needed_per_channel
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if needed > scratch.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    let delay_len = total_buffer
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if delay_len > 0 {
        let encoder_buffer =
            usize::try_from(encoder.encoder_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
        debug_assert!(encoder_buffer >= total_buffer);
        let delay_start = encoder_buffer.saturating_sub(total_buffer);
        let delay_start = delay_start
            .checked_mul(channels)
            .ok_or(OpusEncodeError::BadArgument)?;
        let delay_end = delay_start
            .checked_add(delay_len)
            .ok_or(OpusEncodeError::BadArgument)?;
        if delay_end > encoder.delay_buffer.len() {
            return Err(OpusEncodeError::BadArgument);
        }
        scratch[..delay_len].copy_from_slice(&encoder.delay_buffer[delay_start..delay_end]);
    }

    #[cfg(test)]
    if total_buffer > 0 {
        if let Some(frame_idx) = opus_pcm_trace::current_frame() {
            opus_pcm_trace::dump(
                "delay_copy",
                frame_idx,
                &scratch[..delay_len],
                channels,
                total_buffer,
            );
        }
    }

    let pcm_len = frame_size
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < pcm_len {
        return Err(OpusEncodeError::BadArgument);
    }
    for (dst, &sample) in scratch[delay_len..delay_len + pcm_len]
        .iter_mut()
        .zip(pcm.iter().take(pcm_len))
    {
        *dst = sample.normalized();
    }

    #[cfg(test)]
    if let Some(cfg) = celt_pcm_trace::config_copy() {
        if cfg.frame.is_none() {
            celt_pcm_trace::dump(
                "prepare_pcm",
                &pcm.iter()
                    .map(|sample| crate::celt::float2int16(sample.normalized()))
                    .collect::<Vec<_>>(),
                frame_size,
                channels,
                cfg.start,
                cfg.count,
                cfg.want_bits,
                0,
            );
        }
    }

    Ok(needed)
}

#[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
fn native_celt_prefill(
    encoder: &OpusEncoder<'_>,
    channels: usize,
    total_buffer: usize,
) -> Vec<i32> {
    let n4 = (encoder.fs / 400) as usize;
    let offset = (encoder.encoder_buffer as usize - total_buffer - n4) * channels;
    encoder.native_delay_buffer[offset..offset + n4 * channels].to_vec()
}

#[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
fn update_native_delay(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    total_buffer: usize,
    channels: usize,
) {
    let buffer = encoder.encoder_buffer as usize;
    let total = frame_size + total_buffer;
    if buffer > total {
        let count = (buffer - total) * channels;
        encoder
            .native_delay_buffer
            .copy_within(frame_size * channels..frame_size * channels + count, 0);
        encoder.native_delay_buffer[count..buffer * channels]
            .copy_from_slice(&pcm[..total * channels]);
    } else {
        let offset = (total - buffer) * channels;
        encoder.native_delay_buffer[..buffer * channels]
            .copy_from_slice(&pcm[offset..offset + buffer * channels]);
    }
}

fn update_delay_buffer(
    encoder: &mut OpusEncoder<'_>,
    pcm_buf: &[OpusRes],
    frame_size: usize,
    total_buffer: usize,
    channels: usize,
) -> Result<(), OpusEncodeError> {
    debug_assert!(channels == 1 || channels == 2);

    let encoder_buffer =
        usize::try_from(encoder.encoder_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
    let encoder_samples = encoder_buffer
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if encoder_samples > encoder.delay_buffer.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    let frame_total = frame_size
        .checked_add(total_buffer)
        .ok_or(OpusEncodeError::BadArgument)?;
    let frame_total_samples = frame_total
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if frame_total_samples > pcm_buf.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    if encoder_buffer > frame_total {
        let move_len = (encoder_buffer - frame_total)
            .checked_mul(channels)
            .ok_or(OpusEncodeError::BadArgument)?;
        let src_start = frame_size
            .checked_mul(channels)
            .ok_or(OpusEncodeError::BadArgument)?;
        let src_end = src_start
            .checked_add(move_len)
            .ok_or(OpusEncodeError::BadArgument)?;
        if src_end > encoder_samples {
            return Err(OpusEncodeError::BadArgument);
        }

        encoder.delay_buffer.copy_within(src_start..src_end, 0);

        let dst_end = move_len
            .checked_add(frame_total_samples)
            .ok_or(OpusEncodeError::BadArgument)?;
        if dst_end > encoder_samples {
            return Err(OpusEncodeError::BadArgument);
        }
        encoder.delay_buffer[move_len..dst_end].copy_from_slice(&pcm_buf[..frame_total_samples]);
    } else {
        let offset = frame_total
            .checked_sub(encoder_buffer)
            .ok_or(OpusEncodeError::BadArgument)?;
        let src_start = offset
            .checked_mul(channels)
            .ok_or(OpusEncodeError::BadArgument)?;
        let src_end = src_start
            .checked_add(encoder_samples)
            .ok_or(OpusEncodeError::BadArgument)?;
        if src_end > pcm_buf.len() {
            return Err(OpusEncodeError::BadArgument);
        }

        encoder.delay_buffer[..encoder_samples].copy_from_slice(&pcm_buf[src_start..src_end]);
    }

    #[cfg(test)]
    if let Some(frame_idx) = opus_pcm_trace::current_frame() {
        opus_pcm_trace::dump(
            "delay_buf",
            frame_idx,
            &encoder.delay_buffer[..encoder_samples],
            channels,
            encoder_buffer,
        );
    }

    Ok(())
}

fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((i64::from(b) * i64::from(c as i16)) >> 16) as i32)
}

fn update_high_pass_state(encoder: &mut OpusEncoder<'_>, mode: i32) -> i32 {
    let hp_freq_smth1 = if mode == MODE_CELT_ONLY {
        lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
    } else {
        encoder.silk.state_fxx[0].common().variable_hp_smth1_q15
    };

    encoder.variable_hp_smth2_q15 = smlawb(
        encoder.variable_hp_smth2_q15,
        hp_freq_smth1 - encoder.variable_hp_smth2_q15,
        VARIABLE_HP_SMTH_COEF2_Q16,
    );

    log2lin(encoder.variable_hp_smth2_q15 >> 8)
}

fn hp_cutoff<P: PcmSample>(
    input: &[P],
    cutoff_hz: i32,
    output: &mut [OpusRes],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    assert!(channels == 1 || channels == 2, "unsupported channel count");
    let expected = len.saturating_mul(channels);
    assert!(input.len() >= expected, "input buffer too small");
    assert!(output.len() >= expected, "output buffer too small");

    debug_assert!(fs > 0);
    let fs_div = fs / 1000;
    debug_assert!(fs_div > 0);
    debug_assert!(cutoff_hz <= i32::MAX / HP_CUTOFF_COEF_Q19);
    let fc_q19 = (HP_CUTOFF_COEF_Q19 * cutoff_hz) / fs_div;
    debug_assert!(fc_q19 > 0 && fc_q19 < 32768);

    let r_q28 = (1 << 28) - (HP_CUTOFF_R_COEF_Q9 * fc_q19);
    let b_q28 = [r_q28, -2 * r_q28, r_q28];
    let r_q22 = r_q28 >> 6;
    let fc_sq_q22 = smulww(fc_q19, fc_q19);
    let a_q28 = [smulww(r_q22, fc_sq_q22 - (2 << 22)), smulww(r_q22, r_q22)];

    let scale_q28 = 1.0 / ((1u64 << 28) as f32);
    let b0 = b_q28[0] as f32 * scale_q28;
    let b1 = b_q28[1] as f32 * scale_q28;
    let b2 = b_q28[2] as f32 * scale_q28;
    let a0 = a_q28[0] as f32 * scale_q28;
    let a1 = a_q28[1] as f32 * scale_q28;

    if channels == 2 {
        let mut s0 = hp_mem[0];
        let mut s1 = hp_mem[1];
        let mut s2 = hp_mem[2];
        let mut s3 = hp_mem[3];
        for i in 0..len {
            let idx = i * 2;
            let x0 = input[idx].normalized();
            let x1 = input[idx + 1].normalized();

            let vout0 = s0 + b0 * x0;
            s0 = s1 - vout0 * a0 + b1 * x0;
            s1 = -vout0 * a1 + b2 * x0 + VERY_SMALL;
            output[idx] = vout0;

            let vout1 = s2 + b0 * x1;
            s2 = s3 - vout1 * a0 + b1 * x1;
            s3 = -vout1 * a1 + b2 * x1 + VERY_SMALL;
            output[idx + 1] = vout1;
        }
        hp_mem[0] = s0;
        hp_mem[1] = s1;
        hp_mem[2] = s2;
        hp_mem[3] = s3;
    } else {
        let mut s0 = hp_mem[0];
        let mut s1 = hp_mem[1];
        for i in 0..len {
            let x = input[i].normalized();
            let vout = s0 + b0 * x;
            s0 = s1 - vout * a0 + b1 * x;
            s1 = -vout * a1 + b2 * x + VERY_SMALL;
            output[i] = vout;
        }
        hp_mem[0] = s0;
        hp_mem[1] = s1;
    }
}

#[cfg(feature = "fixed_point")]
fn filter_pcm_fixed<P: PcmSample>(
    input: &[P],
    output: &mut [OpusRes],
    memory: &mut [i32; 4],
    len: usize,
    channels: usize,
    fs: i32,
    cutoff: i32,
    voip: bool,
) -> Vec<crate::celt::FixedOpusRes> {
    use crate::celt::{RES_SHIFT, int16tores, res2float, res2int16};
    let mut native = alloc::vec![0; len * channels];
    let round_shift = |value: i32, shift: u32| value.wrapping_add(1 << (shift - 1)) >> shift;
    if voip {
        let fc = (HP_CUTOFF_COEF_Q19 * cutoff) / (fs / 1000);
        let r = (1 << 28) - HP_CUTOFF_R_COEF_Q9 * fc;
        let b = [r, -2 * r, r];
        let r22 = r >> 6;
        let a = [smulww(r22, smulww(fc, fc) - (2 << 22)), smulww(r22, r22)];
        let samples = input[..len * channels]
            .iter()
            .map(|&sample| res2int16(sample.fixed_res()))
            .collect::<Vec<_>>();
        let mut filtered = alloc::vec![0i16; samples.len()];
        if channels == 1 {
            let mut state = [memory[0], memory[1]];
            crate::silk::biquad_alt::biquad_alt_stride1(
                &samples,
                &b,
                &a,
                &mut state,
                &mut filtered,
            );
            memory[..2].copy_from_slice(&state);
        } else {
            crate::silk::biquad_alt::biquad_alt_stride2(&samples, &b, &a, memory, &mut filtered);
        }
        for (index, (out, value)) in output.iter_mut().zip(filtered).enumerate() {
            native[index] = int16tores(value);
            *out = res2float(native[index]);
        }
    } else {
        let shift = 31 - (fs / 12).leading_zeros();
        for channel in 0..channels {
            for index in 0..len {
                let position = index * channels + channel;
                let sample = (input[position].fixed_res() as i32)
                    .clamp(-((1 << (16 + RES_SHIFT)) - 1), (1 << (16 + RES_SHIFT)) - 1);
                let x = sample << (14 - RES_SHIFT);
                let y = x.wrapping_sub(memory[2 * channel]);
                memory[2 * channel] = memory[2 * channel].wrapping_add(round_shift(y, shift));
                let value = round_shift(y, 14 - RES_SHIFT);
                #[cfg(not(feature = "enable_res24"))]
                let value = value.clamp(-32767, 32767) as i16;
                native[position] = value;
                *output.get_mut(position).expect("validated PCM output") = res2float(value);
            }
        }
    }
    native
}

fn dc_reject<P: PcmSample>(
    input: &[P],
    cutoff_hz: i32,
    output: &mut [OpusRes],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    assert!(channels == 1 || channels == 2, "unsupported channel count");
    let expected = len.saturating_mul(channels);
    assert!(input.len() >= expected, "input buffer too small");
    assert!(output.len() >= expected, "output buffer too small");

    let coef = 6.3f32 * cutoff_hz as f32 / fs as f32;
    let coef2 = 1.0 - coef;

    #[cfg(test)]
    let trace = match (
        opus_pcm_trace::config_copy(),
        opus_pcm_trace::current_frame(),
    ) {
        (Some(cfg), Some(frame_idx)) => {
            if cfg.frame.map_or(false, |want| want != frame_idx) {
                None
            } else {
                let start = cfg.start.min(len);
                let end = start.saturating_add(cfg.count).min(len);
                Some((cfg, frame_idx, start, end))
            }
        }
        _ => None,
    };

    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for i in 0..len {
            let idx = i * 2;
            let x0 = input[idx].normalized();
            let x1 = input[idx + 1].normalized();
            #[cfg(test)]
            let m0_pre = m0;
            #[cfg(test)]
            let m2_pre = m2;
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = coef * x0 + VERY_SMALL + coef2 * m0;
            m2 = coef * x1 + VERY_SMALL + coef2 * m2;
            output[idx] = out0;
            output[idx + 1] = out1;

            #[cfg(test)]
            if let Some((cfg, frame_idx, start, end)) = trace {
                if i >= start && i < end {
                    if i == start {
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].coef={coef:.9e}"
                        );
                        if cfg.want_bits {
                            crate::test_trace::trace_println!(
                                "opus_dc_reject[{frame_idx}].coef_bits=0x{:08x}",
                                coef.to_bits()
                            );
                        }
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].coef2={coef2:.9e}"
                        );
                        if cfg.want_bits {
                            crate::test_trace::trace_println!(
                                "opus_dc_reject[{frame_idx}].coef2_bits=0x{:08x}",
                                coef2.to_bits()
                            );
                        }
                    }
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].x={x0:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_pre={m0_pre:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].out={out0:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_post={m0:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[1].i[{i}].x={x1:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[1].i[{i}].m0_pre={m2_pre:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[1].i[{i}].out={out1:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[1].i[{i}].m0_post={m2:.9e}"
                    );
                    if cfg.want_bits {
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].x_bits=0x{:08x}",
                            x0.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_pre_bits=0x{:08x}",
                            m0_pre.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].out_bits=0x{:08x}",
                            out0.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_post_bits=0x{:08x}",
                            m0.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[1].i[{i}].x_bits=0x{:08x}",
                            x1.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[1].i[{i}].m0_pre_bits=0x{:08x}",
                            m2_pre.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[1].i[{i}].out_bits=0x{:08x}",
                            out1.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[1].i[{i}].m0_post_bits=0x{:08x}",
                            m2.to_bits()
                        );
                    }
                }
            }
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0 = hp_mem[0];
        for i in 0..len {
            let x = input[i].normalized();
            #[cfg(test)]
            let m0_pre = m0;
            let y = x - m0;
            m0 = coef * x + VERY_SMALL + coef2 * m0;
            output[i] = y;

            #[cfg(test)]
            if let Some((cfg, frame_idx, start, end)) = trace {
                if i >= start && i < end {
                    if i == start {
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].coef={coef:.9e}"
                        );
                        if cfg.want_bits {
                            crate::test_trace::trace_println!(
                                "opus_dc_reject[{frame_idx}].coef_bits=0x{:08x}",
                                coef.to_bits()
                            );
                        }
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].coef2={coef2:.9e}"
                        );
                        if cfg.want_bits {
                            crate::test_trace::trace_println!(
                                "opus_dc_reject[{frame_idx}].coef2_bits=0x{:08x}",
                                coef2.to_bits()
                            );
                        }
                    }
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].x={x:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_pre={m0_pre:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].out={y:.9e}"
                    );
                    crate::test_trace::trace_println!(
                        "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_post={m0:.9e}"
                    );
                    if cfg.want_bits {
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].x_bits=0x{:08x}",
                            x.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_pre_bits=0x{:08x}",
                            m0_pre.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].out_bits=0x{:08x}",
                            y.to_bits()
                        );
                        crate::test_trace::trace_println!(
                            "opus_dc_reject[{frame_idx}].ch[0].i[{i}].m0_post_bits=0x{:08x}",
                            m0.to_bits()
                        );
                    }
                }
            }
        }
        hp_mem[0] = m0;
    }
}

/// Applies a smooth gain transition from `g1` to `g2` over the overlap region,
/// then constant `g2` for the remainder of the frame. This mirrors the C
/// `gain_fade()` function used in the hybrid encoder path to attenuate the
/// CELT high-band when it receives fewer bits.
///
/// The overlap region uses a squared-window interpolation to avoid
/// discontinuities when switching between gain values across frames.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, not(feature = "fixed_point")))]
fn gain_fade_float(
    pcm: &mut [OpusRes],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
) {
    debug_assert!(channels == 1 || channels == 2);
    debug_assert!(fs > 0);

    let inc = (48_000 / fs).max(1) as usize;
    let overlap = overlap48 / inc;
    let overlap_samples = overlap.min(frame_size);

    if channels == 1 {
        for (sample, pcm_val) in pcm.iter_mut().enumerate().take(overlap_samples) {
            let w = window.get(sample * inc).copied().unwrap_or(0.0);
            let w_sq = w * w;
            let g = w_sq * g2 + (1.0 - w_sq) * g1;
            *pcm_val *= g;
        }
    } else {
        for sample in 0..overlap_samples {
            let w = window.get(sample * inc).copied().unwrap_or(0.0);
            let w_sq = w * w;
            let g = w_sq * g2 + (1.0 - w_sq) * g1;
            pcm[sample * 2] *= g;
            pcm[sample * 2 + 1] *= g;
        }
    }

    // Apply constant g2 to the remainder of the frame
    for c in 0..channels {
        for sample in overlap..frame_size {
            pcm[sample * channels + c] *= g2;
        }
    }
}

/// Applies a smooth stereo width transition from `g1` to `g2` over the overlap
/// region, then constant `g2` for the remainder of the frame. This mirrors the
/// C `stereo_fade()` function used to attenuate stereo width at low bitrates.
///
/// Unlike `gain_fade()` which scales all samples, this function operates on the
/// stereo difference signal (L-R)/2, reducing stereo separation when width < 1.0.
/// A width of 1.0 preserves full stereo, while 0.0 collapses to mono.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, not(feature = "fixed_point")))]
fn stereo_fade_float(
    pcm: &mut [OpusRes],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
) {
    debug_assert_eq!(channels, 2, "stereo_fade requires stereo input");
    debug_assert!(fs > 0);

    let inc = (48_000 / fs).max(1) as usize;
    let overlap = overlap48 / inc;
    let overlap_samples = overlap.min(frame_size);

    // Invert gains: we attenuate the difference signal, so g=0 means full stereo
    // and g=1 means mono. The input g1/g2 are stereo widths where 1=full stereo.
    let g1 = 1.0 - g1;
    let g2 = 1.0 - g2;

    // Overlap region: interpolate between g1 and g2 using squared window
    for sample in 0..overlap_samples {
        let w = window.get(sample * inc).copied().unwrap_or(0.0);
        let w_sq = w * w;
        let g = w_sq * g2 + (1.0 - w_sq) * g1;

        let left = pcm[sample * 2];
        let right = pcm[sample * 2 + 1];
        let diff = (left - right) * 0.5;
        pcm[sample * 2] = left - g * diff;
        pcm[sample * 2 + 1] = right + g * diff;
    }

    // After overlap: apply constant g2 to the difference signal
    for sample in overlap_samples..frame_size {
        let left = pcm[sample * 2];
        let right = pcm[sample * 2 + 1];
        let diff = (left - right) * 0.5;
        pcm[sample * 2] = left - g2 * diff;
        pcm[sample * 2 + 1] = right + g2 * diff;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_output_fades(
    encoder: &mut OpusEncoder<'_>,
    pcm_buf: &mut [OpusRes],
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))] native_pcm: &mut [i32],
    hb_gain: f32,
    mode: i32,
    equiv_rate: i32,
    unmasked: bool,
    frame_size: usize,
    channels: usize,
) {
    let required = frame_size * channels;
    // Apply HB gain fade after all SILK processing is done.
    // This smoothly transitions gain between frames to avoid discontinuities.
    #[cfg(feature = "fixed_point")]
    let attenuate_high_band =
        encoder.prev_hb_gain < 32767.0 / 32768.0 || hb_gain < 32767.0 / 32768.0;
    #[cfg(not(feature = "fixed_point"))]
    let attenuate_high_band = encoder.prev_hb_gain < 1.0 || hb_gain < 1.0;
    if attenuate_high_band {
        let overlap48 = encoder.celt.mode.overlap;
        let window = encoder.celt.mode.window;
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        fade_native_res24(
            &mut native_pcm[..required],
            encoder.prev_hb_gain,
            hb_gain,
            overlap48,
            frame_size,
            channels,
            window,
            encoder.fs,
            false,
        );
        gain_fade(
            &mut pcm_buf[..required],
            encoder.prev_hb_gain,
            hb_gain,
            overlap48,
            frame_size,
            channels,
            window,
            encoder.fs,
        );
    }
    encoder.prev_hb_gain = hb_gain;

    // Compute stereo width for non-hybrid or mono modes.
    // In hybrid mode, silk_mode.stereo_width_q14 is already set by SILK encoder.
    if mode != MODE_HYBRID || encoder.stream_channels == 1 {
        encoder.silk_mode.stereo_width_q14 = if equiv_rate > 32000 {
            16384
        } else if equiv_rate < 16000 {
            0
        } else {
            16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000)
        };
    }

    // Apply stereo width reduction at low bitrates.
    // This must happen after buffer copying to avoid affecting the SILK part.
    // Skip when energy_masking is present (surround mode handles stereo differently).
    if unmasked && encoder.channels == 2 {
        let prev_width = encoder.hybrid_stereo_width_q14;
        let curr_width = encoder.silk_mode.stereo_width_q14 as i16;

        if prev_width < (1 << 14) || curr_width < (1 << 14) {
            let g1 = f32::from(prev_width) / 16384.0;
            let g2 = f32::from(curr_width) / 16384.0;
            let overlap48 = encoder.celt.mode.overlap;
            let window = encoder.celt.mode.window;

            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            fade_native_res24(
                &mut native_pcm[..required],
                g1,
                g2,
                overlap48,
                frame_size,
                channels,
                window,
                encoder.fs,
                true,
            );
            stereo_fade(
                &mut pcm_buf[..required],
                g1,
                g2,
                overlap48,
                frame_size,
                channels,
                window,
                encoder.fs,
            );
            encoder.hybrid_stereo_width_q14 = curr_width;
        }
    }
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    for (out, &native) in pcm_buf[..required].iter_mut().zip(&native_pcm[..required]) {
        *out = crate::celt::res2float(native);
    }
}

#[cfg(feature = "fixed_point")]
fn fixed_hb_gain(celt_rate: i32) -> i16 {
    (32767 - (crate::celt::celt_exp2_q10_fixed((-celt_rate) as i16) >> 1)) as i16
}

#[allow(clippy::too_many_arguments)]
fn gain_fade(
    pcm: &mut [OpusRes],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
) {
    #[cfg(not(feature = "fixed_point"))]
    gain_fade_float(pcm, g1, g2, overlap48, frame_size, channels, window, fs);
    #[cfg(feature = "fixed_point")]
    fade_fixed(
        pcm, g1, g2, overlap48, frame_size, channels, window, fs, false,
    );
}

#[allow(clippy::too_many_arguments)]
fn stereo_fade(
    pcm: &mut [OpusRes],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
) {
    #[cfg(not(feature = "fixed_point"))]
    stereo_fade_float(pcm, g1, g2, overlap48, frame_size, channels, window, fs);
    #[cfg(feature = "fixed_point")]
    fade_fixed(
        pcm, g1, g2, overlap48, frame_size, channels, window, fs, true,
    );
}

/// Shared fixed-point interpolation for the high-band gain and stereo-width fades.
#[cfg(feature = "fixed_point")]
#[allow(clippy::too_many_arguments)]
fn fade_fixed(
    pcm: &mut [OpusRes],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
    stereo: bool,
) {
    let quantize = |v: f32| ((f64::from(v) * 32768.0 + 0.5) as i32).clamp(-32768, 32767);
    let (mut g1, mut g2) = (quantize(g1), quantize(g2));
    if stereo {
        g1 = 32767 - g1;
        g2 = 32767 - g2;
    }
    let inc = (48000 / fs).max(1) as usize;
    let overlap = overlap48 / inc;
    #[cfg(feature = "enable_qext")]
    let qext_window = if window.len() == 240 {
        crate::celt::canonical_mode_96k().unwrap().fixed_window()
    } else {
        canonical_mode().unwrap().fixed_window()
    };
    for i in 0..frame_size {
        let gain = if i < overlap {
            #[cfg(not(feature = "enable_qext"))]
            let w = quantize(window[i * inc]);
            #[cfg(feature = "enable_qext")]
            let w = qext_window[i * inc] >> 16;
            let w = (w * w) >> 15;
            (w * g2 + (32767 - w) * g1) >> 15
        } else {
            g2
        };
        if stereo {
            let left = i32::from(crate::celt::float2res(pcm[2 * i]));
            let right = i32::from(crate::celt::float2res(pcm[2 * i + 1]));
            let diff = ((i64::from(gain) * i64::from((left - right) >> 1)) >> 15) as i32;
            pcm[2 * i] = crate::celt::res2float((left - diff) as crate::celt::FixedOpusRes);
            pcm[2 * i + 1] = crate::celt::res2float((right + diff) as crate::celt::FixedOpusRes);
        } else {
            for c in 0..channels {
                let sample = crate::celt::float2res(pcm[i * channels + c]);
                let scaled = crate::celt::mult16_res_q15(gain as i16, sample);
                pcm[i * channels + c] = crate::celt::res2float(scaled as crate::celt::FixedOpusRes);
            }
        }
    }
}

#[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
#[allow(clippy::too_many_arguments)]
fn fade_native_res24(
    pcm: &mut [i32],
    g1: f32,
    g2: f32,
    overlap48: usize,
    frame_size: usize,
    channels: usize,
    window: &[CeltCoef],
    fs: i32,
    stereo: bool,
) {
    let quantize = |v: f32| ((f64::from(v) * 32768.0 + 0.5) as i32).clamp(-32768, 32767);
    let (mut g1, mut g2) = (quantize(g1), quantize(g2));
    if stereo {
        g1 = 32767 - g1;
        g2 = 32767 - g2;
    }
    let inc = (48000 / fs).max(1) as usize;
    let overlap = overlap48 / inc;
    #[cfg(feature = "enable_qext")]
    let qext_window = if window.len() == 240 {
        crate::celt::canonical_mode_96k().unwrap().fixed_window()
    } else {
        canonical_mode().unwrap().fixed_window()
    };
    for i in 0..frame_size {
        let gain = if i < overlap {
            #[cfg(not(feature = "enable_qext"))]
            let w = quantize(window[i * inc]);
            #[cfg(feature = "enable_qext")]
            let w = qext_window[i * inc] >> 16;
            let w = (w * w) >> 15;
            (w * g2 + (32767 - w) * g1) >> 15
        } else {
            g2
        };
        if stereo {
            let difference =
                ((i64::from(gain) * i64::from((pcm[2 * i] - pcm[2 * i + 1]) >> 1)) >> 15) as i32;
            pcm[2 * i] -= difference;
            pcm[2 * i + 1] += difference;
        } else {
            for c in 0..channels {
                pcm[i * channels + c] =
                    ((i64::from(gain) * i64::from(pcm[i * channels + c])) >> 15) as i32;
            }
        }
    }
}

fn prepare_silk_prefill(
    encoder: &mut OpusEncoder<'_>,
    channels: usize,
    prefill_buf: &mut [i16],
) -> Result<usize, OpusEncodeError> {
    let encoder_buffer =
        usize::try_from(encoder.encoder_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
    let prefill_len = encoder_buffer
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if prefill_len > encoder.delay_buffer.len() || prefill_len > prefill_buf.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    let ramp_samples =
        usize::try_from(encoder.fs / 400).map_err(|_| OpusEncodeError::BadArgument)?;
    let ramp_len = ramp_samples
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    let delay_comp =
        usize::try_from(encoder.delay_compensation).map_err(|_| OpusEncodeError::BadArgument)?;

    let prefill_offset_samples = encoder_buffer
        .checked_sub(delay_comp)
        .and_then(|value| value.checked_sub(ramp_samples))
        .ok_or(OpusEncodeError::BadArgument)?;
    let prefill_offset = prefill_offset_samples
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    let ramp_end = prefill_offset
        .checked_add(ramp_len)
        .ok_or(OpusEncodeError::BadArgument)?;
    if ramp_end > encoder.delay_buffer.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    #[cfg(not(all(feature = "fixed_point", feature = "enable_res24")))]
    {
        if ramp_len > 0 {
            gain_fade(
                &mut encoder.delay_buffer[prefill_offset..ramp_end],
                0.0,
                1.0,
                encoder.celt.mode.overlap,
                ramp_samples,
                channels,
                encoder.celt.mode.window,
                encoder.fs,
            );
        }
        encoder.delay_buffer[..prefill_offset].fill(0.0);
        for (dst, &src) in prefill_buf[..prefill_len]
            .iter_mut()
            .zip(&encoder.delay_buffer[..prefill_len])
        {
            *dst = crate::celt::float2int16(src);
        }
    }
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    {
        // Retain native precision through the ramp and stereo downmix. This
        // history also supplies the subsequent CELT transition prefill.
        if ramp_len > 0 {
            fade_native_res24(
                &mut encoder.native_delay_buffer[prefill_offset..ramp_end],
                0.0,
                1.0,
                encoder.celt.mode.overlap,
                ramp_samples,
                channels,
                encoder.celt.mode.window,
                encoder.fs,
                false,
            );
        }
        encoder.native_delay_buffer[..prefill_offset].fill(0);
        for i in 0..prefill_len {
            let sample = encoder.native_delay_buffer[i];
            encoder.delay_buffer[i] = crate::celt::res2float(sample);
            prefill_buf[i] = crate::celt::res2int16(sample);
        }
    }

    Ok(prefill_len)
}

fn prepare_celt_prefill_from_delay(
    encoder: &OpusEncoder<'_>,
    channels: usize,
    total_buffer: usize,
    prefill_buf: &mut [OpusRes],
) -> Result<usize, OpusEncodeError> {
    let encoder_buffer =
        usize::try_from(encoder.encoder_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
    let prefill_samples =
        usize::try_from(encoder.fs / 400).map_err(|_| OpusEncodeError::BadArgument)?;
    let prefill_len = prefill_samples
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    if prefill_len > prefill_buf.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    let delay_start_samples = encoder_buffer
        .checked_sub(total_buffer)
        .and_then(|value| value.checked_sub(prefill_samples))
        .ok_or(OpusEncodeError::BadArgument)?;
    let delay_start = delay_start_samples
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    let delay_end = delay_start
        .checked_add(prefill_len)
        .ok_or(OpusEncodeError::BadArgument)?;
    if delay_end > encoder.delay_buffer.len() {
        return Err(OpusEncodeError::BadArgument);
    }

    prefill_buf[..prefill_len].copy_from_slice(&encoder.delay_buffer[delay_start..delay_end]);
    Ok(prefill_len)
}

fn encode_frame_native<'mode, P: PcmSample>(
    encoder: &mut OpusEncoder<'mode>,
    energy_masking: Option<&[f32]>,
    pcm: &[P],
    frame_size: usize,
    data: &mut [u8],
    lsb_depth: i32,
    silk_use_dtx: bool,
    is_silence: bool,
    redundancy: bool,
    celt_to_silk: bool,
    prefill: PrefillMode,
    bandwidth_int: i32,
    mode: i32,
    to_celt: bool,
    equiv_rate: i32,
    first_frame: bool,
    dred_bitrate_bps: i32,
) -> Result<usize, OpusEncodeError> {
    if data.len() < 2 {
        return Err(OpusEncodeError::BufferTooSmall);
    }

    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    if energy_masking.is_some_and(|mask| mask.len() < 21 * channels) {
        return Err(OpusEncodeError::BadArgument);
    }
    encoder.celt.encoder().owned_energy_mask = energy_masking.map(|mask| mask.to_vec());
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }

    #[cfg(test)]
    let _trace_pcm_frame = opus_pcm_trace::begin_frame();
    #[cfg(test)]
    let trace_budget_frame_idx = opus_celt_budget_trace::begin_frame();
    #[cfg(test)]
    let trace_range_frame_idx = crate::range::begin_range_done_trace_frame();
    #[cfg(test)]
    if let Some(frame_idx) = trace_range_frame_idx {
        crate::range::set_range_done_trace_frame(frame_idx);
    }

    let frame_size_i32 = i32::try_from(frame_size).map_err(|_| OpusEncodeError::BadArgument)?;
    let frame_rate = encoder
        .fs
        .checked_div(frame_size_i32)
        .ok_or(OpusEncodeError::BadArgument)?;
    let original_packet_capacity =
        i32::try_from(data.len()).map_err(|_| OpusEncodeError::BadArgument)?;
    let max_data_bytes = original_packet_capacity.min(1276);
    let mut pcm_buf_storage = [OpusRes::default(); MAX_PCM_BUF_SAMPLES];
    let pcm_buf_len = prepare_pcm_buffer(encoder, pcm, frame_size, channels, &mut pcm_buf_storage)?;
    let pcm_buf = &mut pcm_buf_storage[..pcm_buf_len];
    let total_buffer = if matches!(encoder.application, OpusApplication::RestrictedLowDelay) {
        0
    } else {
        encoder.delay_compensation
    };
    let total_buffer = usize::try_from(total_buffer).map_err(|_| OpusEncodeError::BadArgument)?;
    let delay_len = total_buffer
        .checked_mul(channels)
        .ok_or(OpusEncodeError::BadArgument)?;
    let cutoff_hz = update_high_pass_state(encoder, mode);
    let filtered_pcm = &mut pcm_buf[delay_len..delay_len + required];
    #[cfg(not(feature = "fixed_point"))]
    if encoder.application == OpusApplication::Voip {
        hp_cutoff(
            pcm,
            cutoff_hz,
            filtered_pcm,
            &mut encoder.hp_mem,
            frame_size,
            channels,
            encoder.fs,
        );
    } else if cfg!(feature = "enable_qext") && {
        #[cfg(feature = "enable_qext")]
        {
            encoder.enable_qext
        }
        #[cfg(not(feature = "enable_qext"))]
        {
            false
        }
    } {
        for (output, sample) in filtered_pcm.iter_mut().zip(pcm) {
            *output = sample.normalized();
        }
    } else {
        dc_reject(
            pcm,
            3,
            filtered_pcm,
            &mut encoder.hp_mem,
            frame_size,
            channels,
            encoder.fs,
        );
    }

    #[cfg(feature = "fixed_point")]
    let filtered_native = {
        #[cfg(feature = "enable_qext")]
        if encoder.enable_qext && encoder.application != OpusApplication::Voip {
            let native: Vec<_> = pcm[..required]
                .iter()
                .map(|sample| sample.fixed_res())
                .collect();
            for (output, sample) in filtered_pcm.iter_mut().zip(&native) {
                *output = crate::celt::res2float(*sample);
            }
            native
        } else {
            filter_pcm_fixed(
                pcm,
                filtered_pcm,
                &mut encoder.fixed_hp_mem,
                frame_size,
                channels,
                encoder.fs,
                cutoff_hz,
                encoder.application == OpusApplication::Voip,
            )
        }
        #[cfg(not(feature = "enable_qext"))]
        filter_pcm_fixed(
            pcm,
            filtered_pcm,
            &mut encoder.fixed_hp_mem,
            frame_size,
            channels,
            encoder.fs,
            cutoff_hz,
            encoder.application == OpusApplication::Voip,
        )
    };
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    let mut native_pcm = {
        let mut values = alloc::vec![0; pcm_buf_len];
        let offset = encoder.encoder_buffer as usize * channels - delay_len;
        values[..delay_len]
            .copy_from_slice(&encoder.native_delay_buffer[offset..offset + delay_len]);
        values[delay_len..delay_len + required].copy_from_slice(&filtered_native);
        values
    };
    #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
    let _ = filtered_native;

    #[cfg(test)]
    if let Some(frame_idx) = opus_pcm_trace::current_frame() {
        opus_pcm_trace::dump(
            "pcm_buf",
            frame_idx,
            pcm_buf,
            channels,
            total_buffer + frame_size,
        );
    }

    #[cfg(feature = "fixed_point")]
    let _ = is_silence;

    let mut bandwidth = Bandwidth::from_opus_int(bandwidth_int).unwrap_or(Bandwidth::Wide);

    #[cfg(test)]
    if let Ok(path) = std::env::var("OPUS_TRACE_FILTERED_OUT") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("filtered trace");
        for sample in &pcm_buf[delay_len..delay_len + required] {
            std::io::Write::write_all(&mut file, &sample.to_le_bytes())
                .expect("filtered trace sample");
        }
    }

    let max_frame_bytes_i32 = max_data_bytes.min(1276);
    let max_frame_bytes =
        usize::try_from(max_frame_bytes_i32).map_err(|_| OpusEncodeError::BadArgument)?;
    let max_payload_bytes = max_frame_bytes.saturating_sub(1);
    #[cfg(feature = "enable_qext")]
    let max_payload_bytes = if mode == MODE_CELT_ONLY && encoder.enable_qext {
        data.len().saturating_sub(1)
    } else {
        max_payload_bytes
    };

    #[cfg(not(feature = "dred"))]
    let _ = dred_bitrate_bps;

    #[cfg(feature = "fixed_point")]
    let mut activity = if is_silence {
        0
    } else if encoder.analysis_info.valid {
        i32::from(
            encoder.analysis_info.activity_probability >= 0.1
                || (encoder.fixed_peak_signal_energy as f32)
                    < 316.23 * compute_frame_energy_fixed(&pcm[..required]) as f32,
        )
    } else if mode == MODE_CELT_ONLY {
        i32::from(
            i64::from(encoder.fixed_peak_signal_energy)
                < 316 * i64::from(compute_frame_energy_fixed(&pcm[..required]) >> 1),
        )
    } else {
        -1
    };
    #[cfg(not(feature = "fixed_point"))]
    let mut activity = {
        if is_silence {
            0
        } else if encoder.analysis_info.valid {
            i32::from(
                encoder.analysis_info.activity_probability >= 0.1
                    || encoder.peak_signal_energy < 316.23 * compute_frame_energy(&pcm[..required]),
            )
        } else if mode == MODE_CELT_ONLY {
            i32::from(
                encoder.peak_signal_energy
                    < 316.23 * (0.5 * compute_frame_energy(&pcm[..required])),
            )
        } else {
            -1
        }
    };

    #[cfg(feature = "dred")]
    {
        if encoder.dred_duration > 0 && encoder.dred_loaded {
            #[cfg(feature = "fixed_point")]
            {
                let src = &pcm_buf[delay_len..delay_len + required];
                let mut pcm_f32 = Vec::with_capacity(required);
                pcm_f32.extend(src.iter().map(|&value| value as f32));
                dred_compute_latents(
                    &mut encoder.dred_encoder,
                    &pcm_f32,
                    frame_size,
                    total_buffer as i32,
                    encoder.arch,
                );
            }
            #[cfg(not(feature = "fixed_point"))]
            dred_compute_latents(
                &mut encoder.dred_encoder,
                &pcm_buf[delay_len..delay_len + required],
                frame_size,
                total_buffer as i32,
                encoder.arch,
            );

            let frame_size_400hz = ((frame_size_i32 * 400) / encoder.fs)
                .clamp(0, DRED_ACTIVITY_MEM_LEN as i32) as usize;
            let mem_len = encoder.dred_activity_mem.len();
            if frame_size_400hz < mem_len {
                encoder
                    .dred_activity_mem
                    .copy_within(0..(mem_len - frame_size_400hz), frame_size_400hz);
            }
            for value in encoder.dred_activity_mem[..frame_size_400hz.min(mem_len)].iter_mut() {
                *value = activity as u8;
            }
            encoder.dred_latents_buffer_fill = encoder.dred_encoder.latents_buffer_fill;
        } else {
            encoder.dred_encoder.latents_buffer_fill = 0;
            encoder.dred_latents_buffer_fill = 0;
            encoder.dred_activity_mem.fill(0);
        }
    }

    let mut ret = match mode {
        MODE_CELT_ONLY => {
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetPrediction(if encoder.silk_mode.reduced_dependency {
                    0
                } else {
                    2
                }),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            if bandwidth == Bandwidth::Medium {
                bandwidth = Bandwidth::Wide;
            }

            let end_band = match bandwidth {
                Bandwidth::Narrow => 13,
                Bandwidth::Medium | Bandwidth::Wide => 17,
                Bandwidth::SuperWide => 19,
                Bandwidth::Full => 21,
            };
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetEndBand(end_band),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetStartBand(0),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;

            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetLsbDepth(lsb_depth),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetChannels(
                    usize::try_from(encoder.stream_channels)
                        .map_err(|_| OpusEncodeError::BadArgument)?,
                ),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetVbr(encoder.use_vbr),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetVbrConstraint(encoder.vbr_constraint),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetBitrate(
                    if encoder.use_vbr || encoder.dred_duration > 0 {
                        encoder.bitrate_bps
                    } else {
                        OPUS_BITRATE_MAX
                    },
                ),
            )
            .map_err(|_| OpusEncodeError::InternalError)?;

            {
                // Keep CELT analysis in sync with Opus analysis for VBR decisions (matches C).
                encoder.celt.encoder().analysis = encoder.analysis_info.clone();
            }

            let needs_prefill = encoder.prev_mode > 0 && encoder.prev_mode != mode;
            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            let native_prefill = native_celt_prefill(encoder, channels, total_buffer);
            let mut prefill_pcm = [OpusRes::default(); MAX_TMP_PREFILL_SAMPLES];
            let prefill_len = if needs_prefill {
                prepare_celt_prefill_from_delay(encoder, channels, total_buffer, &mut prefill_pcm)?
            } else {
                0
            };
            update_delay_buffer(encoder, pcm_buf, frame_size, total_buffer, channels)?;
            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            update_native_delay(encoder, &native_pcm, frame_size, total_buffer, channels);
            apply_output_fades(
                encoder,
                pcm_buf,
                #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                &mut native_pcm,
                1.0,
                mode,
                equiv_rate,
                energy_masking.is_none(),
                frame_size,
                channels,
            );
            if needs_prefill {
                opus_custom_encoder_ctl(encoder.celt.encoder(), CeltEncoderCtlRequest::ResetState)
                    .map_err(|_| OpusEncodeError::InternalError)?;
                let mut dummy = [0; 2];
                #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                {
                    encoder.celt.encoder().pending_res24_input =
                        Some(native_prefill[..prefill_len].to_vec());
                }
                celt_encode_with_ec(
                    encoder.celt.encoder(),
                    Some(&prefill_pcm[..prefill_len]),
                    (encoder.fs / 400) as usize,
                    Some(&mut dummy),
                    None,
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetPrediction(0),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
            }

            #[cfg(feature = "dred")]
            if !encoder.use_vbr && encoder.dred_duration > 0 {
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetVbr(true),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetVbrConstraint(false),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
            }

            let celt_frame_size = frame_size;
            let celt_pcm = &pcm_buf[..required];
            #[cfg(feature = "dred")]
            let celt_payload_bytes = if encoder.dred_duration > 0 {
                adjust_nb_compr_bytes_for_dred(max_payload_bytes, 1, frame_rate, dred_bitrate_bps)
            } else {
                max_payload_bytes
            };
            #[cfg(not(feature = "dred"))]
            let celt_payload_bytes = max_payload_bytes;
            let mut range_encoder = RangeEncoder::with_capacity(celt_payload_bytes);
            #[cfg(test)]
            if let Some(frame_idx) = trace_budget_frame_idx {
                let tell_pre = range_encoder.tell();
                let tell_frac_pre =
                    crate::celt::ec_tell_frac(range_encoder.encoder_mut().ctx()) as i32;
                let nb_compr_bytes = i32::try_from(max_payload_bytes).unwrap_or(i32::MAX);
                opus_celt_budget_trace::dump_if_match(
                    frame_idx,
                    mode,
                    max_data_bytes,
                    max_frame_bytes_i32,
                    i32::try_from(max_payload_bytes).unwrap_or(i32::MAX),
                    frame_size_i32,
                    frame_rate,
                    equiv_rate,
                    encoder.bitrate_bps,
                    encoder.use_vbr,
                    encoder.vbr_constraint,
                    encoder.channels,
                    encoder.stream_channels,
                    false,
                    false,
                    to_celt,
                    0,
                    nb_compr_bytes,
                    tell_pre,
                    tell_frac_pre,
                    tell_pre,
                    tell_frac_pre,
                );
            }
            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            {
                encoder.celt.encoder().pending_res24_input = Some(native_pcm[..required].to_vec());
            }
            #[cfg(feature = "enable_qext")]
            {
                encoder.celt.encoder().enable_qext = encoder.enable_qext;
            }
            let bytes = celt_encode_with_ec(
                encoder.celt.encoder(),
                Some(celt_pcm),
                celt_frame_size,
                None,
                Some(range_encoder.encoder_mut()),
            )
            .map_err(|err| match err {
                CeltEncodeError::MissingOutput => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;

            #[cfg(feature = "enable_qext")]
            {
                encoder.celt.encoder().enable_qext = false;
            }
            let range_final = range_encoder.range_final();
            let payload = range_encoder.finish_without_done();
            let mut payload_len = payload.len();
            if payload_len > max_payload_bytes {
                return Err(OpusEncodeError::BufferTooSmall);
            }
            if bytes == 0 {
                if max_payload_bytes == 0 {
                    return Err(OpusEncodeError::BufferTooSmall);
                }
                data[1] = 0;
                payload_len = 1;
            } else {
                data[1..1 + payload_len].copy_from_slice(&payload);
            }

            let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
            data[0] = toc;
            #[cfg(feature = "enable_qext")]
            if encoder.celt.encoder().qext_packet {
                data[0] |= 3;
            }
            encoder.bandwidth = bandwidth;
            encoder.range_final = range_final;
            finish_encode(encoder, mode, to_celt, frame_size_i32);

            1 + payload_len
        }
        MODE_SILK_ONLY | MODE_HYBRID => {
            let mut redundancy = redundancy;
            let mut celt_to_silk = celt_to_silk;
            let mut prefill = prefill;
            let mut redundancy_bytes = 0i32;
            let mut redundant_rng = 0u32;

            if encoder.silk_bw_switch {
                redundancy = true;
                celt_to_silk = true;
                encoder.silk_bw_switch = false;
                prefill = PrefillMode::PrefillWithState;
            }

            if redundancy {
                redundancy_bytes = compute_redundancy_bytes(
                    max_frame_bytes_i32,
                    encoder.bitrate_bps,
                    frame_rate,
                    encoder.stream_channels,
                );
                if redundancy_bytes == 0 {
                    redundancy = false;
                }
            }

            // Preserve the partial-byte target: SILK rate control consumes bits,
            // while packet storage is independently rounded to whole bytes.
            let target_bits_from_rate = (i64::from(encoder.bitrate_bps) * i64::from(frame_size_i32)
                / i64::from(encoder.fs)) as i32;
            let bits_target =
                (8 * (max_frame_bytes_i32 - redundancy_bytes)).min(target_bits_from_rate) - 8;

            encoder.silk_mode.payload_size_ms =
                (i64::from(frame_size_i32) * 1000 / i64::from(encoder.fs)) as i32;
            encoder.silk_mode.n_channels_api = encoder.channels;
            encoder.silk_mode.n_channels_internal = encoder.stream_channels;
            encoder.silk_mode.api_sample_rate = encoder.fs;
            encoder.silk_mode.packet_loss_percentage = encoder.packet_loss_perc;
            encoder.silk_mode.complexity = encoder.complexity;
            encoder.silk_mode.use_in_band_fec = i32::from(encoder.inband_fec);
            encoder.silk_mode.use_dtx = i32::from(silk_use_dtx);
            encoder.silk_mode.use_cbr = i32::from(!encoder.use_vbr);
            encoder.silk_mode.reduced_dependency = encoder.prediction_disabled;
            encoder.silk_mode.opus_can_switch = false;

            encoder.silk_mode.desired_internal_sample_rate = match bandwidth_int {
                OPUS_BANDWIDTH_NARROWBAND => 8_000,
                OPUS_BANDWIDTH_MEDIUMBAND => 12_000,
                _ => 16_000,
            };
            debug_assert!(
                mode == MODE_HYBRID || bandwidth_int <= OPUS_BANDWIDTH_WIDEBAND,
                "unexpected SILK internal bandwidth selection",
            );
            encoder.silk_mode.min_internal_sample_rate =
                if mode == MODE_HYBRID { 16_000 } else { 8_000 };
            encoder.silk_mode.max_internal_sample_rate = 16_000;
            if mode == MODE_SILK_ONLY {
                let mut effective_max_rate = max_frame_bytes_i32 * 8 * encoder.fs / frame_size_i32;
                if frame_rate > 50 {
                    effective_max_rate = effective_max_rate * 2 / 3;
                }
                if effective_max_rate < 8000 {
                    encoder.silk_mode.max_internal_sample_rate = 12_000;
                    encoder.silk_mode.desired_internal_sample_rate =
                        encoder.silk_mode.desired_internal_sample_rate.min(12_000);
                }
                if effective_max_rate < 7000 {
                    encoder.silk_mode.max_internal_sample_rate = 8_000;
                    encoder.silk_mode.desired_internal_sample_rate =
                        encoder.silk_mode.desired_internal_sample_rate.min(8_000);
                }
            }

            #[cfg(feature = "enable_qext")]
            if encoder.fs == 96000 {
                encoder.silk_mode.max_internal_sample_rate = 16000;
                encoder.silk_mode.desired_internal_sample_rate = 16000;
            }

            encoder.silk_mode.max_bits = (max_frame_bytes_i32 - 1).saturating_mul(8);
            if redundancy && redundancy_bytes >= 2 {
                encoder.silk_mode.max_bits = encoder
                    .silk_mode
                    .max_bits
                    .saturating_sub(redundancy_bytes * 8 + 1);
                if mode == MODE_HYBRID {
                    encoder.silk_mode.max_bits = encoder.silk_mode.max_bits.saturating_sub(20);
                }
            }

            let frame_20ms = frame_size_i32 * 50 == encoder.fs;
            let total_bitrate = (i64::from(bits_target) * i64::from(encoder.fs)
                / i64::from(frame_size_i32))
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            encoder.silk_mode.bit_rate = if mode == MODE_HYBRID {
                compute_silk_rate_for_hybrid(
                    total_bitrate,
                    bandwidth_int,
                    frame_20ms,
                    encoder.use_vbr,
                    encoder.silk_mode.lbrr_coded != 0,
                    encoder.stream_channels,
                )
            } else {
                total_bitrate
            };

            // Apply surround masking rate offset for SILK bitrate adjustment.
            if let Some(mask) = energy_masking.filter(|_| encoder.use_vbr && !encoder.lfe) {
                let rate_offset = compute_surround_masking_rate_offset(
                    mask,
                    bandwidth_int,
                    encoder.stream_channels,
                );
                let rate_offset = rate_offset.max(-2 * encoder.silk_mode.bit_rate / 3);
                if bandwidth_int == OPUS_BANDWIDTH_SUPERWIDEBAND
                    || bandwidth_int == OPUS_BANDWIDTH_FULLBAND
                {
                    encoder.silk_mode.bit_rate += 3 * rate_offset / 5;
                } else {
                    encoder.silk_mode.bit_rate += rate_offset;
                }
            }

            // Compute HB gain: increasingly attenuate high band when CELT gets fewer bits.
            // Skip attenuation when energy_masking is present (surround mode).
            let hb_gain = if mode == MODE_HYBRID && energy_masking.is_none() {
                let celt_rate = total_bitrate - encoder.silk_mode.bit_rate;
                // SHR32 is an identity operation in the C floating-point build.
                #[cfg(not(feature = "fixed_point"))]
                let attenuation = celt_exp2(-celt_rate as f32 / 1024.0);
                #[cfg(not(feature = "fixed_point"))]
                let gain = 1.0 - attenuation;
                #[cfg(feature = "fixed_point")]
                let gain = f32::from(fixed_hb_gain(celt_rate)) / 32768.0;
                gain
            } else {
                1.0
            };

            if encoder.silk_mode.use_cbr != 0
                && (mode == MODE_HYBRID || (cfg!(feature = "dred") && dred_bitrate_bps > 0))
            {
                let other_bits = (encoder.silk_mode.max_bits
                    - encoder.silk_mode.bit_rate * frame_size_i32 / encoder.fs)
                    .max(0);
                encoder.silk_mode.max_bits =
                    (encoder.silk_mode.max_bits - other_bits * 3 / 4).max(0);
                encoder.silk_mode.use_cbr = 0;
            } else if mode == MODE_HYBRID {
                let max_bit_rate = compute_silk_rate_for_hybrid(
                    encoder.silk_mode.max_bits * encoder.fs / frame_size_i32,
                    bandwidth_int,
                    frame_20ms,
                    encoder.use_vbr,
                    encoder.silk_mode.lbrr_coded != 0,
                    encoder.stream_channels,
                );
                encoder.silk_mode.max_bits = max_bit_rate * frame_size_i32 / encoder.fs;
            }

            if !matches!(prefill, PrefillMode::None) {
                let mut prefill_buf = [0i16; DELAY_BUFFER_SAMPLES];
                let prefill_len = prepare_silk_prefill(encoder, channels, &mut prefill_buf)?;
                if prefill_len > 0 {
                    let mut prefill_encoder = RangeEncoder::new();
                    let mut prefill_bytes = 0i32;
                    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                    let (encode_prefill, prefill_input) = (
                        crate::silk::enc_api::silk_encode_res24,
                        &encoder.native_delay_buffer[..prefill_len],
                    );
                    #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
                    let (encode_prefill, prefill_input) =
                        (silk_encode, &prefill_buf[..prefill_len]);
                    #[cfg(not(feature = "fixed_point"))]
                    let (encode_prefill, prefill_input) =
                        (silk_encode_float, &encoder.delay_buffer[..prefill_len]);
                    encode_prefill(
                        &mut encoder.silk,
                        &mut encoder.silk_mode,
                        prefill_input,
                        &mut prefill_encoder,
                        &mut prefill_bytes,
                        prefill,
                        activity,
                    )?;
                    encoder.silk_mode.opus_can_switch = false;
                }
            }

            let mut range_encoder = RangeEncoder::with_capacity(max_payload_bytes);
            let mut bytes_out = 0i32;
            #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
            let silk_input_storage: Vec<i16> = pcm_buf[delay_len..delay_len + required]
                .iter()
                .map(|&sample| crate::celt::res2int16(crate::celt::float2res(sample)))
                .collect();
            #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
            let silk_input = silk_input_storage.as_slice();
            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            let silk_input = &native_pcm[delay_len..delay_len + required];
            #[cfg(not(feature = "fixed_point"))]
            let silk_input = &pcm_buf[delay_len..delay_len + required];
            #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
            let encode_silk = silk_encode;
            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
            let encode_silk = crate::silk::enc_api::silk_encode_res24;
            #[cfg(not(feature = "fixed_point"))]
            let encode_silk = silk_encode_float;
            encode_silk(
                &mut encoder.silk,
                &mut encoder.silk_mode,
                silk_input,
                &mut range_encoder,
                &mut bytes_out,
                PrefillMode::None,
                activity,
            )?;

            #[cfg(test)]
            if std::env::var_os("OPUS_TRACE_FRONT").is_some() {
                crate::test_trace::trace_println!(
                    "silk_front mode={} rate={} maxbits={} bytes={} range={} tell={}",
                    mode,
                    encoder.silk_mode.bit_rate,
                    encoder.silk_mode.max_bits,
                    bytes_out,
                    range_encoder.range_final(),
                    range_encoder.tell()
                );
            }

            if activity < 0 {
                activity = i32::from(
                    encoder.silk_mode.signal_type != crate::silk::FrameSignalType::Inactive,
                );
                #[cfg(feature = "dred")]
                {
                    let count = (frame_size_i32 * 400 / encoder.fs) as usize;
                    encoder.dred_activity_mem[..count].fill(activity as u8);
                }
            }

            if mode == MODE_SILK_ONLY {
                bandwidth = OpusEncoder::bandwidth_from_silk_control(&encoder.silk_mode);
            }
            if bytes_out == 0 {
                let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
                data[0] = toc;
                encoder.bandwidth = bandwidth;
                encoder.range_final = 0;
                finish_encode(encoder, mode, to_celt, frame_size_i32);
                1
            } else {
                debug_assert!(
                    mode == MODE_SILK_ONLY || encoder.silk_mode.internal_sample_rate == 16_000,
                    "hybrid SILK internal sample rate must remain at 16 kHz",
                );

                encoder.silk_mode.opus_can_switch =
                    encoder.silk_mode.switch_ready && !encoder.nonfinal_frame;
                if encoder.silk_mode.opus_can_switch {
                    redundancy_bytes = compute_redundancy_bytes(
                        max_frame_bytes_i32,
                        encoder.bitrate_bps,
                        frame_rate,
                        encoder.stream_channels,
                    );
                    redundancy = redundancy_bytes != 0;
                    celt_to_silk = false;
                    encoder.silk_bw_switch = true;
                }

                if range_encoder.tell() + 17 + if mode == MODE_HYBRID { 20 } else { 0 }
                    <= 8 * (max_frame_bytes_i32 - 1)
                {
                    if mode == MODE_HYBRID {
                        range_encoder.encode_bit_logp(i32::from(redundancy), 12);
                    }
                    if redundancy {
                        range_encoder.encode_bit_logp(i32::from(celt_to_silk), 1);
                        let max_redundancy = (max_frame_bytes_i32 - 1)
                            - ((range_encoder.tell()
                                + if mode == MODE_HYBRID { 8 + 3 + 7 } else { 7 })
                                >> 3);
                        redundancy_bytes = redundancy_bytes.min(max_redundancy);
                        redundancy_bytes = redundancy_bytes.clamp(2, 257);
                        if mode == MODE_HYBRID {
                            range_encoder.encode_uint((redundancy_bytes - 2) as u32, 256);
                        }
                    }
                } else {
                    redundancy = false;
                }

                if !redundancy {
                    encoder.silk_bw_switch = false;
                    redundancy_bytes = 0;
                }

                let mut nb_compr_bytes = usize::try_from(if mode == MODE_SILK_ONLY {
                    (range_encoder.tell() + 7) >> 3
                } else {
                    (max_frame_bytes_i32 - 1 - redundancy_bytes).max(0)
                })
                .map_err(|_| OpusEncodeError::BadArgument)?;
                #[cfg(test)]
                let (tell_pre, tell_frac_pre) = if trace_budget_frame_idx.is_some() {
                    (
                        range_encoder.tell(),
                        crate::celt::ec_tell_frac(range_encoder.encoder_mut().ctx()) as i32,
                    )
                } else {
                    (0, 0)
                };
                #[cfg(feature = "dred")]
                if encoder.dred_duration > 0 {
                    nb_compr_bytes = adjust_nb_compr_bytes_for_dred(
                        nb_compr_bytes,
                        range_encoder.tell(),
                        frame_rate,
                        dred_bitrate_bps,
                    );
                }
                if nb_compr_bytes < 2 {
                    return Err(OpusEncodeError::BufferTooSmall);
                }
                range_encoder.shrink(nb_compr_bytes.min(max_payload_bytes));
                if mode == MODE_SILK_ONLY {
                    range_encoder.encoder_mut().enc_done();
                }
                #[cfg(test)]
                if let Some(frame_idx) = trace_budget_frame_idx {
                    let tell_post = range_encoder.tell();
                    let tell_frac_post =
                        crate::celt::ec_tell_frac(range_encoder.encoder_mut().ctx()) as i32;
                    let nb_compr_bytes_i32 = i32::try_from(nb_compr_bytes).unwrap_or(i32::MAX);
                    opus_celt_budget_trace::dump_if_match(
                        frame_idx,
                        mode,
                        max_data_bytes,
                        max_frame_bytes_i32,
                        i32::try_from(max_payload_bytes).unwrap_or(i32::MAX),
                        frame_size_i32,
                        frame_rate,
                        equiv_rate,
                        encoder.bitrate_bps,
                        encoder.use_vbr,
                        encoder.vbr_constraint,
                        encoder.channels,
                        encoder.stream_channels,
                        redundancy,
                        celt_to_silk,
                        to_celt,
                        redundancy_bytes,
                        nb_compr_bytes_i32,
                        tell_pre,
                        tell_frac_pre,
                        tell_post,
                        tell_frac_post,
                    );
                }

                let end_band = match bandwidth {
                    Bandwidth::Narrow => 13,
                    Bandwidth::Medium | Bandwidth::Wide => 17,
                    Bandwidth::SuperWide => 19,
                    Bandwidth::Full => 21,
                };
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetEndBand(end_band),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetChannels(
                        usize::try_from(encoder.stream_channels)
                            .map_err(|_| OpusEncodeError::BadArgument)?,
                    ),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetLsbDepth(lsb_depth),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                if mode != MODE_SILK_ONLY {
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetPrediction(
                            if encoder.silk_mode.reduced_dependency {
                                0
                            } else {
                                2
                            },
                        ),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                }

                {
                    if redundancy || mode != MODE_SILK_ONLY {
                        encoder.celt.encoder().analysis = encoder.analysis_info.clone();
                    }
                }

                if mode == MODE_HYBRID {
                    encoder.celt.encoder().silk_info = SilkInfo {
                        signal_type: i32::from(encoder.silk_mode.signal_type),
                        offset: encoder.silk_mode.offset,
                    };
                }

                let need_tmp_prefill =
                    mode != MODE_SILK_ONLY && encoder.prev_mode != mode && encoder.prev_mode > 0;
                #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                let native_prefill = native_celt_prefill(encoder, channels, total_buffer);
                let mut tmp_prefill = [OpusRes::default(); MAX_TMP_PREFILL_SAMPLES];
                let mut tmp_prefill_len = 0usize;
                if need_tmp_prefill {
                    tmp_prefill_len = prepare_celt_prefill_from_delay(
                        encoder,
                        channels,
                        total_buffer,
                        &mut tmp_prefill,
                    )?;
                }

                update_delay_buffer(encoder, pcm_buf, frame_size, total_buffer, channels)?;
                #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                update_native_delay(encoder, &native_pcm, frame_size, total_buffer, channels);

                apply_output_fades(
                    encoder,
                    pcm_buf,
                    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                    &mut native_pcm,
                    hb_gain,
                    mode,
                    equiv_rate,
                    energy_masking.is_none(),
                    frame_size,
                    channels,
                );

                let celt_pcm = &pcm_buf[..required];

                let mut redundancy_buf = [0u8; 257];
                let mut redundancy_len = 0usize;
                if redundancy && celt_to_silk {
                    let n2 = usize::try_from(encoder.fs / 200)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    let red_len = usize::try_from(redundancy_bytes)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    if n2 > 0 && red_len >= 2 {
                        debug_assert!(red_len <= redundancy_buf.len());
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetStartBand(0),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetVbr(false),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                        {
                            encoder.celt.encoder().pending_res24_input =
                                Some(native_pcm[..n2 * channels].to_vec());
                        }
                        let used = celt_encode_with_ec(
                            encoder.celt.encoder(),
                            Some(&celt_pcm[..n2 * channels]),
                            n2,
                            Some(&mut redundancy_buf[..red_len]),
                            None,
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        redundancy_len = used;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::GetFinalRange(&mut redundant_rng),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::ResetState,
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                    }
                }

                opus_custom_encoder_ctl(
                    encoder.celt.encoder(),
                    CeltEncoderCtlRequest::SetStartBand(if mode == MODE_HYBRID { 17 } else { 0 }),
                )
                .map_err(|_| OpusEncodeError::InternalError)?;
                if mode == MODE_HYBRID {
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetVbr(encoder.use_vbr),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    if encoder.use_vbr {
                        let celt_rate = encoder.bitrate_bps - encoder.silk_mode.bit_rate;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetBitrate(celt_rate),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetVbrConstraint(false),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                    }
                }
                #[cfg(feature = "dred")]
                if !encoder.use_vbr && encoder.dred_duration > 0 {
                    let mut celt_bitrate = encoder.bitrate_bps;
                    if mode == MODE_HYBRID {
                        celt_bitrate = celt_bitrate.saturating_sub(encoder.silk_mode.bit_rate);
                    }
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetVbr(true),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetVbrConstraint(false),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetBitrate(celt_bitrate),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                }

                if need_tmp_prefill {
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::ResetState,
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                    let n4 = usize::try_from(encoder.fs / 400)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    let prefill_len = n4
                        .checked_mul(channels)
                        .ok_or(OpusEncodeError::BadArgument)?;
                    if n4 > 0 && tmp_prefill_len >= prefill_len {
                        let mut dummy = [0u8; 2];
                        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                        {
                            encoder.celt.encoder().pending_res24_input =
                                Some(native_prefill[..prefill_len].to_vec());
                        }
                        let _ = celt_encode_with_ec(
                            encoder.celt.encoder(),
                            Some(&tmp_prefill[..prefill_len]),
                            n4,
                            Some(&mut dummy),
                            None,
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                    }
                    opus_custom_encoder_ctl(
                        encoder.celt.encoder(),
                        CeltEncoderCtlRequest::SetPrediction(0),
                    )
                    .map_err(|_| OpusEncodeError::InternalError)?;
                }

                let mut enc_done = mode == MODE_SILK_ONLY;
                if mode != MODE_SILK_ONLY && range_encoder.tell() <= (nb_compr_bytes * 8) as i32 {
                    #[cfg(test)]
                    if let Ok(path) = std::env::var("OPUS_TRACE_CELT_INPUT_OUT") {
                        let mut file = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                            .unwrap();
                        for sample in celt_pcm {
                            std::io::Write::write_all(&mut file, &sample.to_le_bytes()).unwrap();
                        }
                    }

                    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                    {
                        encoder.celt.encoder().pending_res24_input =
                            Some(native_pcm[..required].to_vec());
                    }
                    let _ = celt_encode_with_ec(
                        encoder.celt.encoder(),
                        Some(celt_pcm),
                        frame_size,
                        None,
                        Some(range_encoder.encoder_mut()),
                    )
                    .map_err(|err| match err {
                        CeltEncodeError::MissingOutput => OpusEncodeError::BufferTooSmall,
                        _ => OpusEncodeError::InternalError,
                    })?;
                    enc_done = true;
                }

                if redundancy && !celt_to_silk {
                    let n2 = usize::try_from(encoder.fs / 200)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    let n4 = usize::try_from(encoder.fs / 400)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    let red_len = usize::try_from(redundancy_bytes)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    if n2 > 0 && red_len >= 2 {
                        debug_assert!(red_len <= redundancy_buf.len());
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::ResetState,
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetStartBand(0),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetPrediction(0),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetVbr(false),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;
                        opus_custom_encoder_ctl(
                            encoder.celt.encoder(),
                            CeltEncoderCtlRequest::SetBitrate(OPUS_BITRATE_MAX),
                        )
                        .map_err(|_| OpusEncodeError::InternalError)?;

                        if n4 > 0 {
                            let prefill_start =
                                frame_size.saturating_sub(n2 + n4).saturating_mul(channels);
                            let prefill_end =
                                prefill_start.saturating_add(n4.saturating_mul(channels));
                            if prefill_end <= celt_pcm.len() {
                                let mut dummy = [0u8; 2];
                                #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                                {
                                    encoder.celt.encoder().pending_res24_input =
                                        Some(native_pcm[prefill_start..prefill_end].to_vec());
                                }
                                let _ = celt_encode_with_ec(
                                    encoder.celt.encoder(),
                                    Some(&celt_pcm[prefill_start..prefill_end]),
                                    n4,
                                    Some(&mut dummy),
                                    None,
                                )
                                .map_err(|_| OpusEncodeError::InternalError)?;
                            }
                        }

                        let red_start = frame_size.saturating_sub(n2).saturating_mul(channels);
                        let red_end = red_start.saturating_add(n2.saturating_mul(channels));
                        if red_end <= celt_pcm.len() {
                            #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
                            {
                                encoder.celt.encoder().pending_res24_input =
                                    Some(native_pcm[red_start..red_end].to_vec());
                            }
                            let used = celt_encode_with_ec(
                                encoder.celt.encoder(),
                                Some(&celt_pcm[red_start..red_end]),
                                n2,
                                Some(&mut redundancy_buf[..red_len]),
                                None,
                            )
                            .map_err(|_| OpusEncodeError::InternalError)?;
                            redundancy_len = used;
                            opus_custom_encoder_ctl(
                                encoder.celt.encoder(),
                                CeltEncoderCtlRequest::GetFinalRange(&mut redundant_rng),
                            )
                            .map_err(|_| OpusEncodeError::InternalError)?;
                        }
                    }
                }

                let overflow = range_encoder.tell() > (max_frame_bytes_i32 - 1) * 8;
                let range_final = range_encoder.range_final();
                let payload = if enc_done {
                    range_encoder.finish_without_done()
                } else {
                    range_encoder.finish()
                };
                let mut payload_len = if overflow { 1 } else { payload.len() };
                if mode == MODE_SILK_ONLY && !redundancy {
                    while payload_len > 2 && payload[payload_len - 1] == 0 {
                        payload_len -= 1;
                    }
                }
                let total_len = payload_len + redundancy_len;
                if total_len > max_payload_bytes {
                    return Err(OpusEncodeError::BufferTooSmall);
                }

                let toc = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels) & 0xFC;
                data[0] = toc;
                if overflow {
                    data[1] = 0;
                } else {
                    data[1..1 + payload_len].copy_from_slice(&payload[..payload_len]);
                }
                if redundancy_len != 0 {
                    data[1 + payload_len..1 + total_len]
                        .copy_from_slice(&redundancy_buf[..redundancy_len]);
                }

                encoder.bandwidth = bandwidth;
                encoder.range_final = if overflow {
                    0
                } else {
                    range_final ^ redundant_rng
                };
                finish_encode(encoder, mode, to_celt, frame_size_i32);

                1 + total_len
            }
        }
        _ => return Err(OpusEncodeError::BadArgument),
    };

    if encoder.use_dtx && (encoder.analysis_info.valid || is_silence) {
        let frame_size_ms_q1 = 2 * 1000 * frame_size_i32 / encoder.fs;
        if decide_dtx_mode(
            activity,
            &mut encoder.nb_no_activity_ms_q1,
            frame_size_ms_q1,
        ) {
            encoder.range_final = 0;
            data[0] = gen_toc(mode, frame_rate, bandwidth, encoder.stream_channels);
            ret = 1;
        }
    } else {
        encoder.nb_no_activity_ms_q1 = 0;
    }

    #[cfg(feature = "dred")]
    {
        if ret > 1 && encoder.dred_duration > 0 && encoder.dred_loaded && first_frame {
            let mut dred_chunks =
                ((encoder.dred_duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES / 2);
            if encoder.use_vbr {
                dred_chunks = dred_chunks.min(encoder.dred_target_chunks);
            }

            let ret_i32 = i32::try_from(ret).map_err(|_| OpusEncodeError::BadArgument)?;
            // DRED occupies packet padding and can follow a QEXT payload
            // larger than the base codec's 1276-byte limit.
            let mut dred_bytes_left =
                (original_packet_capacity - ret_i32 - 3).min(DRED_MAX_DATA_SIZE as i32);
            if dred_bytes_left > 0 {
                dred_bytes_left -= (dred_bytes_left + 1 + DRED_EXPERIMENTAL_BYTES) / 255;
            }
            if dred_chunks >= 1
                && dred_bytes_left >= DRED_MIN_BYTES as i32 + DRED_EXPERIMENTAL_BYTES
            {
                let mut buf = [0u8; DRED_MAX_DATA_SIZE];
                buf[0] = b'D';
                buf[1] = DRED_EXPERIMENTAL_VERSION;
                let dred_bytes = dred_encode_silk_frame(
                    &mut encoder.dred_encoder,
                    &mut buf[DRED_EXPERIMENTAL_BYTES as usize..],
                    dred_chunks,
                    dred_bytes_left - DRED_EXPERIMENTAL_BYTES,
                    encoder.dred_q0,
                    encoder.dred_dq,
                    encoder.dred_qmax,
                    &encoder.dred_activity_mem,
                );
                if dred_bytes > 0 {
                    let dred_bytes = (dred_bytes + DRED_EXPERIMENTAL_BYTES) as usize;
                    let extension = OpusExtensionData {
                        id: DRED_EXTENSION_ID,
                        frame: 0,
                        data: &buf[..dred_bytes],
                        len: dred_bytes as i32,
                    };
                    let max_data_bytes_usize = usize::try_from(original_packet_capacity)
                        .map_err(|_| OpusEncodeError::BadArgument)?;
                    ret = opus_packet_pad_with_extensions(
                        data,
                        ret,
                        max_data_bytes_usize,
                        !encoder.use_vbr,
                        &[extension],
                    )
                    .map_err(|err| match err {
                        RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                        _ => OpusEncodeError::InternalError,
                    })?;
                }
            }
        }
    }

    let padding_bytes = data.len();
    if !encoder.use_vbr && ret > 1 && ret < padding_bytes {
        opus_packet_pad(data, ret, padding_bytes).map_err(|err| match err {
            RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
            _ => OpusEncodeError::InternalError,
        })?;
        ret = padding_bytes;
    }

    Ok(ret)
}

pub fn opus_encoder_create<'mode>(
    fs: i32,
    channels: i32,
    application: i32,
) -> Result<OpusEncoder<'mode>, OpusEncoderInitError> {
    if !(matches!(fs, 48_000 | 24_000 | 16_000 | 12_000 | 8_000)
        || cfg!(feature = "enable_qext") && fs == 96_000)
        || !matches!(channels, 1 | 2)
        || OpusApplication::from_opus_int(application).is_none()
    {
        return Err(OpusEncoderInitError::BadArgument);
    }

    let mode = {
        #[cfg(feature = "enable_qext")]
        if fs == 96_000 {
            crate::celt::canonical_mode_96k()
        } else {
            canonical_mode()
        }
        #[cfg(not(feature = "enable_qext"))]
        canonical_mode()
    }
    .ok_or(OpusEncoderInitError::CeltInit)?;
    let celt = opus_custom_encoder_create(mode, fs, channels as usize, 0)
        .map_err(|_| OpusEncoderInitError::CeltInit)?;

    let mut encoder = OpusEncoder {
        celt,
        silk: crate::silk::encoder::state::Encoder::default(),
        silk_mode: SilkEncControl::default(),
        #[cfg(feature = "dred")]
        dred_encoder: Box::new(DredEnc::default()),
        analysis: TonalityAnalysisState::new(fs),
        analysis_info: AnalysisInfo::default(),
        application: OpusApplication::Voip,
        channels,
        stream_channels: channels,
        fs,
        arch: opus_select_arch(),
        use_vbr: true,
        vbr_constraint: true,
        user_bitrate_bps: OPUS_AUTO,
        bitrate_bps: 0,
        packet_loss_perc: 0,
        complexity: 9,
        inband_fec: false,
        use_dtx: false,
        #[cfg(feature = "enable_qext")]
        enable_qext: false,
        fec_config: 0,
        force_channels: OPUS_AUTO,
        user_bandwidth: OPUS_AUTO,
        max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
        signal_type: OPUS_AUTO,
        user_forced_mode: OPUS_AUTO,
        voice_ratio: 0,
        delay_compensation: 0,
        encoder_buffer: 0,
        lsb_depth: 24,
        variable_duration: OPUS_FRAMESIZE_ARG,
        prediction_disabled: false,
        hybrid_stereo_width_q14: 0,
        variable_hp_smth2_q15: 0,
        prev_hb_gain: 0.0,
        hp_mem: [0.0; 4],
        #[cfg(feature = "fixed_point")]
        fixed_hp_mem: [0; 4],
        mode: MODE_SILK_ONLY,
        prev_mode: 0,
        prev_channels: 0,
        prev_framesize: 0,
        bandwidth: Bandwidth::Wide,
        auto_bandwidth: 0,
        silk_bw_switch: false,
        first: false,
        width_mem: StereoWidthState::default(),
        delay_buffer: [OpusRes::default(); DELAY_BUFFER_SAMPLES],
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        native_delay_buffer: [0; DELAY_BUFFER_SAMPLES],
        detected_bandwidth: 0,
        nb_no_activity_ms_q1: 0,
        peak_signal_energy: 0.0,
        #[cfg(feature = "fixed_point")]
        fixed_peak_signal_energy: 0,
        nonfinal_frame: false,
        range_final: 0,
        dred_duration: 0,
        #[cfg(feature = "dred")]
        dred_loaded: false,
        #[cfg(feature = "dred")]
        dred_latents_buffer_fill: 0,
        #[cfg(feature = "dred")]
        dred_q0: 0,
        #[cfg(feature = "dred")]
        dred_dq: 0,
        #[cfg(feature = "dred")]
        dred_qmax: 0,
        #[cfg(feature = "dred")]
        dred_target_chunks: 0,
        #[cfg(feature = "dred")]
        dred_activity_mem: [0; DRED_ACTIVITY_MEM_LEN],
        lfe: false,
    };

    encoder.init(fs, channels, application)?;
    Ok(encoder)
}

fn max_internal_sample_rate_for_bandwidth(user_bandwidth: i32, max_bandwidth: i32) -> i32 {
    let selected = if user_bandwidth == OPUS_AUTO {
        max_bandwidth
    } else {
        user_bandwidth
    };
    match selected {
        OPUS_BANDWIDTH_NARROWBAND => 8_000,
        OPUS_BANDWIDTH_MEDIUMBAND => 12_000,
        OPUS_BANDWIDTH_WIDEBAND | OPUS_BANDWIDTH_SUPERWIDEBAND | OPUS_BANDWIDTH_FULLBAND => 16_000,
        _ => 16_000,
    }
}

fn is_valid_bandwidth(value: i32) -> bool {
    matches!(
        value,
        OPUS_BANDWIDTH_NARROWBAND
            | OPUS_BANDWIDTH_MEDIUMBAND
            | OPUS_BANDWIDTH_WIDEBAND
            | OPUS_BANDWIDTH_SUPERWIDEBAND
            | OPUS_BANDWIDTH_FULLBAND
    )
}

fn is_valid_signal(value: i32) -> bool {
    matches!(value, OPUS_AUTO | OPUS_SIGNAL_VOICE | OPUS_SIGNAL_MUSIC)
}

fn is_valid_expert_frame_duration(value: i32) -> bool {
    matches!(
        value,
        OPUS_FRAMESIZE_ARG
            | OPUS_FRAMESIZE_2_5_MS
            | OPUS_FRAMESIZE_5_MS
            | OPUS_FRAMESIZE_10_MS
            | OPUS_FRAMESIZE_20_MS
            | OPUS_FRAMESIZE_40_MS
            | OPUS_FRAMESIZE_60_MS
            | OPUS_FRAMESIZE_80_MS
            | OPUS_FRAMESIZE_100_MS
            | OPUS_FRAMESIZE_120_MS
    )
}

pub fn opus_encoder_ctl<'req>(
    encoder: &mut OpusEncoder<'_>,
    request: OpusEncoderCtlRequest<'req>,
) -> Result<(), OpusEncoderCtlError> {
    match request {
        OpusEncoderCtlRequest::SetApplication(value) => {
            let application =
                OpusApplication::from_opus_int(value).ok_or(OpusEncoderCtlError::BadArgument)?;
            if !encoder.first && encoder.application != application {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.application = application;
            {
                encoder.analysis.application = value;
            }
        }
        OpusEncoderCtlRequest::GetApplication(out) => {
            *out = encoder.application.to_opus_int();
        }
        OpusEncoderCtlRequest::SetBitrate(value) => {
            if value != OPUS_AUTO && value != OPUS_BITRATE_MAX && value <= 0 {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_bitrate_bps = if matches!(value, OPUS_AUTO | OPUS_BITRATE_MAX) {
                value
            } else {
                value.clamp(500, 750_000 * encoder.channels)
            };
        }
        OpusEncoderCtlRequest::GetBitrate(out) => {
            *out = encoder.user_bitrate_bps;
        }
        OpusEncoderCtlRequest::SetForceChannels(value) => {
            if value != OPUS_AUTO && (value < 1 || value > encoder.channels) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.force_channels = value;
        }
        OpusEncoderCtlRequest::GetForceChannels(out) => {
            *out = encoder.force_channels;
        }
        OpusEncoderCtlRequest::SetMaxBandwidth(value) => {
            if !is_valid_bandwidth(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.max_bandwidth = value;
            encoder.silk_mode.max_internal_sample_rate = max_internal_sample_rate_for_bandwidth(
                encoder.user_bandwidth,
                encoder.max_bandwidth,
            );
        }
        OpusEncoderCtlRequest::GetMaxBandwidth(out) => {
            *out = encoder.max_bandwidth;
        }
        OpusEncoderCtlRequest::SetBandwidth(value) => {
            if value != OPUS_AUTO && !is_valid_bandwidth(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_bandwidth = value;
            encoder.silk_mode.max_internal_sample_rate = max_internal_sample_rate_for_bandwidth(
                encoder.user_bandwidth,
                encoder.max_bandwidth,
            );
        }
        OpusEncoderCtlRequest::GetBandwidth(out) => {
            *out = encoder.bandwidth.to_opus_int();
        }
        OpusEncoderCtlRequest::SetVbr(value) => {
            encoder.use_vbr = value;
        }
        OpusEncoderCtlRequest::GetVbr(out) => {
            *out = encoder.use_vbr;
        }
        OpusEncoderCtlRequest::SetVbrConstraint(value) => {
            encoder.vbr_constraint = value;
        }
        OpusEncoderCtlRequest::GetVbrConstraint(out) => {
            *out = encoder.vbr_constraint;
        }
        OpusEncoderCtlRequest::SetComplexity(value) => {
            if !(0..=10).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.complexity = value;
            encoder.silk_mode.complexity = value;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetComplexity(value),
            )?;
        }
        OpusEncoderCtlRequest::GetComplexity(out) => {
            *out = encoder.complexity;
        }
        OpusEncoderCtlRequest::SetSignal(value) => {
            if !is_valid_signal(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.signal_type = value;
        }
        OpusEncoderCtlRequest::GetSignal(out) => {
            *out = encoder.signal_type;
        }
        OpusEncoderCtlRequest::SetVoiceRatio(value) => {
            if !(-1..=100).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.voice_ratio = value;
        }
        OpusEncoderCtlRequest::GetVoiceRatio(out) => {
            *out = encoder.voice_ratio;
        }
        OpusEncoderCtlRequest::SetPacketLossPerc(value) => {
            if !(0..=100).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.packet_loss_perc = value;
            encoder.silk_mode.packet_loss_percentage = value;
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetPacketLossPerc(value),
            )?;
        }
        OpusEncoderCtlRequest::GetPacketLossPerc(out) => {
            *out = encoder.packet_loss_perc;
        }
        OpusEncoderCtlRequest::SetInbandFec(value) => {
            encoder.inband_fec = value;
        }
        OpusEncoderCtlRequest::GetInbandFec(out) => {
            *out = encoder.inband_fec;
        }
        #[cfg(feature = "enable_qext")]
        OpusEncoderCtlRequest::SetQext(value) => {
            encoder.enable_qext = value;
        }
        #[cfg(feature = "enable_qext")]
        OpusEncoderCtlRequest::GetQext(out) => {
            *out = encoder.enable_qext;
        }
        OpusEncoderCtlRequest::SetDtx(value) => {
            encoder.use_dtx = value;
        }
        OpusEncoderCtlRequest::GetDtx(out) => {
            *out = encoder.use_dtx;
        }
        OpusEncoderCtlRequest::GetInDtx(out) => {
            *out = encoder_in_dtx(encoder);
        }
        OpusEncoderCtlRequest::SetLsbDepth(value) => {
            if !(8..=24).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.lsb_depth = value;
        }
        OpusEncoderCtlRequest::GetLsbDepth(out) => {
            *out = encoder.lsb_depth;
        }
        OpusEncoderCtlRequest::SetExpertFrameDuration(value) => {
            if !is_valid_expert_frame_duration(value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.variable_duration = value;
        }
        OpusEncoderCtlRequest::GetExpertFrameDuration(out) => {
            *out = encoder.variable_duration;
        }
        OpusEncoderCtlRequest::SetPredictionDisabled(value) => {
            encoder.prediction_disabled = value;
            encoder.silk_mode.reduced_dependency = value;
        }
        OpusEncoderCtlRequest::GetPredictionDisabled(out) => {
            *out = encoder.prediction_disabled;
        }
        OpusEncoderCtlRequest::SetPhaseInversionDisabled(value) => {
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::SetPhaseInversionDisabled(value),
            )?;
        }
        OpusEncoderCtlRequest::GetPhaseInversionDisabled(out) => {
            opus_custom_encoder_ctl(
                encoder.celt.encoder(),
                CeltEncoderCtlRequest::GetPhaseInversionDisabled(out),
            )?;
        }
        OpusEncoderCtlRequest::SetDredDuration(value) => {
            if !cfg!(feature = "dred") {
                return Err(OpusEncoderCtlError::Unimplemented);
            }
            if !(0..=DRED_MAX_FRAMES).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.dred_duration = value;
            encoder.silk_mode.use_dred = if value > 0 { 1 } else { 0 };
        }
        OpusEncoderCtlRequest::GetDredDuration(out) => {
            if !cfg!(feature = "dred") {
                return Err(OpusEncoderCtlError::Unimplemented);
            }
            *out = encoder.dred_duration;
        }
        OpusEncoderCtlRequest::SetDnnBlob(data) => {
            if data.is_empty() {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            #[cfg(feature = "dred")]
            {
                dred_encoder_load_model(&mut encoder.dred_encoder, data)
                    .map_err(|_| OpusEncoderCtlError::BadArgument)?;
                encoder.dred_loaded = encoder.dred_encoder.loaded;
            }
            #[cfg(not(feature = "dred"))]
            {
                let _ = data;
                return Err(OpusEncoderCtlError::Unimplemented);
            }
        }
        OpusEncoderCtlRequest::SetForceMode(value) => {
            if value != OPUS_AUTO && !(MODE_SILK_ONLY..=MODE_CELT_ONLY).contains(&value) {
                return Err(OpusEncoderCtlError::BadArgument);
            }
            encoder.user_forced_mode = value;
        }
        OpusEncoderCtlRequest::GetSampleRate(out) => {
            *out = encoder.fs;
        }
        OpusEncoderCtlRequest::GetLookahead(out) => {
            let mut lookahead = encoder.fs / 400;
            if !matches!(encoder.application, OpusApplication::RestrictedLowDelay) {
                lookahead += encoder.delay_compensation;
            }
            *out = lookahead;
        }
        OpusEncoderCtlRequest::GetFinalRange(out) => {
            *out = encoder.range_final;
        }
        OpusEncoderCtlRequest::ResetState => {
            encoder.reset_state()?;
        }
        OpusEncoderCtlRequest::SetLfe(value) => {
            encoder.lfe = value;
            opus_custom_encoder_ctl(encoder.celt.encoder(), CeltEncoderCtlRequest::SetLfe(value))?;
        }
        OpusEncoderCtlRequest::GetLfe(out) => {
            *out = encoder.lfe;
        }
    }
    Ok(())
}

fn encoder_in_dtx(encoder: &OpusEncoder<'_>) -> bool {
    if encoder.silk_mode.use_dtx != 0 && matches!(encoder.prev_mode, MODE_SILK_ONLY | MODE_HYBRID) {
        let first_in_dtx =
            encoder.silk.state_fxx[0].common.no_speech_counter >= NB_SPEECH_FRAMES_BEFORE_DTX;
        if !first_in_dtx {
            return false;
        }

        if encoder.silk.n_channels_internal == 2 && !encoder.silk.prev_decode_only_middle {
            return encoder.silk.state_fxx[1].common.no_speech_counter
                >= NB_SPEECH_FRAMES_BEFORE_DTX;
        }

        return true;
    }

    if encoder.use_dtx {
        return encoder.nb_no_activity_ms_q1 >= NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
    }

    false
}

fn decide_dtx_mode(activity: i32, nb_no_activity_ms_q1: &mut i32, frame_size_ms_q1: i32) -> bool {
    if activity == 0 {
        *nb_no_activity_ms_q1 += frame_size_ms_q1;
        if *nb_no_activity_ms_q1 > NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2 {
            if *nb_no_activity_ms_q1 <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2
            {
                return true;
            }
            *nb_no_activity_ms_q1 = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
        }
    } else {
        *nb_no_activity_ms_q1 = 0;
    }

    false
}

pub fn opus_encode_with_options(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
) -> Result<usize, OpusEncodeError> {
    opus_encode_native_with_options(encoder, pcm, frame_size, data, options, None)
}

fn opus_encode_native_with_options<P: PcmSample>(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[P],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
    analysis_input: Option<OpusAnalysisInput<'_>>,
) -> Result<usize, OpusEncodeError> {
    encoder.range_final = 0;
    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    if channels == 0 || channels > MAX_CHANNELS || frame_size == 0 {
        return Err(OpusEncodeError::BadArgument);
    }
    let required_input = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required_input {
        return Err(OpusEncodeError::BadArgument);
    }
    if data.is_empty() {
        return Err(OpusEncodeError::BadArgument);
    }

    let frame_size_i32 = i32::try_from(frame_size).map_err(|_| OpusEncodeError::BadArgument)?;
    let frame_size_i32 = frame_size_select(frame_size_i32, encoder.variable_duration, encoder.fs)
        .ok_or(OpusEncodeError::BadArgument)?;
    let frame_size = usize::try_from(frame_size_i32).map_err(|_| OpusEncodeError::BadArgument)?;
    if data.len() == 1 && encoder.fs == frame_size_i32 * 10 {
        return Err(OpusEncodeError::BufferTooSmall);
    }
    #[cfg(test)]
    let trace_frame_idx = opus_mode_trace::begin_frame();

    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;

    let frame_rate = encoder
        .fs
        .checked_div(frame_size_i32)
        .ok_or(OpusEncodeError::BadArgument)?;

    let mut max_data_bytes = i32::try_from(data.len()).map_err(|_| OpusEncodeError::BadArgument)?;
    let packet_size_cap = {
        #[cfg(feature = "enable_qext")]
        {
            if encoder.enable_qext { 3825 } else { 1276 }
        }
        #[cfg(not(feature = "enable_qext"))]
        {
            1276
        }
    };
    max_data_bytes = max_data_bytes.min(packet_size_cap * 6);
    encoder.bitrate_bps = user_bitrate_to_bitrate(encoder, frame_size_i32, max_data_bytes);
    if !encoder.use_vbr {
        // CBR accounts for the complete packet, including its TOC byte. The
        // input slice's capacity is only an upper bound on the target size.
        let target_bits =
            i64::from(encoder.bitrate_bps) * i64::from(frame_size_i32) / i64::from(encoder.fs);
        let cbr_bytes = ((target_bits + 4) / 8).min(i64::from(max_data_bytes)) as i32;
        encoder.bitrate_bps =
            (i64::from(cbr_bytes) * 8 * i64::from(encoder.fs) / i64::from(frame_size_i32)) as i32;
        max_data_bytes = cbr_bytes.max(1);
    }

    let lsb_depth = encoder.lsb_depth.min(
        analysis_input
            .as_ref()
            .map_or(P::DEPTH, |input| input.depth),
    );
    #[cfg(feature = "fixed_point")]
    let lsb_depth = lsb_depth.min(crate::celt::MAX_ENCODING_DEPTH as i32);

    #[cfg(not(feature = "fixed_point"))]
    let stereo_width = if encoder.channels == 2 && encoder.force_channels != 1 {
        compute_stereo_width(
            &pcm[..required],
            frame_size,
            encoder.fs,
            &mut encoder.width_mem,
        )
    } else {
        0.0
    };
    #[cfg(feature = "fixed_point")]
    let stereo_width_q15 = if encoder.channels == 2 && encoder.force_channels != 1 {
        compute_stereo_width_fixed(
            &pcm[..required],
            frame_size,
            encoder.fs,
            &mut encoder.width_mem,
        )
    } else {
        0
    };

    let is_silence = is_digital_silence(&pcm[..required], frame_size, channels, lsb_depth);
    let mut analysis_read_state = None;
    {
        encoder.analysis_info.valid = false;
        if encoder.silk_mode.complexity >= if cfg!(feature = "fixed_point") { 10 } else { 7 }
            && encoder.fs >= 16_000
            && encoder.fs <= 48_000
        {
            analysis_read_state = Some(encoder.analysis.snapshot_read_state());
            encoder.analysis.application = encoder.application.to_opus_int();
            let analysis_pcm: Vec<f32> = pcm[..required]
                .iter()
                .map(|sample| sample.normalized())
                .collect();
            let (analysis_pcm, analysis_channels, analysis_c1, analysis_c2) =
                analysis_input.as_ref().map_or(
                    (analysis_pcm.as_slice(), encoder.channels, 0, -2),
                    |input| (input.pcm, input.channels, input.c1, input.c2),
                );
            if analysis_channels <= 0
                || analysis_c1 < 0
                || analysis_c1 >= analysis_channels
                || analysis_c2 >= analysis_channels
                || analysis_pcm.len() < required / channels * analysis_channels as usize
            {
                return Err(OpusEncodeError::BadArgument);
            }
            run_analysis(
                &mut encoder.analysis,
                encoder.celt.mode,
                Some(analysis_pcm),
                frame_size,
                frame_size,
                analysis_c1,
                analysis_c2,
                analysis_channels,
                encoder.fs,
                lsb_depth,
                &mut encoder.analysis_info,
            );
        } else {
            tonality_analysis_reset(&mut encoder.analysis);
        }

        if !is_silence {
            encoder.voice_ratio = -1;
        }
        encoder.detected_bandwidth = 0;
        if encoder.analysis_info.valid {
            if encoder.signal_type == OPUS_AUTO {
                let prob = if encoder.prev_mode == 0 {
                    encoder.analysis_info.music_prob
                } else if encoder.prev_mode == MODE_CELT_ONLY {
                    encoder.analysis_info.music_prob_max
                } else {
                    encoder.analysis_info.music_prob_min
                };
                encoder.voice_ratio = libm::floorf(0.5 + 100.0 * (1.0 - prob)) as i32;
            }

            let analysis_bandwidth = encoder.analysis_info.bandwidth;
            encoder.detected_bandwidth = if analysis_bandwidth <= 12 {
                OPUS_BANDWIDTH_NARROWBAND
            } else if analysis_bandwidth <= 14 {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if analysis_bandwidth <= 16 {
                OPUS_BANDWIDTH_WIDEBAND
            } else if analysis_bandwidth <= 18 {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };
        }
    }

    #[cfg(not(feature = "fixed_point"))]
    if (!encoder.analysis_info.valid || encoder.analysis_info.activity_probability > 0.1)
        && !is_silence
    {
        encoder.peak_signal_energy =
            (0.999 * encoder.peak_signal_energy).max(compute_frame_energy(&pcm[..required]));
    }

    #[cfg(feature = "fixed_point")]
    if (!encoder.analysis_info.valid || encoder.analysis_info.activity_probability > 0.1)
        && !is_silence
    {
        encoder.fixed_peak_signal_energy = (((i64::from(encoder.fixed_peak_signal_energy) * 32735)
            >> 15) as i32)
            .max(compute_frame_energy_fixed(&pcm[..required]));
    }

    #[cfg(feature = "dred")]
    let dred_bitrate_bps = {
        let dred_bitrate = compute_dred_bitrate(encoder, encoder.bitrate_bps, frame_size_i32);
        encoder.bitrate_bps = encoder.bitrate_bps.saturating_sub(dred_bitrate);
        dred_bitrate
    };
    #[cfg(not(feature = "dred"))]
    let dred_bitrate_bps = 0;

    if max_data_bytes < 3
        || encoder.bitrate_bps < 3 * frame_rate * 8
        || (frame_rate < 50 && (max_data_bytes * frame_rate < 300 || encoder.bitrate_bps < 2400))
    {
        let mut mode = if encoder.mode == 0 {
            MODE_SILK_ONLY
        } else {
            encoder.mode
        };
        let mut rate = frame_rate;
        let mut code = 0u8;
        let mut frames = 0u8;
        if rate > 100 {
            mode = MODE_CELT_ONLY;
        }
        if rate == 25 && mode != MODE_SILK_ONLY {
            rate = 50;
            code = 1;
        }
        if rate <= 16 {
            if data.len() == 1 || (mode == MODE_SILK_ONLY && rate != 10) {
                mode = MODE_SILK_ONLY;
                code = u8::from(rate <= 12);
                rate = if rate == 12 { 25 } else { 16 };
            } else {
                frames = (50 / rate) as u8;
                rate = 50;
                code = 3;
            }
        }
        let mut bandwidth = encoder.bandwidth;
        if mode == MODE_SILK_ONLY && bandwidth.to_opus_int() > OPUS_BANDWIDTH_WIDEBAND {
            bandwidth = Bandwidth::Wide;
        } else if mode == MODE_CELT_ONLY && bandwidth == Bandwidth::Medium {
            bandwidth = Bandwidth::Narrow;
        } else if mode == MODE_HYBRID && bandwidth.to_opus_int() <= OPUS_BANDWIDTH_SUPERWIDEBAND {
            bandwidth = Bandwidth::SuperWide;
        }
        let bytes = if code <= 1 { 1 } else { 2 };
        let target = (max_data_bytes as usize).max(bytes);
        if target > data.len() {
            return Err(OpusEncodeError::BufferTooSmall);
        }
        data[0] = gen_toc(mode, rate, bandwidth, encoder.stream_channels) | code;
        if code == 3 {
            data[1] = frames;
        }
        if !encoder.use_vbr {
            opus_packet_pad(data, bytes, target).map_err(|_| OpusEncodeError::InternalError)?;
            return Ok(target);
        }
        return Ok(bytes);
    }

    let mut equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.channels,
        frame_rate,
        encoder.use_vbr,
        0,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    let voice_est = if encoder.signal_type == OPUS_SIGNAL_VOICE {
        127
    } else if encoder.signal_type == OPUS_SIGNAL_MUSIC {
        0
    } else if encoder.voice_ratio >= 0 {
        let mut est = (encoder.voice_ratio * 327) >> 8;
        if matches!(encoder.application, OpusApplication::Audio) {
            est = est.min(115);
        }
        est
    } else if matches!(encoder.application, OpusApplication::Voip) {
        115
    } else {
        48
    };

    let prev_stream_channels = encoder.stream_channels;
    let stream_channels = if encoder.force_channels != OPUS_AUTO && encoder.channels == 2 {
        encoder.force_channels
    } else if encoder.channels == 2 {
        let mut stereo_threshold = STEREO_MUSIC_THRESHOLD
            + ((i64::from(voice_est)
                * i64::from(voice_est)
                * i64::from(STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD))
                >> 14) as i32;
        if prev_stream_channels == 2 {
            stereo_threshold -= 1000;
        } else {
            stereo_threshold += 1000;
        }
        if equiv_rate > stereo_threshold { 2 } else { 1 }
    } else {
        encoder.channels
    };
    encoder.stream_channels = stream_channels;

    equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.stream_channels,
        frame_rate,
        encoder.use_vbr,
        0,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    let silk_use_dtx = encoder.use_dtx && !(encoder.analysis_info.valid || is_silence);
    encoder.silk_mode.use_dtx = i32::from(silk_use_dtx);

    let mut mode = if matches!(encoder.application, OpusApplication::RestrictedLowDelay) {
        MODE_CELT_ONLY
    } else if encoder.user_forced_mode != OPUS_AUTO {
        encoder.user_forced_mode
    } else {
        #[cfg(feature = "fixed_point")]
        let q15_one_minus = Q15_ONE - stereo_width_q15;
        #[cfg(feature = "fixed_point")]
        let mode_voice = ((i64::from(q15_one_minus) * i64::from(MODE_THRESHOLDS[0][0]) >> 15)
            + (i64::from(stereo_width_q15) * i64::from(MODE_THRESHOLDS[1][0]) >> 15))
            as i32;
        #[cfg(feature = "fixed_point")]
        let mode_music = ((i64::from(q15_one_minus) * i64::from(MODE_THRESHOLDS[1][1]) >> 15)
            + (i64::from(stereo_width_q15) * i64::from(MODE_THRESHOLDS[1][1]) >> 15))
            as i32;

        #[cfg(not(feature = "fixed_point"))]
        let mode_voice = ((1.0 - stereo_width) * MODE_THRESHOLDS[0][0] as f32
            + stereo_width * MODE_THRESHOLDS[1][0] as f32) as i32;
        #[cfg(not(feature = "fixed_point"))]
        let mode_music = ((1.0 - stereo_width) * MODE_THRESHOLDS[1][1] as f32
            + stereo_width * MODE_THRESHOLDS[1][1] as f32) as i32;

        let mut threshold = mode_music
            + ((i64::from(voice_est) * i64::from(voice_est) * i64::from(mode_voice - mode_music))
                >> 14) as i32;
        if matches!(encoder.application, OpusApplication::Voip) {
            threshold += 8000;
        }
        if encoder.prev_mode == MODE_CELT_ONLY {
            threshold -= 4000;
        } else if encoder.prev_mode > 0 {
            threshold += 4000;
        }

        let mut selected = if equiv_rate >= threshold {
            MODE_CELT_ONLY
        } else {
            MODE_SILK_ONLY
        };

        if encoder.inband_fec
            && encoder.packet_loss_perc > (128 - voice_est) >> 4
            && (encoder.fec_config != 2 || voice_est > 25)
        {
            selected = MODE_SILK_ONLY;
        }
        if silk_use_dtx && voice_est > 100 {
            selected = MODE_SILK_ONLY;
        }
        let rate_threshold = if frame_rate > 50 { 9000 } else { 6000 };
        let threshold_bytes =
            i64::from(rate_threshold) * i64::from(frame_size_i32) / i64::from(encoder.fs * 8);
        if i64::from(max_data_bytes) < threshold_bytes {
            selected = MODE_CELT_ONLY;
        }
        selected
    };

    let min_celt = encoder.fs / 100;
    if encoder.lfe || (mode != MODE_CELT_ONLY && frame_size_i32 < min_celt) {
        mode = MODE_CELT_ONLY;
    }

    let mut redundancy = false;
    let mut celt_to_silk = false;
    let mut to_celt = false;
    let mut prefill = PrefillMode::None;
    if encoder.prev_mode > 0
        && ((mode != MODE_CELT_ONLY && encoder.prev_mode == MODE_CELT_ONLY)
            || (mode == MODE_CELT_ONLY && encoder.prev_mode != MODE_CELT_ONLY))
    {
        redundancy = true;
        celt_to_silk = mode != MODE_CELT_ONLY;
        if !celt_to_silk {
            if frame_size_i32 >= min_celt {
                mode = encoder.prev_mode;
                to_celt = true;
            } else {
                redundancy = false;
            }
        }
    }

    if encoder.stream_channels == 1
        && encoder.prev_channels == 2
        && !encoder.silk_mode.to_mono
        && mode != MODE_CELT_ONLY
        && encoder.prev_mode != MODE_CELT_ONLY
    {
        encoder.silk_mode.to_mono = true;
        encoder.stream_channels = 2;
    } else {
        encoder.silk_mode.to_mono = false;
    }

    equiv_rate = compute_equiv_rate(
        encoder.bitrate_bps,
        encoder.stream_channels,
        frame_rate,
        encoder.use_vbr,
        mode,
        encoder.complexity,
        encoder.packet_loss_perc,
    );

    if mode != MODE_CELT_ONLY && encoder.prev_mode == MODE_CELT_ONLY {
        let mut dummy = SilkEncControl::default();
        silk_init_encoder(&mut encoder.silk, encoder.arch, &mut dummy)?;
        prefill = PrefillMode::Prefill;
    }

    let mut bandwidth_int = encoder.bandwidth.to_opus_int();
    if mode == MODE_CELT_ONLY || encoder.first || encoder.silk_mode.allow_bandwidth_switch {
        let (voice_thresholds, music_thresholds) =
            if encoder.channels == 2 && encoder.force_channels != 1 {
                (
                    &STEREO_VOICE_BANDWIDTH_THRESHOLDS,
                    &STEREO_MUSIC_BANDWIDTH_THRESHOLDS,
                )
            } else {
                (
                    &MONO_VOICE_BANDWIDTH_THRESHOLDS,
                    &MONO_MUSIC_BANDWIDTH_THRESHOLDS,
                )
            };
        let mut bandwidth_thresholds = [0i32; 8];
        for i in 0..bandwidth_thresholds.len() {
            bandwidth_thresholds[i] = music_thresholds[i]
                + ((i64::from(voice_est)
                    * i64::from(voice_est)
                    * i64::from(voice_thresholds[i] - music_thresholds[i]))
                    >> 14) as i32;
        }

        bandwidth_int = OPUS_BANDWIDTH_FULLBAND;
        loop {
            let idx = usize::try_from(2 * (bandwidth_int - OPUS_BANDWIDTH_MEDIUMBAND)).unwrap_or(0);
            let mut threshold = *bandwidth_thresholds.get(idx).unwrap_or(&0);
            let hysteresis = *bandwidth_thresholds.get(idx + 1).unwrap_or(&0);
            if !encoder.first {
                if encoder.auto_bandwidth >= bandwidth_int {
                    threshold -= hysteresis;
                } else {
                    threshold += hysteresis;
                }
            }
            if equiv_rate >= threshold {
                break;
            }
            if bandwidth_int <= OPUS_BANDWIDTH_NARROWBAND {
                break;
            }
            bandwidth_int -= 1;
        }
        if bandwidth_int == OPUS_BANDWIDTH_MEDIUMBAND {
            bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
        }
        encoder.auto_bandwidth = bandwidth_int;
        if !encoder.first
            && mode != MODE_CELT_ONLY
            && !encoder.silk_mode.in_wb_mode_without_variable_lp
            && bandwidth_int > OPUS_BANDWIDTH_WIDEBAND
        {
            bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
        }
    }

    if bandwidth_int > encoder.max_bandwidth {
        bandwidth_int = encoder.max_bandwidth;
    }
    if encoder.user_bandwidth != OPUS_AUTO {
        bandwidth_int = encoder.user_bandwidth;
    }
    let max_rate = frame_rate.saturating_mul(max_data_bytes).saturating_mul(8);
    if mode != MODE_CELT_ONLY && max_rate < 15_000 {
        bandwidth_int = bandwidth_int.min(OPUS_BANDWIDTH_WIDEBAND);
    }
    if encoder.fs <= 24_000 && bandwidth_int > OPUS_BANDWIDTH_SUPERWIDEBAND {
        bandwidth_int = OPUS_BANDWIDTH_SUPERWIDEBAND;
    }
    if encoder.fs <= 16_000 && bandwidth_int > OPUS_BANDWIDTH_WIDEBAND {
        bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
    }
    if encoder.fs <= 12_000 && bandwidth_int > OPUS_BANDWIDTH_MEDIUMBAND {
        bandwidth_int = OPUS_BANDWIDTH_MEDIUMBAND;
    }
    if encoder.fs <= 8_000 && bandwidth_int > OPUS_BANDWIDTH_NARROWBAND {
        bandwidth_int = OPUS_BANDWIDTH_NARROWBAND;
    }
    if encoder.detected_bandwidth != 0 && encoder.user_bandwidth == OPUS_AUTO {
        let min_detected_bandwidth =
            if equiv_rate <= 18_000 * encoder.stream_channels && mode == MODE_CELT_ONLY {
                OPUS_BANDWIDTH_NARROWBAND
            } else if equiv_rate <= 24_000 * encoder.stream_channels && mode == MODE_CELT_ONLY {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if equiv_rate <= 30_000 * encoder.stream_channels {
                OPUS_BANDWIDTH_WIDEBAND
            } else if equiv_rate <= 44_000 * encoder.stream_channels {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };
        encoder.detected_bandwidth = encoder.detected_bandwidth.max(min_detected_bandwidth);
        bandwidth_int = bandwidth_int.min(encoder.detected_bandwidth);
    }

    let use_fec = decide_fec(
        encoder.inband_fec,
        encoder.packet_loss_perc,
        encoder.silk_mode.lbrr_coded != 0,
        mode,
        &mut bandwidth_int,
        equiv_rate,
    );
    encoder.silk_mode.lbrr_coded = i32::from(use_fec);

    if mode == MODE_CELT_ONLY && bandwidth_int == OPUS_BANDWIDTH_MEDIUMBAND {
        bandwidth_int = OPUS_BANDWIDTH_WIDEBAND;
    }

    if encoder.lfe {
        bandwidth_int = OPUS_BANDWIDTH_NARROWBAND;
    }
    let curr_bandwidth_int = bandwidth_int;
    if mode == MODE_SILK_ONLY && curr_bandwidth_int > OPUS_BANDWIDTH_WIDEBAND {
        mode = MODE_HYBRID;
    }
    if mode == MODE_HYBRID && curr_bandwidth_int <= OPUS_BANDWIDTH_WIDEBAND {
        mode = MODE_SILK_ONLY;
    }

    let bandwidth = Bandwidth::from_opus_int(bandwidth_int).unwrap_or(Bandwidth::Wide);
    encoder.bandwidth = bandwidth;
    #[cfg(test)]
    if let Some(frame_idx) = trace_frame_idx {
        opus_mode_trace::dump_if_match(
            frame_idx,
            mode,
            encoder.prev_mode,
            equiv_rate,
            bandwidth_int,
            encoder.stream_channels,
            encoder.voice_ratio,
            is_silence,
            &encoder.analysis_info,
        );
    }

    let max_celt = usize::try_from(encoder.fs / 50).map_err(|_| OpusEncodeError::BadArgument)?;
    if mode == MODE_SILK_ONLY && frame_size * 2 != max_celt {
        let multiples = frame_size / max_celt;
        if !frame_size.is_multiple_of(max_celt) || !(1..=6).contains(&multiples) {
            return Err(OpusEncodeError::Unimplemented);
        }
    }

    if (frame_size > max_celt && mode != MODE_SILK_ONLY) || frame_size > max_celt * 3 {
        let fs = encoder.fs;
        let enc_frame_size_i32 = if mode == MODE_SILK_ONLY {
            if frame_size_i32 == 2 * fs / 25 {
                fs / 25
            } else if frame_size_i32 == 3 * fs / 25 {
                3 * fs / 50
            } else {
                fs / 50
            }
        } else {
            fs / 50
        };
        if enc_frame_size_i32 <= 0 || frame_size_i32 % enc_frame_size_i32 != 0 {
            return Err(OpusEncodeError::Unimplemented);
        }
        let enc_frame_size =
            usize::try_from(enc_frame_size_i32).map_err(|_| OpusEncodeError::BadArgument)?;
        let nb_frames = frame_size / enc_frame_size;
        if !(1..=6).contains(&nb_frames) {
            return Err(OpusEncodeError::Unimplemented);
        }

        if let Some(read_state) = analysis_read_state {
            encoder.analysis.restore_read_state(read_state);
        }

        let max_header_bytes = if nb_frames == 2 {
            3
        } else {
            2 + (nb_frames - 1) * 2
        };
        #[cfg(feature = "enable_qext")]
        let max_header_bytes = max_header_bytes + if encoder.enable_qext { nb_frames } else { 0 };
        let nb_frames_i32 = i32::try_from(nb_frames).map_err(|_| OpusEncodeError::BadArgument)?;
        let max_header_bytes_i32 =
            i32::try_from(max_header_bytes).map_err(|_| OpusEncodeError::BadArgument)?;
        let max_len_sum = nb_frames_i32
            .checked_add(max_data_bytes)
            .and_then(|value| value.checked_sub(max_header_bytes_i32))
            .ok_or(OpusEncodeError::BufferTooSmall)?;
        if max_len_sum < 2 * nb_frames_i32 {
            return Err(OpusEncodeError::BufferTooSmall);
        }
        let max_len_sum_usize =
            usize::try_from(max_len_sum).map_err(|_| OpusEncodeError::BadArgument)?;
        if max_len_sum_usize > MAX_REPACKETIZER_BYTES {
            return Err(OpusEncodeError::BufferTooSmall);
        }

        let mut tmp_data = [0u8; MAX_REPACKETIZER_BYTES];
        let tmp_data = &mut tmp_data[..max_len_sum_usize];
        let mut repacketizer = OpusRepacketizer::new();
        repacketizer.opus_repacketizer_init();

        let bak_to_mono = encoder.silk_mode.to_mono;
        if bak_to_mono {
            encoder.force_channels = 1;
        } else {
            encoder.prev_channels = encoder.stream_channels;
        }

        let mut tot_size = 0i32;
        let mut dtx_count = 0usize;
        for frame_idx in 0..nb_frames {
            let first_frame = frame_idx == 0 || frame_idx == dtx_count;
            encoder.silk_mode.to_mono = false;
            encoder.nonfinal_frame = frame_idx < nb_frames - 1;
            let frame_to_celt = to_celt && frame_idx == nb_frames - 1;
            let frame_redundancy = redundancy && (frame_to_celt || (!to_celt && frame_idx == 0));

            let frames_left =
                i32::try_from(nb_frames - frame_idx).map_err(|_| OpusEncodeError::BadArgument)?;
            let max_len_per_frame = (max_len_sum - tot_size) / frames_left;
            let rate_bytes = (i64::from(encoder.bitrate_bps) * i64::from(enc_frame_size_i32))
                / (8 * i64::from(encoder.fs));
            let rate_bytes = rate_bytes.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            // Each subframe receives the same initial budget. Redistributing
            // the header remainder to the last frame changes its bitstream.
            let curr_max = rate_bytes.min(max_len_sum / nb_frames_i32);
            #[cfg(feature = "dred")]
            let curr_max = {
                // Reserve DRED for the first non-DTX frame of the complete packet.
                let dred_bytes = ((i64::from(dred_bitrate_bps) * i64::from(frame_size_i32))
                    / (8 * i64::from(encoder.fs))) as i32;
                let core_bytes = curr_max.min((max_len_sum - dred_bytes) / nb_frames_i32);
                core_bytes + if first_frame { dred_bytes } else { 0 }
            };
            #[cfg(feature = "enable_qext")]
            let curr_max = if encoder.enable_qext {
                // Padding lacing bytes consume one byte per 254 payload bytes.
                curr_max - curr_max / 254
            } else {
                curr_max
            };
            let curr_max = curr_max.min(max_len_sum - tot_size);
            #[cfg(feature = "dred")]
            let max_len_per_frame = max_len_per_frame.max(curr_max);
            if curr_max < 2 {
                return Err(OpusEncodeError::BufferTooSmall);
            }

            let tot_size_usize =
                usize::try_from(tot_size).map_err(|_| OpusEncodeError::BadArgument)?;
            let max_len_per_frame_usize =
                usize::try_from(max_len_per_frame).map_err(|_| OpusEncodeError::BadArgument)?;
            if tot_size_usize + max_len_per_frame_usize > tmp_data.len() {
                return Err(OpusEncodeError::BufferTooSmall);
            }
            let curr_max_usize =
                usize::try_from(curr_max).map_err(|_| OpusEncodeError::BadArgument)?;

            if analysis_read_state.is_some() {
                tonality_get_info(
                    &mut encoder.analysis,
                    &mut encoder.analysis_info,
                    enc_frame_size,
                );
            }

            let start = frame_idx
                .checked_mul(enc_frame_size)
                .and_then(|value| value.checked_mul(channels))
                .ok_or(OpusEncodeError::BadArgument)?;
            let end = start
                .checked_add(enc_frame_size * channels)
                .ok_or(OpusEncodeError::BadArgument)?;
            let is_silence =
                is_digital_silence(&pcm[start..end], enc_frame_size, channels, lsb_depth);
            let frame_buf = &mut tmp_data[tot_size_usize..tot_size_usize + max_len_per_frame_usize];
            let len = match encode_frame_native(
                encoder,
                options.energy_masking,
                &pcm[start..end],
                enc_frame_size,
                &mut frame_buf[..curr_max_usize],
                lsb_depth,
                silk_use_dtx,
                is_silence,
                frame_redundancy,
                celt_to_silk,
                prefill,
                bandwidth_int,
                mode,
                frame_to_celt,
                equiv_rate,
                first_frame,
                dred_bitrate_bps,
            ) {
                Ok(len) => len,
                Err(OpusEncodeError::BufferTooSmall) if curr_max < max_len_per_frame => {
                    encode_frame_native(
                        encoder,
                        options.energy_masking,
                        &pcm[start..end],
                        enc_frame_size,
                        &mut frame_buf[..max_len_per_frame_usize],
                        lsb_depth,
                        silk_use_dtx,
                        is_silence,
                        frame_redundancy,
                        celt_to_silk,
                        prefill,
                        bandwidth_int,
                        mode,
                        frame_to_celt,
                        equiv_rate,
                        first_frame,
                        dred_bitrate_bps,
                    )?
                }
                Err(err) => return Err(err),
            };
            if len == 1 {
                dtx_count += 1;
            }
            repacketizer
                .opus_repacketizer_cat(&frame_buf[..len], len)
                .map_err(|err| match err {
                    RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                    _ => OpusEncodeError::InternalError,
                })?;
            tot_size = tot_size.saturating_add(len as i32);
        }

        let maxlen = usize::try_from(max_data_bytes).map_err(|_| OpusEncodeError::BadArgument)?;
        if maxlen > data.len() {
            return Err(OpusEncodeError::BufferTooSmall);
        }
        let mut written = repacketizer
            .opus_repacketizer_out(&mut data[..], maxlen)
            .map_err(|err| match err {
                RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;
        if !encoder.use_vbr && dtx_count != nb_frames {
            opus_packet_pad(data, written, maxlen).map_err(|err| match err {
                RepacketizerError::BufferTooSmall => OpusEncodeError::BufferTooSmall,
                _ => OpusEncodeError::InternalError,
            })?;
            written = maxlen;
        }

        encoder.silk_mode.to_mono = bak_to_mono;
        encoder.nonfinal_frame = false;
        return Ok(written);
    }

    encoder.nonfinal_frame = false;
    encode_frame_native(
        encoder,
        options.energy_masking,
        &pcm[..required],
        frame_size,
        &mut data[..max_data_bytes as usize],
        lsb_depth,
        silk_use_dtx,
        is_silence,
        redundancy,
        celt_to_silk,
        prefill,
        bandwidth_int,
        mode,
        to_celt,
        equiv_rate,
        true,
        dred_bitrate_bps,
    )
}

pub fn opus_encode(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusEncodeError> {
    opus_encode_with_options(encoder, pcm, frame_size, data, OpusEncodeOptions::default())
}

/// Wrapper for encoding 24-bit PCM stored in `i32`, mirroring `opus_encode24`.
pub fn opus_encode24_with_options(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
) -> Result<usize, OpusEncodeError> {
    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }

    opus_encode_native_with_options(encoder, pcm, frame_size, data, options, None)
}

pub fn opus_encode24(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusEncodeError> {
    opus_encode24_with_options(encoder, pcm, frame_size, data, OpusEncodeOptions::default())
}

pub fn opus_encode_float_with_options(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
) -> Result<usize, OpusEncodeError> {
    let channels = usize::try_from(encoder.channels).map_err(|_| OpusEncodeError::BadArgument)?;
    let required = channels
        .checked_mul(frame_size)
        .ok_or(OpusEncodeError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusEncodeError::BadArgument);
    }

    opus_encode_native_with_options(encoder, pcm, frame_size, data, options, None)
}

pub fn opus_encode_float(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusEncodeError> {
    opus_encode_float_with_options(encoder, pcm, frame_size, data, OpusEncodeOptions::default())
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{
        CELT_SIG_SCALE, DELAY_BUFFER_SAMPLES, DRED_MAX_FRAMES, MAX_PCM_BUF_SAMPLES,
        MAX_TMP_PREFILL_SAMPLES, MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_AUTO,
        OPUS_BANDWIDTH_NARROWBAND, OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND,
        OPUS_BITRATE_MAX, OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_40_MS, OPUS_SIGNAL_MUSIC,
        OpusEncodeError, OpusEncodeOptions, OpusEncoderCtlError, OpusEncoderCtlRequest,
        OpusEncoderInitError, OpusRes, StereoWidthState, VARIABLE_HP_MIN_CUTOFF_HZ,
        VARIABLE_HP_SMTH_COEF2_Q16, VERY_SMALL, compute_equiv_rate, compute_redundancy_bytes,
        compute_silk_rate_for_hybrid, compute_stereo_width, dc_reject, decide_fec, hp_cutoff,
        lin2log, log2lin, opus_encode, opus_encode_with_options, opus_encode24,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size,
        prepare_celt_prefill_from_delay, prepare_silk_prefill, update_high_pass_state,
        user_bitrate_to_bitrate,
    };
    #[cfg(feature = "dred")]
    use super::{
        adjust_nb_compr_bytes_for_dred, compute_dred_bitrate, update_dred_activity_history,
    };
    use crate::packet::{
        Bandwidth, opus_packet_get_bandwidth, opus_packet_get_mode, opus_packet_get_nb_frames,
        opus_packet_get_samples_per_frame,
    };
    use alloc::vec;
    use alloc::vec::Vec;

    #[cfg(feature = "fixed_point")]
    #[test]
    fn fixed_frontend_matches_pinned_c() {
        let sample = |frame: usize, i: usize, pattern: usize| -> i16 {
            let value = (i as u32 + 1)
                .wrapping_mul(2654435761)
                .wrapping_add((frame as u32).wrapping_mul(2246822519));
            match pattern {
                0 => (value >> 16) as i16,
                1 => {
                    if i & 1 != 0 {
                        2000
                    } else {
                        27000
                    }
                }
                _ => {
                    if i & 2 != 0 {
                        -12345
                    } else {
                        12345
                    }
                }
            }
        };
        let mut state = StereoWidthState::default();
        let gains = [0, 8192, 16384, 24576, 32767];
        #[cfg(all(not(feature = "enable_res24"), not(feature = "enable_qext")))]
        let fixture = include_str!("../tests/fixtures/reference/fixed-frontend.txt");
        #[cfg(all(feature = "enable_res24", not(feature = "enable_qext")))]
        let fixture = include_str!("../tests/fixtures/reference/fixed-frontend-res24.txt");
        #[cfg(all(not(feature = "enable_res24"), feature = "enable_qext"))]
        let fixture = include_str!("../tests/fixtures/reference/fixed-frontend-qext.txt");
        #[cfg(all(feature = "enable_res24", feature = "enable_qext"))]
        let fixture = include_str!("../tests/fixtures/reference/fixed-frontend-res24-qext.txt");
        for line in fixture.lines() {
            let fields: alloc::vec::Vec<_> = line.split_whitespace().collect();
            let num = |i: usize| fields[i].parse::<i32>().unwrap();
            let rate = num(1);
            if fields[0] == "P" {
                #[cfg(feature = "enable_res24")]
                {
                    let channels = rate as usize;
                    let upsample = num(2) as usize;
                    let n = 32 * upsample;
                    let pcm: Vec<i32> = (0..32 * channels)
                        .map(|i| i as i32 * 167111 - 18331119)
                        .collect();
                    let mut signal = alloc::vec![0i32;n];
                    let mut memory = 123456;
                    crate::celt::celt_preemphasis_fixed(
                        &pcm,
                        &mut signal,
                        n,
                        channels,
                        upsample,
                        &crate::celt::canonical_mode().unwrap().pre_emphasis,
                        &mut memory,
                        true,
                    );
                    assert_eq!(memory, num(3), "{line}");
                    let mut hash = 14695981039346656037u64;
                    for value in signal {
                        for byte in value.to_le_bytes() {
                            hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211);
                        }
                    }
                    assert_eq!(hash, u64::from_str_radix(fields[4], 16).unwrap(), "{line}");
                }
            } else if fields[0] == "H" {
                assert_eq!(i32::from(super::fixed_hb_gain(rate)), num(2), "{line}");
            } else if fields[0] == "W" {
                let pattern = num(2) as usize;
                let frame = num(3) as usize;
                if frame == 0 {
                    state = StereoWidthState::default();
                }
                let n = (rate / 50) as usize;
                let pcm: alloc::vec::Vec<_> =
                    (0..2 * n).map(|i| sample(frame, i, pattern)).collect();
                assert_eq!(
                    super::compute_stereo_width_fixed(&pcm, n, rate, &mut state),
                    num(4),
                    "{line}"
                );
                for i in 0..5 {
                    assert_eq!(state.fixed[i], num(5 + i), "{line}");
                }
            } else {
                let a = num(2) as usize;
                let b = num(3) as usize;
                let channels = num(4) as usize;
                let n = (rate / 100) as usize;
                let mut pcm: alloc::vec::Vec<_> = (0..channels * n)
                    .map(|i| f32::from(sample(a, i, b % 3)) / 32768.0)
                    .collect();
                #[cfg(feature = "enable_qext")]
                let mode = if rate == 96000 {
                    crate::celt::canonical_mode_96k().unwrap()
                } else {
                    crate::celt::canonical_mode().unwrap()
                };
                #[cfg(not(feature = "enable_qext"))]
                let mode = crate::celt::canonical_mode().unwrap();
                super::fade_fixed(
                    &mut pcm,
                    gains[a] as f32 / 32768.0,
                    gains[b] as f32 / 32768.0,
                    mode.overlap,
                    n,
                    channels,
                    mode.window,
                    rate,
                    fields[0] == "S",
                );
                let mut hash = 14695981039346656037u64;
                for value in pcm {
                    for byte in (i32::from(crate::celt::float2res(value))).to_le_bytes() {
                        hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211);
                    }
                }
                assert_eq!(hash, u64::from_str_radix(fields[5], 16).unwrap(), "{line}");
            }
        }
    }

    #[test]
    fn encoder_get_size_matches_components() {
        let size = opus_encoder_get_size(1).expect("size");
        assert!(size > 0);
    }

    #[test]
    fn create_rejects_invalid_arguments() {
        assert_eq!(
            opus_encoder_create(44_100, 1, 2048).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
        assert_eq!(
            opus_encoder_create(48_000, 3, 2048).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
        assert_eq!(
            opus_encoder_create(48_000, 1, 123).unwrap_err(),
            OpusEncoderInitError::BadArgument
        );
    }

    #[test]
    fn init_sets_hybrid_state_defaults() {
        let enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(enc.voice_ratio, -1);
        assert_eq!(enc.encoder_buffer, enc.fs / 100);
        assert_eq!(enc.hybrid_stereo_width_q14, 1_i16 << 14);
        assert_eq!(
            enc.variable_hp_smth2_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
        assert_eq!(enc.prev_hb_gain, 1.0);
        assert!(enc.hp_mem.iter().all(|&value| value == 0.0));
        assert_eq!(enc.delay_compensation, enc.fs / 250);
        assert_eq!(enc.mode, MODE_HYBRID);
        assert_eq!(enc.bandwidth, Bandwidth::Full);
        assert_eq!(enc.auto_bandwidth, 0);
        assert_eq!(enc.prev_mode, 0);
        assert_eq!(enc.prev_channels, 0);
        assert_eq!(enc.prev_framesize, 0);
        assert!(!enc.silk_bw_switch);
        assert!(enc.first);
        assert_eq!(enc.delay_buffer.len(), DELAY_BUFFER_SAMPLES);
        assert!(enc.delay_buffer.iter().all(|&value| value == 0.0));
        assert_eq!(enc.width_mem.xx, 0.0);
        assert_eq!(enc.width_mem.xy, 0.0);
        assert_eq!(enc.width_mem.yy, 0.0);
        assert_eq!(enc.width_mem.smoothed_width, 0.0);
        assert_eq!(enc.width_mem.max_follower, 0.0);
        assert!(!enc.nonfinal_frame);
        #[cfg(not(feature = "fixed_point"))]
        assert_eq!(enc.detected_bandwidth, 0);
        #[cfg(feature = "dred")]
        {
            assert_eq!(enc.dred_duration, 0);
            assert!(enc.dred_loaded);
            assert_eq!(enc.dred_latents_buffer_fill, 0);
            assert!(enc.dred_activity_mem.iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn prepare_pcm_buffer_includes_delay_history() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        for (idx, sample) in enc.delay_buffer.iter_mut().enumerate() {
            *sample = idx as f32;
        }
        let frame_size = 480;
        let channels = 1usize;
        let pcm = vec![1000i16; frame_size * channels];
        let mut scratch = [OpusRes::default(); MAX_PCM_BUF_SAMPLES];

        let used = super::prepare_pcm_buffer(&enc, &pcm, frame_size, channels, &mut scratch)
            .expect("prepare pcm");
        let total_buffer = enc.delay_compensation as usize;
        let delay_len = total_buffer * channels;
        let delay_start = (enc.encoder_buffer as usize - total_buffer) * channels;

        assert_eq!(used, (total_buffer + frame_size) * channels);
        assert_eq!(
            &scratch[..delay_len],
            &enc.delay_buffer[delay_start..delay_start + delay_len]
        );
        let expected = f32::from(pcm[0]) * (1.0 / CELT_SIG_SCALE);
        assert_eq!(scratch[delay_len], expected);
    }

    #[test]
    fn silk_prefill_ramps_delay_buffer_and_converts() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        for sample in enc.delay_buffer.iter_mut() {
            *sample = 0.5;
        }
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        enc.native_delay_buffer.fill(crate::celt::float2res(0.5));
        let channels = 1usize;
        let mut prefill_buf = [0i16; DELAY_BUFFER_SAMPLES];
        let prefill_len =
            prepare_silk_prefill(&mut enc, channels, &mut prefill_buf).expect("prefill");
        let encoder_buffer = enc.encoder_buffer as usize;
        assert_eq!(prefill_len, encoder_buffer * channels);

        let ramp_samples = (enc.fs / 400) as usize;
        let delay_comp = enc.delay_compensation as usize;
        let prefill_offset = (encoder_buffer - delay_comp - ramp_samples) * channels;
        assert!(
            enc.delay_buffer[..prefill_offset]
                .iter()
                .all(|&value| value == 0.0)
        );

        if delay_comp > 0 {
            let tail_index = prefill_len - 1;
            assert!((enc.delay_buffer[tail_index] - 0.5).abs() < 1e-6);
            let expected = libm::roundf(0.5 * CELT_SIG_SCALE) as i16;
            assert_eq!(prefill_buf[tail_index], expected);
        }
        assert_eq!(prefill_buf[0], 0);
    }

    #[test]
    fn celt_prefill_copies_delay_buffer_tail() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        for (idx, sample) in enc.delay_buffer.iter_mut().enumerate() {
            *sample = idx as f32;
        }
        let channels = 2usize;
        let total_buffer = enc.delay_compensation as usize;
        let mut tmp_prefill = [OpusRes::default(); MAX_TMP_PREFILL_SAMPLES];
        let prefill_len =
            prepare_celt_prefill_from_delay(&enc, channels, total_buffer, &mut tmp_prefill)
                .expect("prefill");
        let encoder_buffer = enc.encoder_buffer as usize;
        let prefill_samples = (enc.fs / 400) as usize;
        let expected_start = (encoder_buffer - total_buffer - prefill_samples) * channels;

        assert_eq!(prefill_len, prefill_samples * channels);
        assert_eq!(
            &tmp_prefill[..prefill_len],
            &enc.delay_buffer[expected_start..expected_start + prefill_len]
        );
    }

    #[test]
    fn update_delay_buffer_shifts_when_frame_smaller_than_encoder_buffer() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let channels = 1usize;
        let frame_size = 240usize;
        let total_buffer = enc.delay_compensation as usize;
        let encoder_buffer = enc.encoder_buffer as usize;
        assert!(encoder_buffer > frame_size + total_buffer);

        for (idx, sample) in enc.delay_buffer.iter_mut().enumerate() {
            *sample = idx as f32;
        }
        let original = enc.delay_buffer;

        let frame_total = frame_size + total_buffer;
        let mut pcm_buf = vec![0.0f32; frame_total * channels];
        for (idx, sample) in pcm_buf.iter_mut().enumerate() {
            *sample = 1000.0 + idx as f32;
        }

        super::update_delay_buffer(&mut enc, &pcm_buf, frame_size, total_buffer, channels)
            .expect("update delay buffer");

        let move_len = (encoder_buffer - frame_total) * channels;
        let src_start = frame_size * channels;
        let src_end = src_start + move_len;
        assert_eq!(&enc.delay_buffer[..move_len], &original[src_start..src_end]);

        let dst_end = move_len + frame_total * channels;
        assert_eq!(
            &enc.delay_buffer[move_len..dst_end],
            &pcm_buf[..frame_total * channels]
        );

        let encoder_samples = encoder_buffer * channels;
        assert_eq!(
            &enc.delay_buffer[encoder_samples..],
            &original[encoder_samples..]
        );
    }

    #[test]
    fn update_delay_buffer_copies_tail_when_frame_exceeds_encoder_buffer() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let channels = 2usize;
        let frame_size = 480usize;
        let total_buffer = enc.delay_compensation as usize;
        let encoder_buffer = enc.encoder_buffer as usize;
        assert!(encoder_buffer <= frame_size + total_buffer);

        let frame_total = frame_size + total_buffer;
        let mut pcm_buf = vec![0.0f32; frame_total * channels];
        for (idx, sample) in pcm_buf.iter_mut().enumerate() {
            *sample = -500.0 + idx as f32;
        }

        super::update_delay_buffer(&mut enc, &pcm_buf, frame_size, total_buffer, channels)
            .expect("update delay buffer");

        let encoder_samples = encoder_buffer * channels;
        let expected_start = (frame_total - encoder_buffer) * channels;
        assert_eq!(
            &enc.delay_buffer[..encoder_samples],
            &pcm_buf[expected_start..expected_start + encoder_samples]
        );
    }

    #[test]
    fn update_high_pass_state_smooths_towards_silk_target() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let prev = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        let target = lin2log(80) << 8;
        enc.variable_hp_smth2_q15 = prev;
        enc.silk.state_fxx[0].common_mut().variable_hp_smth1_q15 = target;

        let expected = prev
            + (((i64::from(target - prev) * i64::from(VARIABLE_HP_SMTH_COEF2_Q16 as i16)) >> 16)
                as i32);
        let cutoff = update_high_pass_state(&mut enc, MODE_HYBRID);

        assert_eq!(enc.variable_hp_smth2_q15, expected);
        assert_eq!(cutoff, log2lin(expected >> 8));
    }

    #[test]
    fn update_high_pass_state_uses_min_cutoff_for_celt() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let prev = lin2log(80) << 8;
        let target = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        enc.variable_hp_smth2_q15 = prev;
        enc.silk.state_fxx[0].common_mut().variable_hp_smth1_q15 = lin2log(100) << 8;

        let expected = prev
            + (((i64::from(target - prev) * i64::from(VARIABLE_HP_SMTH_COEF2_Q16 as i16)) >> 16)
                as i32);
        let cutoff = update_high_pass_state(&mut enc, MODE_CELT_ONLY);

        assert_eq!(enc.variable_hp_smth2_q15, expected);
        assert_eq!(cutoff, log2lin(expected >> 8));
    }

    #[test]
    fn dc_reject_matches_reference_steps() {
        let pcm = [1000i16; 3];
        let mut out = [0.0f32; 3];
        let mut mem = [0.0f32; 4];

        dc_reject(&pcm, 3, &mut out, &mut mem, 3, 1, 48_000);

        let scale = 1.0 / CELT_SIG_SCALE;
        let x = 1000.0 * scale;
        let coef = 6.3f32 * 3.0 / 48_000.0;
        let coef2 = 1.0 - coef;
        let mut m0 = 0.0f32;
        let mut expected = [0.0f32; 3];
        for value in expected.iter_mut() {
            *value = x - m0;
            m0 = coef * x + VERY_SMALL + coef2 * m0;
        }

        for idx in 0..3 {
            assert!((out[idx] - expected[idx]).abs() < 1.0e-6);
        }
        assert!((mem[0] - m0).abs() < 1.0e-6);
        assert_eq!(mem[2], 0.0);
    }

    #[test]
    fn hp_cutoff_modifies_signal_and_state() {
        let pcm = [1000i16; 16];
        let mut out = [0.0f32; 16];
        let mut mem = [0.0f32; 4];

        hp_cutoff(
            &pcm,
            VARIABLE_HP_MIN_CUTOFF_HZ,
            &mut out,
            &mut mem,
            16,
            1,
            48_000,
        );

        let reference = 1000.0 * (1.0 / CELT_SIG_SCALE);
        assert!(out.iter().any(|&value| (value - reference).abs() > 1.0e-6));
        assert!(mem[0] != 0.0 || mem[1] != 0.0);
    }

    #[test]
    fn encodes_silk_only_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetBitrate(12_000)).unwrap();
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert!(len > 1);
        assert_eq!(
            opus_packet_get_mode(&out[..len]).unwrap(),
            crate::packet::Mode::SILK
        );
        assert_eq!(
            opus_packet_get_bandwidth(&out[..len]).unwrap(),
            Bandwidth::Wide
        );
    }

    #[test]
    fn encodes_celt_only_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY),
        )
        .unwrap();
        let pcm = [0i16; 480];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 480, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(
            opus_packet_get_mode(&out[..len]).unwrap(),
            crate::packet::Mode::CELT
        );
        assert_eq!(
            opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(),
            480
        );
    }

    #[test]
    fn encodes_hybrid_frame_with_valid_toc() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID)).unwrap();
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(
            opus_packet_get_mode(&out[..len]).unwrap(),
            crate::packet::Mode::HYBRID
        );
        assert_eq!(
            opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(),
            960
        );
    }

    #[test]
    fn celt_start_band_resets_when_switching_to_celt_only() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID)).unwrap();
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        opus_encode(&mut enc, &pcm, 960, &mut out).expect("hybrid encode");
        assert_eq!(enc.celt.encoder().start_band, 17);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY),
        )
        .unwrap();
        opus_encode(&mut enc, &pcm, 960, &mut out).expect("celt encode");
        assert_eq!(enc.celt.encoder().start_band, 0);
    }

    #[test]
    fn encodes_hybrid_multiframe_packet() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceMode(MODE_HYBRID)).unwrap();
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_40_MS),
        )
        .unwrap();
        let pcm = [0i16; 1920];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 1920, &mut out).expect("encode");
        assert!(len >= 1);
        assert_eq!(
            opus_packet_get_mode(&out[..len]).unwrap(),
            crate::packet::Mode::HYBRID
        );
        assert_eq!(opus_packet_get_nb_frames(&out[..len], len).unwrap(), 2);
        assert_eq!(
            opus_packet_get_samples_per_frame(&out[..len], 48_000).unwrap(),
            960
        );
    }

    #[test]
    fn restricted_low_delay_forces_celt() {
        let mut enc = opus_encoder_create(48_000, 1, 2051).expect("encoder");
        let pcm = [0i16; 960];
        let mut out = [0u8; 4000];

        let len = opus_encode(&mut enc, &pcm, 960, &mut out).expect("encode");
        assert_eq!(
            opus_packet_get_mode(&out[..len]).unwrap(),
            crate::packet::Mode::CELT
        );
    }

    #[test]
    fn bandwidth_clamps_to_sample_rate_limits() {
        let mut enc = opus_encoder_create(16_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY),
        )
        .unwrap();
        let pcm = [0i16; 320];
        let mut out = [0u8; 4000];

        let _ = opus_encode(&mut enc, &pcm, 320, &mut out).expect("encode");
        assert!(enc.bandwidth.to_opus_int() <= OPUS_BANDWIDTH_WIDEBAND);
    }

    #[test]
    fn ctl_round_trips_basic_settings() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetVbr(false)).unwrap();
        let mut vbr = true;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetVbr(&mut vbr)).unwrap();
        assert!(!vbr);
    }

    #[test]
    fn ctl_round_trips_extended_settings() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(1)).unwrap();
        let mut force_channels = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetForceChannels(&mut force_channels),
        )
        .unwrap();
        assert_eq!(force_channels, 1);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetMaxBandwidth(OPUS_BANDWIDTH_SUPERWIDEBAND),
        )
        .unwrap();
        let mut max_bandwidth = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetMaxBandwidth(&mut max_bandwidth),
        )
        .unwrap();
        assert_eq!(max_bandwidth, OPUS_BANDWIDTH_SUPERWIDEBAND);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetBandwidth(OPUS_BANDWIDTH_NARROWBAND),
        )
        .unwrap();

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetSignal(OPUS_SIGNAL_MUSIC),
        )
        .unwrap();
        let mut signal = 0;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetSignal(&mut signal)).unwrap();
        assert_eq!(signal, OPUS_SIGNAL_MUSIC);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLsbDepth(16)).unwrap();
        let mut lsb_depth = 0;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetLsbDepth(&mut lsb_depth)).unwrap();
        assert_eq!(lsb_depth, 16);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetExpertFrameDuration(OPUS_FRAMESIZE_20_MS),
        )
        .unwrap();
        let mut frame_duration = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetExpertFrameDuration(&mut frame_duration),
        )
        .unwrap();
        assert_eq!(frame_duration, OPUS_FRAMESIZE_20_MS);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetPredictionDisabled(true)).unwrap();
        let mut prediction_disabled = false;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetPredictionDisabled(&mut prediction_disabled),
        )
        .unwrap();
        assert!(prediction_disabled);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetPhaseInversionDisabled(true),
        )
        .unwrap();
        let mut phase_inversion_disabled = false;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetPhaseInversionDisabled(&mut phase_inversion_disabled),
        )
        .unwrap();
        assert!(phase_inversion_disabled);

        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::SetForceMode(MODE_SILK_ONLY),
        )
        .unwrap();
    }

    #[cfg(feature = "dred")]
    #[test]
    fn ctl_round_trips_dred_duration() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDredDuration(12)).unwrap();
        let mut duration = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetDredDuration(&mut duration),
        )
        .unwrap();
        assert_eq!(duration, 12);
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::ResetState).unwrap();
        duration = 0;
        opus_encoder_ctl(
            &mut enc,
            OpusEncoderCtlRequest::GetDredDuration(&mut duration),
        )
        .unwrap();
        assert_eq!(duration, 12);
        assert_eq!(enc.silk_mode.use_dred, 1);
    }

    #[cfg(feature = "dred")]
    #[test]
    fn ctl_rejects_invalid_dred_duration() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDredDuration(-1)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetDredDuration(DRED_MAX_FRAMES + 1),
            )
            .unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
    }

    #[cfg(feature = "dred")]
    #[test]
    fn ctl_rejects_invalid_dnn_blob() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        assert!(enc.dred_loaded);
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetDnnBlob(&[1, 2, 3])).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert!(enc.dred_loaded);
    }

    #[cfg(feature = "dred")]
    #[test]
    fn dred_activity_history_updates_when_loaded() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        enc.dred_duration = 10;
        enc.dred_loaded = true;
        for (idx, value) in enc.dred_activity_mem.iter_mut().enumerate() {
            *value = idx as u8;
        }

        let frame_size = 480usize;
        update_dred_activity_history(&mut enc, 1, frame_size);

        let frame_size_400hz = frame_size * 400 / enc.fs as usize;
        assert!(
            enc.dred_activity_mem[..frame_size_400hz]
                .iter()
                .all(|&value| value == 1)
        );
        assert_eq!(enc.dred_activity_mem[frame_size_400hz], 0);
        assert_eq!(enc.dred_activity_mem[frame_size_400hz + 1], 1);
    }

    #[cfg(feature = "dred")]
    #[test]
    fn dred_activity_history_clears_when_unloaded() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        enc.dred_duration = 10;
        enc.dred_loaded = false;
        enc.dred_latents_buffer_fill = 5;
        enc.dred_activity_mem.fill(7);

        update_dred_activity_history(&mut enc, 1, 480);

        assert_eq!(enc.dred_latents_buffer_fill, 0);
        assert!(enc.dred_activity_mem.iter().all(|&value| value == 0));
    }

    #[cfg(feature = "dred")]
    #[test]
    fn dred_celt_bytes_adjustment_caps_budget() {
        let adjusted = adjust_nb_compr_bytes_for_dred(100, 80, 50, 8_000);
        assert_eq!(adjusted, 85);

        let adjusted = adjust_nb_compr_bytes_for_dred(40, 320, 50, 8_000);
        assert_eq!(adjusted, 40);
    }

    #[cfg(feature = "dred")]
    #[test]
    fn dred_bitrate_zero_when_duration_unset() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        enc.dred_duration = 0;
        enc.inband_fec = true;
        enc.packet_loss_perc = 20;

        let dred_bitrate = compute_dred_bitrate(&mut enc, 64_000, 960);
        assert_eq!(dred_bitrate, 0);
        assert_eq!(enc.dred_target_chunks, 0);
    }

    #[test]
    fn ctl_rejects_invalid_extended_settings() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");

        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetForceChannels(3)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetMaxBandwidth(999)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetSignal(42)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLsbDepth(7)).unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetExpertFrameDuration(4242)
            )
            .unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
        assert_eq!(
            opus_encoder_ctl(
                &mut enc,
                OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY + 1)
            )
            .unwrap_err(),
            OpusEncoderCtlError::BadArgument
        );
    }

    #[test]
    fn encode_rejects_unsupported_frames() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        let pcm = [0i16; 479];
        let mut out = [0u8; 4000];
        let err = opus_encode(&mut enc, &pcm, 479, &mut out).unwrap_err();
        assert_eq!(err, OpusEncodeError::BadArgument);
    }

    #[test]
    fn low_bitrate_produces_reference_plc_packet() {
        let mut encoder = opus_encoder_create(48_000, 1, 2049).unwrap();
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetBitrate(500)).unwrap();
        opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbr(false)).unwrap();
        let mut packet = [0u8; 1];
        assert_eq!(
            opus_encode(&mut encoder, &[0i16; 960], 960, &mut packet),
            Ok(1)
        );
        assert_eq!(packet, [0x78]);
        assert_eq!(encoder.range_final, 0);
        assert_eq!(
            opus_encode(&mut encoder, &[0i16; 4800], 4800, &mut packet),
            Err(OpusEncodeError::BufferTooSmall)
        );
    }

    #[test]
    fn opus_encode24_accepts_int24_input() {
        let mut enc = opus_encoder_create(48_000, 1, 2051).expect("encoder");
        let pcm = vec![0i32; 960];
        let mut out = vec![0u8; 1500];
        let result = opus_encode24(&mut enc, &pcm, 960, &mut out);
        assert!(result.is_ok(), "opus_encode24 should accept int24 input");
    }

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn native_precision_matches_scalar_c_packet() {
        // Generated with the pinned scalar C reference, CBR at 96000 bit/s.
        // Nonzero low eight bits make an intermediate i16 conversion visible.
        let pcm24: Vec<i32> = (0..960)
            .map(|i| (i * 53_137) % 16_000_001 - 8_000_000)
            .collect();
        let pcm_float: Vec<f32> = pcm24
            .iter()
            .map(|&sample| sample as f32 / 8_388_608.0)
            .collect();
        let expected = include_bytes!("../../tests/fixtures/native24-celt-48000-mono-20ms.packet");
        for float_api in [false, true] {
            let mut encoder = opus_encoder_create(48_000, 1, 2049).unwrap();
            for request in [
                OpusEncoderCtlRequest::SetBitrate(96_000),
                OpusEncoderCtlRequest::SetVbr(false),
                OpusEncoderCtlRequest::SetComplexity(10),
                OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY),
            ] {
                opus_encoder_ctl(&mut encoder, request).unwrap();
            }
            let mut packet = [0u8; 1276];
            let size = if float_api {
                super::opus_encode_float(&mut encoder, &pcm_float, 960, &mut packet)
            } else {
                opus_encode24(&mut encoder, &pcm24, 960, &mut packet)
            }
            .unwrap();
            assert_eq!(&packet[..size], expected);
            let mut final_range = 0;
            opus_encoder_ctl(
                &mut encoder,
                OpusEncoderCtlRequest::GetFinalRange(&mut final_range),
            )
            .unwrap();
            assert_eq!(final_range, 743_246_336);
        }
    }

    #[test]
    fn user_bitrate_to_bitrate_respects_auto_and_max() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let frame_size = 960;
        let max_data_bytes = 200;

        enc.user_bitrate_bps = OPUS_AUTO;
        let expected_auto = 60 * 48_000 / frame_size + 48_000 * 2;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            expected_auto
        );

        enc.user_bitrate_bps = OPUS_BITRATE_MAX;
        let expected_max = max_data_bytes * 8 * 48_000 / frame_size;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            expected_max
        );

        enc.user_bitrate_bps = 12_345;
        assert_eq!(
            user_bitrate_to_bitrate(&enc, frame_size, max_data_bytes),
            12_345
        );
    }

    #[test]
    fn compute_equiv_rate_matches_reference_math() {
        let equiv = compute_equiv_rate(10_000, 1, 100, true, MODE_CELT_ONLY, 10, 0);
        assert_eq!(equiv, 7_000);
    }

    #[test]
    fn decide_fec_restores_bandwidth_when_unavailable() {
        let mut bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        let enabled = decide_fec(true, 20, false, MODE_SILK_ONLY, &mut bandwidth, 1_000);
        assert!(!enabled);
        assert_eq!(bandwidth, OPUS_BANDWIDTH_WIDEBAND);
    }

    #[test]
    fn compute_redundancy_bytes_respects_caps() {
        let redundancy = compute_redundancy_bytes(100, 20_000, 50, 1);
        assert_eq!(redundancy, 24);
    }

    #[test]
    fn compute_silk_rate_for_hybrid_from_table() {
        let silk_rate = compute_silk_rate_for_hybrid(
            24_000,
            OPUS_BANDWIDTH_SUPERWIDEBAND,
            true,
            true,
            false,
            1,
        );
        assert_eq!(silk_rate, 18_300);
    }

    #[test]
    fn compute_stereo_width_stays_zero_for_silence() {
        let pcm = vec![0i16; 2 * 96];
        let mut mem = StereoWidthState::default();
        let width = compute_stereo_width(&pcm, 96, 48_000, &mut mem);
        assert_eq!(width, 0.0);
        assert_eq!(mem.xx, 0.0);
        assert_eq!(mem.xy, 0.0);
        assert_eq!(mem.yy, 0.0);
        assert_eq!(mem.smoothed_width, 0.0);
        assert_eq!(mem.max_follower, 0.0);
    }

    #[test]
    fn gain_fade_unity_gain_preserves_signal() {
        // When both gains are 1.0, the signal should be unchanged
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut pcm = original.clone();
        let window = vec![0.5f32; 120]; // Typical overlap window

        super::gain_fade_float(&mut pcm, 1.0, 1.0, 120, 8, 1, &window, 48_000);

        for (i, &val) in pcm.iter().enumerate() {
            assert!(
                (val - original[i]).abs() < 1e-6,
                "Sample {} changed from {} to {} with unity gain",
                i,
                original[i],
                val
            );
        }
    }

    #[test]
    fn gain_fade_applies_constant_gain_after_overlap() {
        // After the overlap region, constant g2 should be applied
        let mut pcm = vec![1.0; 960]; // 20ms at 48kHz mono
        let window = vec![1.0f32; 120]; // Full overlap window (120 samples at 48kHz)
        let g2 = 0.5;

        super::gain_fade_float(&mut pcm, 1.0, g2, 120, 960, 1, &window, 48_000);

        // Samples after the overlap region should have g2 applied
        for &val in &pcm[120..] {
            assert!(
                (val - g2).abs() < 1e-6,
                "Post-overlap sample should be {} but got {}",
                g2,
                val
            );
        }
    }

    #[test]
    fn gain_fade_stereo_applies_to_both_channels() {
        // Both channels should get the same gain applied
        let mut pcm = vec![1.0; 960 * 2]; // 20ms at 48kHz stereo
        let window = vec![1.0f32; 120];
        let g2 = 0.5;

        super::gain_fade_float(&mut pcm, 1.0, g2, 120, 960, 2, &window, 48_000);

        // After overlap, both channels should have g2
        for i in 120..960 {
            let left = pcm[i * 2];
            let right = pcm[i * 2 + 1];
            assert!(
                (left - g2).abs() < 1e-6 && (right - g2).abs() < 1e-6,
                "Sample {}: left={}, right={}, expected {}",
                i,
                left,
                right,
                g2
            );
        }
    }

    #[test]
    fn gain_fade_interpolates_in_overlap_region() {
        // In the overlap region, gain should smoothly transition from g1 to g2
        let mut pcm = vec![1.0; 240]; // Larger than overlap
        // Create a window that ramps from 0 to 1 (simplified)
        let window: Vec<f32> = (0..120).map(|i| i as f32 / 119.0).collect();
        let g1 = 1.0;
        let g2 = 0.5;

        super::gain_fade_float(&mut pcm, g1, g2, 120, 240, 1, &window, 48_000);

        // First sample: w=0, w_sq=0, g = 0*g2 + 1*g1 = g1
        assert!(
            (pcm[0] - g1).abs() < 1e-6,
            "First sample should be close to g1={}, got {}",
            g1,
            pcm[0]
        );

        // Last overlap sample: w≈1, w_sq≈1, g ≈ g2
        let last_overlap = 119;
        assert!(
            (pcm[last_overlap] - g2).abs() < 0.02,
            "Last overlap sample should be close to g2={}, got {}",
            g2,
            pcm[last_overlap]
        );
    }

    #[test]
    fn hb_gain_computation_matches_expected_values() {
        // Test the HB_gain formula: 1.0 - celt_exp2(-celt_rate / 1024.0)
        use crate::celt::celt_exp2;

        // At high CELT rates, HB_gain should approach 1.0
        let high_celt_rate = 32_000i32;
        let hb_gain_high = 1.0 - celt_exp2(-high_celt_rate as f32 / 1024.0);
        assert!(
            hb_gain_high > 0.99,
            "High CELT rate should give HB_gain near 1.0, got {}",
            hb_gain_high
        );

        // At low CELT rates, HB_gain should be lower
        let low_celt_rate = 2_000i32;
        let hb_gain_low = 1.0 - celt_exp2(-low_celt_rate as f32 / 1024.0);
        assert!(
            hb_gain_low < hb_gain_high,
            "Low CELT rate ({}) should give lower HB_gain than high rate ({})",
            hb_gain_low,
            hb_gain_high
        );
        assert!(
            hb_gain_low > 0.3,
            "Low CELT rate HB_gain should still be positive, got {}",
            hb_gain_low
        );

        // At zero CELT rate, HB_gain = 1.0 - exp2(0) = 0.0
        let zero_celt_rate = 0i32;
        let hb_gain_zero = 1.0 - celt_exp2(-zero_celt_rate as f32 / 1024.0);
        assert!(
            hb_gain_zero == 0.0,
            "Zero CELT rate should give HB_gain=0.0, got {}",
            hb_gain_zero
        );
    }

    #[test]
    fn stereo_fade_unity_width_preserves_signal() {
        // When both widths are 1.0 (full stereo), the signal should be unchanged
        // because g1 = g2 = 1.0 - 1.0 = 0.0, so no difference attenuation occurs
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut pcm = original.clone();
        let window = vec![0.5f32; 120];

        super::stereo_fade_float(&mut pcm, 1.0, 1.0, 120, 4, 2, &window, 48_000);

        for (i, &val) in pcm.iter().enumerate() {
            assert!(
                (val - original[i]).abs() < 1e-6,
                "Sample {} changed from {} to {} with unity width",
                i,
                original[i],
                val
            );
        }
    }

    #[test]
    fn stereo_fade_zero_width_collapses_to_mono() {
        // When width is 0.0, g = 1.0 - 0.0 = 1.0, so full difference attenuation
        // L' = L - diff = L - (L-R)/2 = (L+R)/2
        // R' = R + diff = R + (L-R)/2 = (L+R)/2
        // So L' = R' = mid signal
        let mut pcm = vec![
            4.0, 2.0, // Sample 0: L=4, R=2, mid=(4+2)/2=3
            6.0, 0.0, // Sample 1: L=6, R=0, mid=(6+0)/2=3
            10.0, 2.0, // Sample 2: L=10, R=2, mid=(10+2)/2=6
            8.0, 4.0, // Sample 3: L=8, R=4, mid=(8+4)/2=6
        ];
        let window = vec![1.0f32; 120]; // Full window so overlap uses g2 immediately

        super::stereo_fade_float(&mut pcm, 0.0, 0.0, 120, 4, 2, &window, 48_000);

        // All samples should collapse to mono (L = R = mid)
        for i in 0..4 {
            let left = pcm[i * 2];
            let right = pcm[i * 2 + 1];
            assert!(
                (left - right).abs() < 1e-6,
                "Sample {}: L={} should equal R={} for mono collapse",
                i,
                left,
                right
            );
        }
        // Check specific expected values
        assert!((pcm[0] - 3.0).abs() < 1e-6, "Sample 0 mid should be 3.0");
        assert!((pcm[2] - 3.0).abs() < 1e-6, "Sample 1 mid should be 3.0");
        assert!((pcm[4] - 6.0).abs() < 1e-6, "Sample 2 mid should be 6.0");
        assert!((pcm[6] - 6.0).abs() < 1e-6, "Sample 3 mid should be 6.0");
    }

    #[test]
    fn stereo_fade_applies_constant_after_overlap() {
        // After the overlap region, constant g2 should be applied
        let mut pcm = vec![0.0; 960 * 2]; // 20ms at 48kHz stereo
        // Set up stereo signal: L=1.0, R=-1.0 for all samples
        for i in 0..960 {
            pcm[i * 2] = 1.0;
            pcm[i * 2 + 1] = -1.0;
        }
        let window = vec![1.0f32; 120];
        let g2 = 0.5; // Half stereo width

        super::stereo_fade_float(&mut pcm, 1.0, g2, 120, 960, 2, &window, 48_000);

        // After overlap, with g2=0.5, inverted g2 = 0.5
        // diff = (1.0 - (-1.0)) / 2 = 1.0
        // L' = 1.0 - 0.5 * 1.0 = 0.5
        // R' = -1.0 + 0.5 * 1.0 = -0.5
        for i in 120..960 {
            let left = pcm[i * 2];
            let right = pcm[i * 2 + 1];
            assert!(
                (left - 0.5).abs() < 1e-6,
                "Sample {} left should be 0.5, got {}",
                i,
                left
            );
            assert!(
                (right - (-0.5)).abs() < 1e-6,
                "Sample {} right should be -0.5, got {}",
                i,
                right
            );
        }
    }

    #[test]
    fn stereo_fade_interpolates_in_overlap_region() {
        // In the overlap region, width should smoothly transition from g1 to g2
        let mut pcm = vec![0.0; 240 * 2];
        // Set up stereo signal: L=1.0, R=-1.0
        for i in 0..240 {
            pcm[i * 2] = 1.0;
            pcm[i * 2 + 1] = -1.0;
        }
        // Window ramps from 0 to 1
        let window: Vec<f32> = (0..120).map(|i| i as f32 / 119.0).collect();
        let g1 = 1.0; // Full stereo
        let g2 = 0.0; // Mono

        super::stereo_fade_float(&mut pcm, g1, g2, 120, 240, 2, &window, 48_000);

        // First sample: w=0, w_sq=0, inverted_g = 0*1.0 + 1*0.0 = 0.0 (full stereo)
        // So L and R should be unchanged
        assert!(
            (pcm[0] - 1.0).abs() < 1e-6,
            "First sample L should be 1.0, got {}",
            pcm[0]
        );
        assert!(
            (pcm[1] - (-1.0)).abs() < 1e-6,
            "First sample R should be -1.0, got {}",
            pcm[1]
        );

        // Last overlap sample: w≈1, w_sq≈1, inverted_g ≈ 1.0 (mono)
        // So L and R should be close to 0 (mid of 1.0 and -1.0)
        let last = 119;
        assert!(
            pcm[last * 2].abs() < 0.02,
            "Last overlap L should be near 0, got {}",
            pcm[last * 2]
        );
        assert!(
            pcm[last * 2 + 1].abs() < 0.02,
            "Last overlap R should be near 0, got {}",
            pcm[last * 2 + 1]
        );
    }

    #[test]
    fn surround_masking_rate_offset_zero_masks() {
        // With all zeros, masking_depth = 0.0 + 0.2 = 0.2
        // rate_offset = 16000 * 0.2 = 3200
        let mask = [0.0f32; 42]; // 21 bands * 2 channels
        let offset = super::compute_surround_masking_rate_offset(&mask, OPUS_BANDWIDTH_WIDEBAND, 2);
        assert_eq!(offset, 3200);
    }

    #[test]
    fn surround_masking_rate_offset_clamps_negative_masks() {
        // All masks at -10.0 should be clamped to -2.0
        // C evaluates mask_sum / end * channels, retaining its scalar order.
        // masking_depth = -8.0 + 0.2 = -7.8
        let mask = [-10.0f32; 42];
        let offset = super::compute_surround_masking_rate_offset(&mask, OPUS_BANDWIDTH_WIDEBAND, 2);
        assert_eq!(offset, -124800);
    }

    #[test]
    fn surround_masking_rate_offset_halves_positive_masks() {
        // All masks at 0.4, halved to 0.2
        // masking_depth = 0.8 + 0.2 = 1.0
        // rate_offset = 16000 * 1.0 = 16000
        let mask = [0.4f32; 42];
        let offset = super::compute_surround_masking_rate_offset(&mask, OPUS_BANDWIDTH_WIDEBAND, 2);
        // Allow small floating-point tolerance
        assert!(
            (offset - 16000).abs() <= 1,
            "expected ~16000, got {}",
            offset
        );
    }

    #[test]
    fn surround_masking_rate_offset_narrowband_uses_fewer_bands() {
        // NB uses 13 bands, srate=8000
        // With all zeros: masking_depth = 0.2, rate_offset = 8000 * 0.2 = 1600
        let mask = [0.0f32; 42];
        let offset =
            super::compute_surround_masking_rate_offset(&mask, OPUS_BANDWIDTH_NARROWBAND, 2);
        assert_eq!(offset, 1600);
    }

    #[test]
    fn ctl_lfe_round_trip() {
        let mut enc = opus_encoder_create(48_000, 1, 2048).expect("encoder");
        assert!(!enc.lfe);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLfe(true)).unwrap();
        assert!(enc.lfe);

        let mut lfe_out = false;
        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::GetLfe(&mut lfe_out)).unwrap();
        assert!(lfe_out);

        opus_encoder_ctl(&mut enc, OpusEncoderCtlRequest::SetLfe(false)).unwrap();
        assert!(!enc.lfe);
    }

    #[test]
    fn encode_with_options_accepts_energy_masking_slice() {
        let mut enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let pcm = vec![0i16; 960 * 2];
        let mut out = vec![0u8; 512];
        let mask = [0.1f32; 42];

        let len = opus_encode_with_options(
            &mut enc,
            &pcm,
            960,
            &mut out,
            OpusEncodeOptions {
                energy_masking: Some(&mask),
            },
        )
        .expect("encode");

        assert!(len > 0);
    }

    #[test]
    fn init_sets_energy_masking_defaults() {
        let enc = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        assert!(!enc.lfe);
    }

    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    #[test]
    fn opus_res24_trace_output() {
        let path = match std::env::var("OPUS_TRACE_PCM24") {
            Ok(path) => path,
            Err(_) => return,
        };
        let bytes = std::fs::read(path).unwrap();
        let input: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let mut encoder = opus_encoder_create(48000, 2, 2051).unwrap();
        for request in [
            OpusEncoderCtlRequest::SetBitrate(128000),
            OpusEncoderCtlRequest::SetComplexity(10),
            OpusEncoderCtlRequest::SetForceMode(MODE_CELT_ONLY),
            OpusEncoderCtlRequest::SetVbr(false),
        ] {
            opus_encoder_ctl(&mut encoder, request).unwrap();
        }
        for frame in input.chunks_exact(1920) {
            let mut packet = [0u8; 1276];
            super::opus_encode24(&mut encoder, frame, 960, &mut packet).unwrap();
        }
    }

    #[test]
    fn opus_mode_trace_output() {
        #[cfg(test)]
        use crate::opus_encoder::celt_pcm_trace;
        let path = match std::env::var("OPUS_TRACE_PCM") {
            Ok(value) => value,
            Err(_) => return,
        };
        let frames = std::env::var("OPUS_TRACE_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);

        let parameter = |name: &str, fallback: i32| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(fallback)
        };
        let rate = parameter("OPUS_TRACE_RATE", 48_000);
        let channels = parameter("OPUS_TRACE_CHANNELS", 2) as usize;
        let frame_size = parameter("OPUS_TRACE_FRAME_SIZE", rate / 50) as usize;
        let mut file = std::fs::File::open(&path).expect("open OPUS_TRACE_PCM");
        let mut encoder = opus_encoder_create(
            rate,
            channels as i32,
            parameter("OPUS_TRACE_APPLICATION", 2049),
        )
        .expect("encoder init");
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetBitrate(parameter("OPUS_TRACE_BITRATE", 64_000)),
        )
        .expect("set bitrate");
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetVbr(parameter("OPUS_TRACE_VBR", 1) != 0),
        )
        .unwrap();
        #[cfg(feature = "enable_qext")]
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetQext(parameter("OPUS_TRACE_QEXT", 0) != 0),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetDtx(parameter("OPUS_TRACE_DTX", 0) != 0),
        )
        .unwrap();
        #[cfg(feature = "dred")]
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetDredDuration(parameter("OPUS_TRACE_DRED", 0)),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetInbandFec(parameter("OPUS_TRACE_FEC", 0) != 0),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetPacketLossPerc(parameter("OPUS_TRACE_LOSS", 0)),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetComplexity(parameter("OPUS_TRACE_COMPLEXITY", 10)),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetForceMode(parameter("OPUS_TRACE_FORCE_MODE", OPUS_AUTO)),
        )
        .unwrap();
        opus_encoder_ctl(
            &mut encoder,
            OpusEncoderCtlRequest::SetBandwidth(parameter("OPUS_TRACE_BANDWIDTH", OPUS_AUTO)),
        )
        .unwrap();

        let mut input_bytes = vec![0u8; frame_size * channels * 2];
        let mut input_pcm = vec![0i16; frame_size * channels];
        let mut packet = vec![0u8; 3 * 1276];

        for frame_idx in 0..frames {
            if std::env::var_os("OPUS_TRACE_TRANSITIONS").is_some() {
                let mode = [
                    MODE_SILK_ONLY,
                    MODE_HYBRID,
                    MODE_CELT_ONLY,
                    MODE_HYBRID,
                    MODE_SILK_ONLY,
                ][(frame_idx / 4) % 5];
                opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetForceMode(mode)).unwrap();
                opus_encoder_ctl(
                    &mut encoder,
                    OpusEncoderCtlRequest::SetBitrate(
                        if mode == MODE_SILK_ONLY { 24000 } else { 48000 } * channels as i32,
                    ),
                )
                .unwrap();
                opus_encoder_ctl(
                    &mut encoder,
                    OpusEncoderCtlRequest::SetBandwidth(if mode == MODE_SILK_ONLY {
                        1103
                    } else if rate >= 48000 {
                        1105
                    } else {
                        1104
                    }),
                )
                .unwrap();
                opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetInbandFec(true)).unwrap();
                opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetPacketLossPerc(15))
                    .unwrap();
            }
            match std::io::Read::read_exact(&mut file, &mut input_bytes) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(err) => panic!("read OPUS_TRACE_PCM failed: {err}"),
            }

            #[cfg(test)]
            if let Some(cfg) = celt_pcm_trace::config_copy()
                && cfg.frame.map_or(true, |frame| frame == frame_idx)
            {
                for (sample, chunk) in input_pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
                    *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                }
                celt_pcm_trace::dump(
                    "opus_trace_pcm",
                    &input_pcm,
                    frame_size,
                    channels,
                    cfg.start,
                    cfg.count,
                    cfg.want_bits,
                    frame_idx,
                );
            }

            for (sample, chunk) in input_pcm.iter_mut().zip(input_bytes.chunks_exact(2)) {
                *sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            }

            let _ = opus_encode(&mut encoder, &input_pcm, frame_size, &mut packet)
                .expect("opus_encode");
        }
    }
}
