//! Channel layout helpers mirrored from `opus_multistream.c`.
#![cfg_attr(not(test), allow(dead_code))]

use alloc::vec;
use alloc::vec::Vec;

use crate::celt::{
    CELT_SIG_SCALE, OpusRes, amp2_log2, celt_inner_prod, celt_log2, celt_preemphasis,
    clt_mdct_forward, compute_band_energies, float2int, float2int16, isqrt32,
    opus_custom_mode_find_static_ref, resampling_factor,
};
use crate::opus_decoder::{
    OpusDecodeError, OpusDecoder, OpusDecoderCtlError, OpusDecoderCtlRequest, OpusDecoderInitError,
    opus_decode_native, opus_decoder_create, opus_decoder_ctl, opus_decoder_get_size,
};
use crate::opus_encoder::{
    OPUS_FRAMESIZE_ARG, OpusAnalysisInput, OpusEncodeError, OpusEncodeOptions, OpusEncoder,
    OpusEncoderCtlError, OpusEncoderCtlRequest, OpusEncoderInitError, frame_size_select,
    opus_encode_float_with_analysis, opus_encode_float_with_options, opus_encode_with_options,
    opus_encode24_with_options, opus_encoder_create, opus_encoder_ctl, opus_encoder_get_size,
};
use crate::packet::{PacketError, opus_packet_get_nb_samples, opus_packet_parse_impl};
use crate::repacketizer::OpusRepacketizer;

/// Sentinel used by the reference encoder when auto-selecting the bitrate.
pub(crate) const OPUS_AUTO: i32 = -1000;
/// Maximum bitrate marker mirrored from the public Opus defines.
pub(crate) const OPUS_BITRATE_MAX: i32 = -1;

#[cfg(feature = "enable_qext")]
const MS_FRAME_TMP: usize = 6 * 3825 + 12;
#[cfg(not(feature = "enable_qext"))]
const MS_FRAME_TMP: usize = 6 * 1275 + 12;

/// Mirrors the mapping type enum embedded in the multistream encoder state.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MappingType {
    None,
    Surround,
    Ambisonics,
}

/// Internal multistream channel layout description.
///
/// Mirrors the layout prefix embedded in the multistream encoder/decoder
/// states. The `mapping` table uses `255` as a sentinel for channels that are
/// omitted from the encoded streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelLayout {
    pub nb_channels: usize,
    pub nb_streams: usize,
    pub nb_coupled_streams: usize,
    pub mapping: [u8; 256],
}

/// Errors surfaced by the multistream decoder front-end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusMultistreamDecoderError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    InvalidPacket,
    Unimplemented,
    DecoderInit(OpusDecoderInitError),
    DecoderCtl(OpusDecoderCtlError),
    DecoderDecode(OpusDecodeError),
}

impl OpusMultistreamDecoderError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
            Self::DecoderInit(OpusDecoderInitError::BadArgument) => -1,
            Self::DecoderInit(_) => -3,
            Self::DecoderCtl(OpusDecoderCtlError::BadArgument) => -1,
            Self::DecoderCtl(OpusDecoderCtlError::Unimplemented) => -5,
            Self::DecoderCtl(OpusDecoderCtlError::Silk(_)) => -3,
            Self::DecoderDecode(err) => err.code(),
        }
    }
}

impl From<PacketError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: PacketError) -> Self {
        match value {
            PacketError::BadArgument => Self::BadArgument,
            PacketError::InvalidPacket => Self::InvalidPacket,
        }
    }
}

impl From<OpusDecoderInitError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecoderInitError) -> Self {
        Self::DecoderInit(value)
    }
}

impl From<OpusDecoderCtlError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecoderCtlError) -> Self {
        Self::DecoderCtl(value)
    }
}

impl From<OpusDecodeError> for OpusMultistreamDecoderError {
    #[inline]
    fn from(value: OpusDecodeError) -> Self {
        Self::DecoderDecode(value)
    }
}

#[repr(C)]
struct ChannelLayoutLayout {
    nb_channels: i32,
    nb_streams: i32,
    nb_coupled_streams: i32,
    mapping: [u8; 256],
}

#[repr(C)]
struct OpusMsDecoderLayout {
    layout: ChannelLayoutLayout,
}

#[repr(C)]
struct OpusMsEncoderLayout {
    layout: ChannelLayoutLayout,
}

/// Mirrors the alignment helper from `opus_private.h`.
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

/// Returns the number of bytes required to allocate a multistream decoder.
#[must_use]
pub fn opus_multistream_decoder_get_size(
    nb_streams: usize,
    nb_coupled_streams: usize,
) -> Option<usize> {
    if nb_streams == 0 || nb_coupled_streams > nb_streams {
        return None;
    }

    let coupled_size = opus_decoder_get_size(2)?;
    let mono_size = opus_decoder_get_size(1)?;
    let header_size = align(core::mem::size_of::<OpusMsDecoderLayout>());

    let coupled_total = nb_coupled_streams.checked_mul(align(coupled_size))?;
    let mono_total = nb_streams
        .checked_sub(nb_coupled_streams)?
        .checked_mul(align(mono_size))?;

    header_size
        .checked_add(coupled_total)?
        .checked_add(mono_total)
}

/// Multistream decoder state mirroring `OpusMSDecoder` from the reference code.
#[derive(Debug)]
pub struct OpusMultistreamDecoder<'mode> {
    layout: ChannelLayout,
    decoders: Vec<OpusDecoder<'mode>>,
}

impl<'mode> OpusMultistreamDecoder<'mode> {
    #[inline]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    #[inline]
    fn sample_rate(&self) -> Option<i32> {
        self.decoders.first().map(|decoder| decoder.fs)
    }

    /// Resets the decoder to a new layout and sample rate.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<(), OpusMultistreamDecoderError> {
        let layout = build_layout(channels, streams, coupled_streams, mapping)?;
        let decoders = build_stream_decoders(sample_rate, streams, coupled_streams)?;

        self.layout = layout;
        self.decoders = decoders;
        Ok(())
    }

    /// Returns a mutable reference to the decoder for `stream_id`, mirroring
    /// `OPUS_MULTISTREAM_GET_DECODER_STATE`.
    pub fn decoder_state(&mut self, stream_id: usize) -> Option<&mut OpusDecoder<'mode>> {
        self.decoders.get_mut(stream_id)
    }
}

/// Mirrors `opus_multistream_decoder_create` by allocating and initialising all
/// component decoders.
pub fn opus_multistream_decoder_create(
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<OpusMultistreamDecoder<'static>, OpusMultistreamDecoderError> {
    let layout = build_layout(channels, streams, coupled_streams, mapping)?;
    let decoders = build_stream_decoders(sample_rate, streams, coupled_streams)?;

    Ok(OpusMultistreamDecoder { layout, decoders })
}

/// Mirrors `opus_multistream_decoder_init` by resetting an existing decoder instance.
pub fn opus_multistream_decoder_init(
    decoder: &mut OpusMultistreamDecoder<'_>,
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<(), OpusMultistreamDecoderError> {
    decoder.init(sample_rate, channels, streams, coupled_streams, mapping)
}

fn build_layout(
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<ChannelLayout, OpusMultistreamDecoderError> {
    if channels == 0
        || channels > 255
        || coupled_streams > streams
        || streams == 0
        || streams > 255 - coupled_streams
    {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    if mapping.len() < channels {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut layout = ChannelLayout {
        nb_channels: channels,
        nb_streams: streams,
        nb_coupled_streams: coupled_streams,
        mapping: [u8::MAX; 256],
    };
    layout.mapping[..channels].copy_from_slice(&mapping[..channels]);
    if !validate_layout(&layout) {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    Ok(layout)
}

fn build_stream_decoders(
    sample_rate: i32,
    streams: usize,
    coupled_streams: usize,
) -> Result<Vec<OpusDecoder<'static>>, OpusMultistreamDecoderError> {
    if sample_rate <= 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut decoders = Vec::with_capacity(streams);
    for stream in 0..streams {
        let channels: i32 = if stream < coupled_streams { 2 } else { 1 };
        decoders.push(opus_decoder_create(sample_rate, channels)?);
    }

    Ok(decoders)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VorbisLayout {
    nb_streams: usize,
    nb_coupled_streams: usize,
    mapping: [u8; 8],
}

/* Index is nb_channel-1 */
#[allow(dead_code)]
const VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 0,
        mapping: [0, 255, 255, 255, 255, 255, 255, 255],
    }, /* 1: mono */
    VorbisLayout {
        nb_streams: 1,
        nb_coupled_streams: 1,
        mapping: [0, 1, 255, 255, 255, 255, 255, 255],
    }, /* 2: stereo */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 1,
        mapping: [0, 2, 1, 255, 255, 255, 255, 255],
    }, /* 3: 1-d surround */
    VorbisLayout {
        nb_streams: 2,
        nb_coupled_streams: 2,
        mapping: [0, 1, 2, 3, 255, 255, 255, 255],
    }, /* 4: quadraphonic surround */
    VorbisLayout {
        nb_streams: 3,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 255, 255, 255],
    }, /* 5: 5-channel surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 255, 255],
    }, /* 6: 5.1 surround */
    VorbisLayout {
        nb_streams: 4,
        nb_coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 255],
    }, /* 7: 6.1 surround */
    VorbisLayout {
        nb_streams: 5,
        nb_coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    }, /* 8: 7.1 surround */
];

/// Verifies that the layout only references stream indices that exist.
#[must_use]
pub(crate) fn validate_layout(layout: &ChannelLayout) -> bool {
    let Some(max_channel) = layout.nb_streams.checked_add(layout.nb_coupled_streams) else {
        return false;
    };

    if max_channel > u8::MAX as usize {
        return false;
    }

    if layout.nb_channels > layout.mapping.len() {
        return false;
    }

    layout
        .mapping
        .iter()
        .take(layout.nb_channels)
        .all(|&value| value == u8::MAX || usize::from(value) < max_channel)
}

/// Ensures each stream in the layout has a channel mapping.
#[must_use]
pub(crate) fn validate_encoder_layout(layout: &ChannelLayout) -> bool {
    for stream in 0..layout.nb_streams {
        if stream < layout.nb_coupled_streams {
            if get_left_channel(layout, stream, None).is_none() {
                return false;
            }
            if get_right_channel(layout, stream, None).is_none() {
                return false;
            }
        } else if get_mono_channel(layout, stream, None).is_none() {
            return false;
        }
    }

    true
}

/// Validates the ambisonics channel count and returns the derived stream layout.
#[must_use]
pub(crate) fn validate_ambisonics(channels: usize) -> Option<(usize, usize)> {
    if !(1..=227).contains(&channels) {
        return None;
    }

    let order_plus_one = isqrt32(channels as u32) as usize;
    let acn_channels = order_plus_one.checked_mul(order_plus_one)?;
    let nondiegetic_channels = channels.checked_sub(acn_channels)?;

    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    let streams = acn_channels + usize::from(nondiegetic_channels != 0);
    let coupled_streams = usize::from(nondiegetic_channels != 0);
    Some((streams, coupled_streams))
}

fn surround_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    let nb_streams = layout.nb_streams;
    let nb_coupled = layout.nb_coupled_streams;
    let nb_lfe = usize::from(lfe_stream.is_some());
    if nb_streams == 0 || nb_coupled > nb_streams || nb_streams < nb_coupled + nb_lfe {
        return None;
    }
    if frame_size == 0 || sample_rate <= 0 {
        return None;
    }
    if rates.len() < nb_streams {
        return None;
    }

    let nb_uncoupled = nb_streams - nb_coupled - nb_lfe;
    let nb_normal = 2 * nb_coupled + nb_uncoupled;
    if nb_normal == 0 {
        return None;
    }

    let frame_rate = sample_rate / frame_size as i32;
    let channel_offset = 40 * frame_rate.max(50);
    let bitrate = if bitrate_bps == OPUS_AUTO {
        nb_normal as i32 * (channel_offset + sample_rate + 10_000) + 8000 * nb_lfe as i32
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_normal as i32 * 750_000 + nb_lfe as i32 * 128_000
    } else {
        bitrate_bps
    };

    let lfe_offset = bitrate
        .checked_div(20)
        .map(|value| value.min(3000))
        .and_then(|value| value.checked_add(15 * frame_rate.max(50)))?;
    let stream_offset =
        (((bitrate - channel_offset * nb_normal as i32 - lfe_offset * nb_lfe as i32)
            / nb_normal as i32)
            / 2)
        .clamp(0, 20_000);
    let coupled_ratio = 512;
    let lfe_ratio = 32;

    let total = ((nb_uncoupled as i32) << 8)
        + coupled_ratio * nb_coupled as i32
        + lfe_ratio * nb_lfe as i32;
    if total == 0 {
        return None;
    }
    let channel_rate = (256i64
        * i64::from(
            bitrate
                - lfe_offset * nb_lfe as i32
                - stream_offset * (nb_coupled as i32 + nb_uncoupled as i32)
                - channel_offset * nb_normal as i32,
        )
        / i64::from(total)) as i32;

    for (stream, slot) in rates.iter_mut().take(nb_streams).enumerate() {
        let value = if stream < nb_coupled {
            2 * channel_offset + (stream_offset + ((channel_rate * coupled_ratio) >> 8)).max(0)
        } else if lfe_stream == Some(stream) {
            (lfe_offset + ((channel_rate * lfe_ratio) >> 8)).max(0)
        } else {
            channel_offset + (stream_offset + channel_rate).max(0)
        };
        *slot = value;
    }

    Some(())
}

fn ambisonics_rate_allocation(
    layout: &ChannelLayout,
    bitrate_bps: i32,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<()> {
    if frame_size == 0 || sample_rate <= 0 || layout.nb_streams == 0 {
        return None;
    }
    if rates.len() < layout.nb_streams {
        return None;
    }

    let nb_channels = layout.nb_streams + layout.nb_coupled_streams;
    let total_rate = if bitrate_bps == OPUS_AUTO {
        let term = sample_rate + 60 * sample_rate / frame_size as i32;
        (layout.nb_coupled_streams + layout.nb_streams) as i32 * term
            + layout.nb_streams as i32 * 15_000
    } else if bitrate_bps == OPUS_BITRATE_MAX {
        nb_channels as i32 * 750_000
    } else {
        bitrate_bps
    };

    let per_stream_rate = total_rate / layout.nb_streams as i32;
    for slot in rates.iter_mut().take(layout.nb_streams) {
        *slot = per_stream_rate;
    }

    Some(())
}

/// Computes the bitrate distribution across all streams.
#[must_use]
pub(crate) fn rate_allocation(
    layout: &ChannelLayout,
    mapping_type: MappingType,
    bitrate_bps: i32,
    lfe_stream: Option<usize>,
    frame_size: usize,
    sample_rate: i32,
    rates: &mut [i32],
) -> Option<i32> {
    match mapping_type {
        MappingType::Ambisonics => {
            ambisonics_rate_allocation(layout, bitrate_bps, frame_size, sample_rate, rates)?
        }
        _ => surround_rate_allocation(
            layout,
            bitrate_bps,
            lfe_stream,
            frame_size,
            sample_rate,
            rates,
        )?,
    }

    let mut sum = 0i64;
    for rate in rates.iter_mut().take(layout.nb_streams) {
        *rate = (*rate).max(500);
        sum += i64::from(*rate);
    }

    i32::try_from(sum).ok()
}

fn next_index(prev: Option<usize>) -> Option<usize> {
    match prev {
        Some(idx) => idx.checked_add(1),
        None => Some(0),
    }
}

fn find_channel(layout: &ChannelLayout, start: usize, target: usize) -> Option<usize> {
    let limit = layout.nb_channels.min(layout.mapping.len());
    (start..limit).find(|&i| usize::from(layout.mapping[i]) == target)
}

/// Returns the next channel mapped to the left slot of `stream_id`.
#[must_use]
pub(crate) fn get_left_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the right slot of `stream_id`.
#[must_use]
pub(crate) fn get_right_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_mul(2)?.checked_add(1)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Returns the next channel mapped to the mono stream `stream_id`.
#[must_use]
pub(crate) fn get_mono_channel(
    layout: &ChannelLayout,
    stream_id: usize,
    prev: Option<usize>,
) -> Option<usize> {
    let target = stream_id.checked_add(layout.nb_coupled_streams)?;
    let start = next_index(prev)?;
    find_channel(layout, start, target)
}

/// Strongly-typed replacement for the multistream decoder CTL dispatcher.
pub enum OpusMultistreamDecoderCtlRequest<'req> {
    SetGain(i32),
    GetGain(&'req mut i32),
    SetComplexity(i32),
    GetComplexity(&'req mut i32),
    GetBandwidth(&'req mut i32),
    GetSampleRate(&'req mut i32),
    GetFinalRange(&'req mut u32),
    ResetState,
    GetLastPacketDuration(&'req mut i32),
    SetPhaseInversionDisabled(bool),
    GetPhaseInversionDisabled(&'req mut bool),
    SetIgnoreExtensions(bool),
    GetIgnoreExtensions(&'req mut bool),
    SetDnnBlob(&'req [u8]),
    #[cfg(feature = "osce")]
    SetOsceBwe(bool),
    #[cfg(feature = "osce")]
    GetOsceBwe(&'req mut bool),
}

/// Applies a control request across the embedded decoders.
pub fn opus_multistream_decoder_ctl<'req>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    request: OpusMultistreamDecoderCtlRequest<'req>,
) -> Result<(), OpusMultistreamDecoderError> {
    if decoder.decoders.is_empty() {
        return Err(OpusMultistreamDecoderError::InternalError);
    }

    match request {
        OpusMultistreamDecoderCtlRequest::SetGain(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetGain(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetGain(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetGain(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetComplexity(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetComplexity(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetComplexity(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetComplexity(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetBandwidth(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetBandwidth(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetSampleRate(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetSampleRate(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::GetFinalRange(slot) => {
            let mut acc = 0u32;
            for dec in &mut decoder.decoders {
                let mut value = 0u32;
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::GetFinalRange(&mut value))?;
                acc ^= value;
            }
            *slot = acc;
        }
        OpusMultistreamDecoderCtlRequest::ResetState => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::ResetState)?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetLastPacketDuration(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetLastPacketDuration(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetPhaseInversionDisabled(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetPhaseInversionDisabled(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetPhaseInversionDisabled(slot) => {
            opus_decoder_ctl(
                decoder
                    .decoders
                    .first_mut()
                    .ok_or(OpusMultistreamDecoderError::InternalError)?,
                OpusDecoderCtlRequest::GetPhaseInversionDisabled(slot),
            )?;
        }
        #[cfg(feature = "osce")]
        OpusMultistreamDecoderCtlRequest::SetOsceBwe(enabled) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetOsceBwe(enabled))?;
            }
        }
        #[cfg(feature = "osce")]
        OpusMultistreamDecoderCtlRequest::GetOsceBwe(enabled) => {
            opus_decoder_ctl(
                &mut decoder.decoders[0],
                OpusDecoderCtlRequest::GetOsceBwe(enabled),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetIgnoreExtensions(value) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetIgnoreExtensions(value))?;
            }
        }
        OpusMultistreamDecoderCtlRequest::GetIgnoreExtensions(slot) => {
            opus_decoder_ctl(
                &mut decoder.decoders[0],
                OpusDecoderCtlRequest::GetIgnoreExtensions(slot),
            )?;
        }
        OpusMultistreamDecoderCtlRequest::SetDnnBlob(data) => {
            for dec in &mut decoder.decoders {
                opus_decoder_ctl(dec, OpusDecoderCtlRequest::SetDnnBlob(data))?;
            }
        }
    }

    Ok(())
}

fn opus_multistream_packet_validate(
    data: &[u8],
    len: usize,
    nb_streams: usize,
    sample_rate: i32,
) -> Result<usize, OpusMultistreamDecoderError> {
    if len > data.len() || nb_streams == 0 || sample_rate <= 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let mut remaining = len;
    let mut cursor = data;
    let mut samples: Option<usize> = None;

    for stream in 0..nb_streams {
        if remaining == 0 {
            return Err(OpusMultistreamDecoderError::InvalidPacket);
        }

        let parsed = opus_packet_parse_impl(cursor, remaining, stream + 1 != nb_streams)?;
        let tmp_samples =
            opus_packet_get_nb_samples(cursor, parsed.packet_offset, sample_rate as u32)?;
        if let Some(prev) = samples {
            if prev != tmp_samples {
                return Err(OpusMultistreamDecoderError::InvalidPacket);
            }
        } else {
            samples = Some(tmp_samples);
        }

        cursor = &cursor[parsed.packet_offset..];
        remaining = remaining
            .checked_sub(parsed.packet_offset)
            .ok_or(OpusMultistreamDecoderError::InvalidPacket)?;
    }

    samples.ok_or(OpusMultistreamDecoderError::InvalidPacket)
}

#[cfg(feature = "fixed_point")]
const OPTIONAL_CLIP: bool = false;
#[cfg(not(feature = "fixed_point"))]
const OPTIONAL_CLIP: bool = true;

fn opus_multistream_decode_native<T: PcmSample>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [T],
    frame_size: usize,
    decode_fec: bool,
    soft_clip: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native_with_handler(
        decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        soft_clip,
        |dst, dst_stride, dst_channel, src, frame_size, src_offset| {
            copy_channel_out(dst, dst_stride, dst_channel, src, frame_size, src_offset);
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn opus_multistream_decode_native_with_handler<T>(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [T],
    mut frame_size: usize,
    decode_fec: bool,
    soft_clip: bool,
    mut handler: impl FnMut(
        &mut [T],
        usize,
        usize,
        Option<(&[OpusRes], usize, Option<&[i32]>)>,
        usize,
        usize,
    ),
) -> Result<usize, OpusMultistreamDecoderError> {
    if frame_size == 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    if len > data.len() {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let nb_streams = decoder.layout.nb_streams;
    if nb_streams == 0 {
        return Err(OpusMultistreamDecoderError::BadArgument);
    }

    let sample_rate = decoder
        .sample_rate()
        .ok_or(OpusMultistreamDecoderError::InternalError)?;

    let max_frame = sample_rate as usize / 25 * 3;
    frame_size = frame_size.min(max_frame);

    let required = frame_size
        .checked_mul(decoder.layout.nb_channels)
        .ok_or(OpusMultistreamDecoderError::BadArgument)?;
    if pcm.len() < required {
        return Err(OpusMultistreamDecoderError::BufferTooSmall);
    }

    let do_plc = len == 0;
    if !do_plc {
        let minimum = nb_streams
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(OpusMultistreamDecoderError::BadArgument)?;
        if len < minimum {
            return Err(OpusMultistreamDecoderError::InvalidPacket);
        }

        let samples = opus_multistream_packet_validate(data, len, nb_streams, sample_rate)?;
        if samples > frame_size {
            return Err(OpusMultistreamDecoderError::BufferTooSmall);
        }
    }

    let mut scratch = vec![OpusRes::default(); 2 * frame_size];
    let mut cursor = data;
    let mut remaining = len;
    let mut decoded_frame_size = frame_size;

    for stream in 0..nb_streams {
        let self_delimited = stream + 1 != nb_streams;

        let decoder_state = decoder
            .decoders
            .get_mut(stream)
            .ok_or(OpusMultistreamDecoderError::InternalError)?;

        let mut packet_offset = 0usize;
        let decoded = if do_plc {
            opus_decode_native(
                decoder_state,
                None,
                0,
                &mut scratch,
                decoded_frame_size,
                decode_fec,
                self_delimited,
                None,
                soft_clip,
            )?
        } else {
            if remaining == 0 {
                return Err(OpusMultistreamDecoderError::InternalError);
            }
            opus_decode_native(
                decoder_state,
                Some(cursor),
                remaining,
                &mut scratch,
                decoded_frame_size,
                decode_fec,
                self_delimited,
                Some(&mut packet_offset),
                soft_clip,
            )?
        };

        if !do_plc {
            if packet_offset > remaining {
                return Err(OpusMultistreamDecoderError::InvalidPacket);
            }
            cursor = &cursor[packet_offset..];
            remaining -= packet_offset;
        }

        if decoded == 0 {
            return Err(OpusMultistreamDecoderError::InternalError);
        }
        decoded_frame_size = decoded;
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        let native_pcm = {
            let samples = decoder_state.native_pcm24();
            let stream_channels = if stream < decoder.layout.nb_coupled_streams {
                2
            } else {
                1
            };
            samples.get(..decoded_frame_size * stream_channels)
        };
        #[cfg(not(all(feature = "fixed_point", feature = "enable_res24")))]
        let native_pcm: Option<&[i32]> = None;

        if stream < decoder.layout.nb_coupled_streams {
            let mut prev = None;
            while let Some(chan) = get_left_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[..], 2, native_pcm)),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }

            let mut prev = None;
            while let Some(chan) = get_right_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[1..], 2, native_pcm.map(|samples| &samples[1..]))),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }
        } else {
            let mut prev = None;
            while let Some(chan) = get_mono_channel(&decoder.layout, stream, prev) {
                handler(
                    pcm,
                    decoder.layout.nb_channels,
                    chan,
                    Some((&scratch[..], 1, native_pcm)),
                    decoded_frame_size,
                    0,
                );
                prev = Some(chan);
            }
        }
    }

    // Handle muted channels.
    for channel in 0..decoder.layout.nb_channels {
        if decoder.layout.mapping[channel] == u8::MAX {
            handler(
                pcm,
                decoder.layout.nb_channels,
                channel,
                None,
                decoded_frame_size,
                0,
            );
        }
    }

    Ok(decoded_frame_size)
}

pub fn opus_multistream_decode(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i16],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(
        decoder,
        data,
        len,
        pcm,
        frame_size,
        decode_fec,
        OPTIONAL_CLIP,
    )
}

pub fn opus_multistream_decode24(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [i32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(decoder, data, len, pcm, frame_size, decode_fec, false)
}

pub fn opus_multistream_decode_float(
    decoder: &mut OpusMultistreamDecoder<'_>,
    data: &[u8],
    len: usize,
    pcm: &mut [f32],
    frame_size: usize,
    decode_fec: bool,
) -> Result<usize, OpusMultistreamDecoderError> {
    opus_multistream_decode_native(decoder, data, len, pcm, frame_size, decode_fec, false)
}

trait PcmSample: Copy + Default {
    fn from_opus_res(value: OpusRes) -> Self;
    fn from_native24(value: i32) -> Self;
}

#[inline]
fn res_to_int24(sample: OpusRes) -> i32 {
    let scale = CELT_SIG_SCALE * 256.0;
    // libopus leaves headroom in its i32 output; it does not saturate to the
    // nominal 24-bit input range here.
    float2int(sample * scale)
}

impl PcmSample for i16 {
    fn from_native24(value: i32) -> Self {
        (value.wrapping_add(128) >> 8).clamp(-32768, 32767) as i16
    }
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
        {
            // RES2INT16 uses an arithmetic rounding shift, including halfway values.
            return (res_to_int24(value).wrapping_add(128) >> 8).clamp(-32768, 32767) as i16;
        }
        #[cfg(not(all(feature = "fixed_point", feature = "enable_res24")))]
        float2int16(value)
    }
}

impl PcmSample for i32 {
    fn from_native24(value: i32) -> Self {
        value
    }
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        res_to_int24(value)
    }
}

impl PcmSample for f32 {
    fn from_native24(value: i32) -> Self {
        value as f32 / 8388608.0
    }
    #[inline]
    fn from_opus_res(value: OpusRes) -> Self {
        value
    }
}

fn copy_channel_out<T: PcmSample>(
    dst: &mut [T],
    dst_stride: usize,
    dst_channel: usize,
    src: Option<(&[OpusRes], usize, Option<&[i32]>)>,
    frame_size: usize,
    src_offset: usize,
) {
    if dst_stride == 0 || frame_size == 0 {
        return;
    }
    for i in 0..frame_size {
        let dst_index = i * dst_stride + dst_channel;
        if dst_index >= dst.len() {
            break;
        }
        dst[dst_index] = match src {
            Some((src_data, src_stride, native)) => {
                let src_index = src_offset + i * src_stride;
                if let Some(value) = native.and_then(|samples| samples.get(src_index)) {
                    T::from_native24(*value)
                } else {
                    let value = src_data.get(src_index).copied().unwrap_or_default();
                    T::from_opus_res(value)
                }
            }
            None => T::default(),
        };
    }
}

/// Errors surfaced by the multistream encoder front-end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpusMultistreamEncoderError {
    BadArgument,
    BufferTooSmall,
    InternalError,
    Unimplemented,
    EncoderInit(OpusEncoderInitError),
    EncoderCtl(OpusEncoderCtlError),
    Encode(OpusEncodeError),
}

impl OpusMultistreamEncoderError {
    #[inline]
    pub const fn code(&self) -> i32 {
        match self {
            Self::BadArgument => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::Unimplemented => -5,
            Self::EncoderInit(err) => err.code(),
            Self::EncoderCtl(err) => err.code(),
            Self::Encode(err) => err.code(),
        }
    }
}

impl From<OpusEncoderInitError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncoderInitError) -> Self {
        Self::EncoderInit(value)
    }
}

impl From<OpusEncoderCtlError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncoderCtlError) -> Self {
        Self::EncoderCtl(value)
    }
}

impl From<OpusEncodeError> for OpusMultistreamEncoderError {
    #[inline]
    fn from(value: OpusEncodeError) -> Self {
        match value {
            OpusEncodeError::BadArgument => Self::BadArgument,
            OpusEncodeError::BufferTooSmall => Self::BufferTooSmall,
            OpusEncodeError::InternalError => Self::InternalError,
            OpusEncodeError::Unimplemented => Self::Unimplemented,
            OpusEncodeError::Silk(_) => Self::Encode(value),
        }
    }
}

/// Multistream encoder state mirroring `OpusMSEncoder` from the reference code.
#[derive(Debug)]
pub struct OpusMultistreamEncoder<'mode> {
    layout: ChannelLayout,
    encoders: Vec<OpusEncoder<'mode>>,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    sample_rate: i32,
    application: i32,
    bitrate_bps: i32,
    variable_duration: i32,
    surround_window: Vec<f32>,
    surround_preemphasis: Vec<f32>,
    #[cfg(feature = "fixed_point")]
    surround_fixed_window: Vec<i32>,
    #[cfg(feature = "fixed_point")]
    surround_fixed_preemphasis: Vec<i32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OpusMultistreamEncodeOptions<'a> {
    pub stream_energy_masks: Option<&'a [Option<&'a [f32]>]>,
}

impl<'mode> OpusMultistreamEncoder<'mode> {
    #[inline]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Resets the encoder to a new layout and sample rate.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: i32,
    ) -> Result<(), OpusMultistreamEncoderError> {
        let layout = build_encoder_layout(channels, streams, coupled_streams, mapping)?;
        let encoders = build_stream_encoders(sample_rate, streams, coupled_streams, application)?;

        self.layout = layout;
        self.encoders = encoders;
        self.mapping_type = MappingType::None;
        self.lfe_stream = None;
        self.sample_rate = sample_rate;
        self.application = application;
        self.bitrate_bps = OPUS_AUTO;
        self.variable_duration = OPUS_FRAMESIZE_ARG;
        self.surround_window.clear();
        self.surround_preemphasis.clear();
        #[cfg(feature = "fixed_point")]
        {
            self.surround_fixed_window.clear();
            self.surround_fixed_preemphasis.clear();
        }
        Ok(())
    }

    /// Returns a mutable reference to the encoder for `stream_id`, mirroring
    /// `OPUS_MULTISTREAM_GET_ENCODER_STATE`.
    pub fn encoder_state(&mut self, stream_id: usize) -> Option<&mut OpusEncoder<'mode>> {
        self.encoders.get_mut(stream_id)
    }
}

/// Returns the encoder state for a specific stream, mirroring
/// `OPUS_MULTISTREAM_GET_ENCODER_STATE`.
pub fn opus_multistream_encoder_get_encoder_state<'a, 'mode>(
    encoder: &'a mut OpusMultistreamEncoder<'mode>,
    stream_id: usize,
) -> Result<&'a mut OpusEncoder<'mode>, OpusMultistreamEncoderError> {
    encoder
        .encoders
        .get_mut(stream_id)
        .ok_or(OpusMultistreamEncoderError::BadArgument)
}

/// Returns the number of bytes required to allocate a multistream encoder.
#[must_use]
pub fn opus_multistream_encoder_get_size(
    nb_streams: usize,
    nb_coupled_streams: usize,
) -> Option<usize> {
    if nb_streams == 0 || nb_coupled_streams > nb_streams {
        return None;
    }

    let coupled_size = opus_encoder_get_size(2)?;
    let mono_size = opus_encoder_get_size(1)?;
    let header_size = align(core::mem::size_of::<OpusMsEncoderLayout>());

    let coupled_total = nb_coupled_streams.checked_mul(align(coupled_size))?;
    let mono_total = nb_streams
        .checked_sub(nb_coupled_streams)?
        .checked_mul(align(mono_size))?;

    header_size
        .checked_add(coupled_total)?
        .checked_add(mono_total)
}

/// Mirrors `opus_multistream_encoder_create` by allocating and initialising all
/// component encoders.
pub fn opus_multistream_encoder_create(
    sample_rate: i32,
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
    application: i32,
) -> Result<OpusMultistreamEncoder<'static>, OpusMultistreamEncoderError> {
    let layout = build_encoder_layout(channels, streams, coupled_streams, mapping)?;
    let encoders = build_stream_encoders(sample_rate, streams, coupled_streams, application)?;

    Ok(OpusMultistreamEncoder {
        layout,
        encoders,
        mapping_type: MappingType::None,
        lfe_stream: None,
        sample_rate,
        application,
        bitrate_bps: OPUS_AUTO,
        variable_duration: OPUS_FRAMESIZE_ARG,
        surround_window: Vec::new(),
        surround_preemphasis: Vec::new(),
        #[cfg(feature = "fixed_point")]
        surround_fixed_window: Vec::new(),
        #[cfg(feature = "fixed_point")]
        surround_fixed_preemphasis: Vec::new(),
    })
}

fn build_encoder_layout(
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Result<ChannelLayout, OpusMultistreamEncoderError> {
    let layout =
        build_layout(channels, streams, coupled_streams, mapping).map_err(|err| match err {
            OpusMultistreamDecoderError::BadArgument => OpusMultistreamEncoderError::BadArgument,
            _ => OpusMultistreamEncoderError::InternalError,
        })?;
    if streams
        .checked_add(coupled_streams)
        .is_none_or(|total| total > channels)
    {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    if !validate_encoder_layout(&layout) {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    Ok(layout)
}

fn build_stream_encoders(
    sample_rate: i32,
    streams: usize,
    coupled_streams: usize,
    application: i32,
) -> Result<Vec<OpusEncoder<'static>>, OpusMultistreamEncoderError> {
    if sample_rate <= 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let mut encoders = Vec::with_capacity(streams);
    for stream in 0..streams {
        let channels: i32 = if stream < coupled_streams { 2 } else { 1 };
        encoders.push(opus_encoder_create(sample_rate, channels, application)?);
    }
    Ok(encoders)
}

/// Strongly-typed replacement for the multistream encoder CTL dispatcher.
pub enum OpusMultistreamEncoderCtlRequest<'req> {
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
    GetVoiceRatio(&'req mut i32),
    SetPacketLossPerc(i32),
    GetPacketLossPerc(&'req mut i32),
    SetInbandFec(bool),
    GetInbandFec(&'req mut bool),
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
    #[cfg(feature = "enable_qext")]
    SetQext(bool),
    #[cfg(feature = "enable_qext")]
    GetQext(&'req mut bool),
    SetDredDuration(i32),
    GetDredDuration(&'req mut i32),
    SetDnnBlob(&'req [u8]),
    SetForceMode(i32),
    GetSampleRate(&'req mut i32),
    GetLookahead(&'req mut i32),
    GetFinalRange(&'req mut u32),
    ResetState,
}

/// Applies a control request across the embedded encoders.
pub fn opus_multistream_encoder_ctl<'req>(
    encoder: &mut OpusMultistreamEncoder<'_>,
    request: OpusMultistreamEncoderCtlRequest<'req>,
) -> Result<(), OpusMultistreamEncoderError> {
    if encoder.encoders.is_empty() {
        return Err(OpusMultistreamEncoderError::InternalError);
    }

    match request {
        OpusMultistreamEncoderCtlRequest::SetApplication(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetApplication(value)).map_err(
                    |err| match err {
                        OpusEncoderCtlError::BadArgument => {
                            OpusMultistreamEncoderError::BadArgument
                        }
                        _ => OpusMultistreamEncoderError::EncoderCtl(err),
                    },
                )?;
            }
            encoder.application = value;
        }
        OpusMultistreamEncoderCtlRequest::GetApplication(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetApplication(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetBitrate(value) => {
            if value != OPUS_AUTO && value != OPUS_BITRATE_MAX && value <= 0 {
                return Err(OpusMultistreamEncoderError::BadArgument);
            }
            let clamped = if value == OPUS_AUTO || value == OPUS_BITRATE_MAX {
                value
            } else {
                let channels = i32::try_from(encoder.layout.nb_channels)
                    .map_err(|_| OpusMultistreamEncoderError::BadArgument)?;
                let min_rate = 500i32.saturating_mul(channels);
                let max_rate = 750_000i32.saturating_mul(channels);
                value.clamp(min_rate, max_rate)
            };
            encoder.bitrate_bps = clamped;
        }
        OpusMultistreamEncoderCtlRequest::GetBitrate(out) => {
            let mut total = 0i64;
            for enc in &mut encoder.encoders {
                let mut rate = 0i32;
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::GetBitrate(&mut rate))?;
                total += i64::from(rate);
            }
            *out = i32::try_from(total).map_err(|_| OpusMultistreamEncoderError::InternalError)?;
        }
        OpusMultistreamEncoderCtlRequest::SetForceChannels(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetForceChannels(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetForceChannels(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetForceChannels(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetMaxBandwidth(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetMaxBandwidth(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetMaxBandwidth(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetMaxBandwidth(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetBandwidth(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetBandwidth(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetBandwidth(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetBandwidth(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetVbr(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetVbr(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetVbr(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetVbr(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetVbrConstraint(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetVbrConstraint(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetVbrConstraint(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetVbrConstraint(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetComplexity(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetComplexity(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetComplexity(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetComplexity(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetSignal(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetSignal(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetSignal(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetSignal(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::GetVoiceRatio(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetVoiceRatio(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetPacketLossPerc(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetPacketLossPerc(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetPacketLossPerc(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetPacketLossPerc(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetInbandFec(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetInbandFec(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetInbandFec(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetInbandFec(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetDtx(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetDtx(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetDtx(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetDtx(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::GetInDtx(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetInDtx(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetLsbDepth(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetLsbDepth(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetLsbDepth(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetLsbDepth(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetExpertFrameDuration(value) => {
            encoder.variable_duration = value;
        }
        OpusMultistreamEncoderCtlRequest::GetExpertFrameDuration(out) => {
            *out = encoder.variable_duration;
        }
        OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetPredictionDisabled(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetPredictionDisabled(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetPhaseInversionDisabled(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetPhaseInversionDisabled(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetPhaseInversionDisabled(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetPhaseInversionDisabled(out),
            )?;
        }
        #[cfg(feature = "enable_qext")]
        OpusMultistreamEncoderCtlRequest::SetQext(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetQext(value))?;
            }
        }
        #[cfg(feature = "enable_qext")]
        OpusMultistreamEncoderCtlRequest::GetQext(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetQext(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetDredDuration(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetDredDuration(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetDredDuration(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetDredDuration(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::SetDnnBlob(data) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetDnnBlob(data))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::SetForceMode(value) => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::SetForceMode(value))?;
            }
        }
        OpusMultistreamEncoderCtlRequest::GetSampleRate(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetSampleRate(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::GetLookahead(out) => {
            opus_encoder_ctl(
                encoder
                    .encoders
                    .first_mut()
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::GetLookahead(out),
            )?;
        }
        OpusMultistreamEncoderCtlRequest::GetFinalRange(out) => {
            let mut acc = 0u32;
            for enc in &mut encoder.encoders {
                let mut value = 0u32;
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::GetFinalRange(&mut value))?;
                acc ^= value;
            }
            *out = acc;
        }
        OpusMultistreamEncoderCtlRequest::ResetState => {
            for enc in &mut encoder.encoders {
                opus_encoder_ctl(enc, OpusEncoderCtlRequest::ResetState)?;
            }
            encoder.surround_window.fill(0.0);
            encoder.surround_preemphasis.fill(0.0);
            #[cfg(feature = "fixed_point")]
            {
                encoder.surround_fixed_window.fill(0);
                encoder.surround_fixed_preemphasis.fill(0);
            }
        }
    }

    Ok(())
}

trait MultistreamPcm: Copy + Default {
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    fn native24(self) -> i32 {
        float2int((self.normalized() * 8388608.0).clamp(-16777216.0, 16777216.0))
    }
    const DEPTH: i32;
    fn normalized(self) -> f32;
    fn coding(self) -> f32 {
        self.normalized()
    }
}

impl MultistreamPcm for i16 {
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    fn native24(self) -> i32 {
        i32::from(self) << 8
    }
    const DEPTH: i32 = 16;
    fn normalized(self) -> f32 {
        f32::from(self) * (1.0 / 32768.0)
    }
}

impl MultistreamPcm for i32 {
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    fn native24(self) -> i32 {
        self
    }
    const DEPTH: i32 = 24;
    fn normalized(self) -> f32 {
        self as f32 * (1.0 / 8388608.0)
    }
    fn coding(self) -> f32 {
        #[cfg(all(feature = "fixed_point", not(feature = "enable_res24")))]
        {
            return ((self.wrapping_add(128) >> 8).clamp(-32768, 32767)) as f32 / 32768.0;
        }
        #[cfg(not(all(feature = "fixed_point", not(feature = "enable_res24"))))]
        self.normalized()
    }
}

impl MultistreamPcm for f32 {
    const DEPTH: i32 = 24;
    fn normalized(self) -> f32 {
        self
    }
}

fn surround_log_sum(a: f32, b: f32) -> f32 {
    const DIFF: [f32; 17] = [
        0.5000000, 0.2924813, 0.1609640, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136,
        0.0028123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (maximum, difference) = if a > b { (a, a - b) } else { (b, b - a) };
    if !(difference < 8.0) {
        return maximum;
    }
    let low = libm::floorf(2.0 * difference) as usize;
    let fraction = 2.0 * difference - low as f32;
    maximum + DIFF[low] + fraction * (DIFF[low + 1] - DIFF[low])
}

/// Computes inter-channel masking while preserving MDCT overlap and pre-emphasis
/// memory across packets. Each channel contributes 21 CELT masking values.
fn surround_analysis<T: MultistreamPcm>(
    pcm: &[T],
    channels: usize,
    length: usize,
    rate: i32,
    memory: &mut Vec<f32>,
    preemphasis: &mut Vec<f32>,
) -> Vec<f32> {
    let mode = opus_custom_mode_find_static_ref(48_000, 960).expect("standard CELT mode");
    #[cfg(feature = "enable_qext")]
    let mode = if rate == 96_000 {
        crate::celt::canonical_mode_96k().expect("96 kHz CELT mode")
    } else {
        mode
    };
    let overlap = mode.overlap;
    let upsample = resampling_factor(rate) as usize;
    let frame_size = length * upsample;
    let lm = (0..mode.max_lm)
        .find(|&lm| mode.short_mdct_size << lm == frame_size)
        .unwrap_or(mode.max_lm);
    let frequency_size = mode.short_mdct_size << lm;
    let positions: &[usize] = match channels {
        3 => &[1, 2, 3],
        4 => &[1, 3, 1, 3],
        5 => &[1, 2, 3, 1, 3],
        6 => &[1, 2, 3, 1, 3, 0],
        7 => &[1, 2, 3, 1, 3, 2, 0],
        8 => &[1, 2, 3, 1, 3, 1, 3, 0],
        _ => unreachable!("surround layouts contain three to eight channels"),
    };
    memory.resize(channels * overlap, 0.0);
    preemphasis.resize(channels, 0.0);
    let mut mask = [[-28.0f32; 21]; 3];
    let mut band_log = vec![0.0f32; 21 * channels];
    let mut input = vec![0.0f32; frame_size + overlap];
    let mut samples = vec![0.0f32; length];
    let mut frequency = vec![0.0f32; frequency_size];
    for channel in 0..channels {
        input[..overlap].copy_from_slice(&memory[channel * overlap..(channel + 1) * overlap]);
        for (index, sample) in samples.iter_mut().enumerate() {
            *sample = pcm[index * channels + channel].normalized();
        }
        celt_preemphasis(
            &samples,
            &mut input[overlap..],
            frame_size,
            1,
            upsample,
            &mode.pre_emphasis,
            &mut preemphasis[channel],
            false,
        );
        let energy = celt_inner_prod(&input, &input);
        if !(energy < 1e18) || energy.is_nan() {
            input.fill(0.0);
            preemphasis[channel] = 0.0;
        }
        let mut band_energy = [0.0f32; 21];
        for frame in 0..frame_size / frequency_size {
            clt_mdct_forward(
                mode.mdct,
                &input[frequency_size * frame..],
                &mut frequency,
                mode.window,
                overlap,
                mode.max_lm - lm,
                1,
            );
            if upsample != 1 {
                let bound = frequency_size / upsample;
                for bin in &mut frequency[..bound] {
                    *bin *= upsample as f32;
                }
                frequency[bound..].fill(0.0);
            }
            let mut energy = [0.0f32; 21];
            compute_band_energies(mode, &frequency, &mut energy, 21, 1, lm, 0);
            for (band, energy) in band_energy.iter_mut().zip(energy) {
                *band = band.max(energy);
            }
        }
        let logs = &mut band_log[channel * 21..(channel + 1) * 21];
        amp2_log2(mode, 21, 21, &band_energy, logs, 1);
        for i in 1..21 {
            logs[i] = logs[i].max(logs[i - 1] - 1.0);
        }
        for i in (0..20).rev() {
            logs[i] = logs[i].max(logs[i + 1] - 2.0);
        }
        for i in 0..21 {
            match positions[channel] {
                1 => mask[0][i] = surround_log_sum(mask[0][i], logs[i]),
                3 => mask[2][i] = surround_log_sum(mask[2][i], logs[i]),
                2 => {
                    mask[0][i] = surround_log_sum(mask[0][i], logs[i] - 0.5);
                    mask[2][i] = surround_log_sum(mask[2][i], logs[i] - 0.5);
                }
                _ => {}
            }
        }
        memory[channel * overlap..(channel + 1) * overlap].copy_from_slice(&input[frame_size..]);
    }
    for i in 0..21 {
        mask[1][i] = mask[0][i].min(mask[2][i]);
    }
    let channel_offset = 0.5 * celt_log2(2.0 / (channels - 1) as f32);
    for side in &mut mask {
        for band in side {
            *band += channel_offset;
        }
    }
    for channel in 0..channels {
        for i in 0..21 {
            if positions[channel] == 0 {
                band_log[channel * 21 + i] = 0.0;
            } else {
                band_log[channel * 21 + i] -= mask[positions[channel] - 1][i];
            }
        }
    }
    band_log
}

#[cfg(feature = "fixed_point")]
fn surround_log_sum_fixed(a: i32, b: i32) -> i32 {
    use crate::celt::{mult16_32_q15, qconst32};
    const DIFF: [f32; 17] = [
        0.5000000, 0.2924813, 0.1609640, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136,
        0.0028123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (maximum, difference) = if a > b { (a, a - b) } else { (b, b - a) };
    // The pinned reference returns opus_val16 here, including the narrowing
    // conversion from its Q24 logarithmic intermediate values.
    if difference >= 8 << 24 {
        return i32::from(maximum as i16);
    }
    let low = (difference >> 23) as usize;
    let fraction = ((difference - ((low as i32) << 23)) >> 8) as i16;
    let current = qconst32(f64::from(DIFF[low]), 24);
    let next = qconst32(f64::from(DIFF[low + 1]), 24);
    i32::from((maximum + current + mult16_32_q15(fraction, next - current)) as i16)
}

#[cfg(feature = "fixed_point")]
fn surround_analysis_fixed<T: MultistreamPcm>(
    pcm: &[T],
    channels: usize,
    length: usize,
    rate: i32,
    memory: &mut Vec<i32>,
    preemphasis: &mut Vec<i32>,
) -> Vec<i32> {
    let mode = opus_custom_mode_find_static_ref(48_000, 960).expect("standard CELT mode");
    #[cfg(feature = "enable_qext")]
    let mode = if rate == 96_000 {
        crate::celt::canonical_mode_96k().expect("96 kHz CELT mode")
    } else {
        mode
    };
    use crate::celt::celt_preemphasis_fixed;
    use crate::celt::compute_band_energies_fixed;
    use crate::celt::{FixedMdctLookup, clt_mdct_forward_fixed};
    use crate::celt::{amp2_log2_fixed, celt_log2_q10};
    const UNIT: i32 = 1 << 24;
    let mdct = FixedMdctLookup::new(mode.mdct.len(), mode.mdct.max_shift());
    let window = mode.fixed_window();
    let overlap = mode.overlap;
    let upsample = resampling_factor(rate) as usize;
    let frame_size = length * upsample;
    let lm = (0..mode.max_lm)
        .find(|&lm| mode.short_mdct_size << lm == frame_size)
        .unwrap_or(mode.max_lm);
    let frequency_size = mode.short_mdct_size << lm;
    let positions: &[usize] = match channels {
        3 => &[1, 2, 3],
        4 => &[1, 3, 1, 3],
        5 => &[1, 2, 3, 1, 3],
        6 => &[1, 2, 3, 1, 3, 0],
        7 => &[1, 2, 3, 1, 3, 2, 0],
        8 => &[1, 2, 3, 1, 3, 1, 3, 0],
        _ => unreachable!("surround layouts contain three to eight channels"),
    };
    memory.resize(channels * overlap, 0);
    preemphasis.resize(channels, 0);
    let mut mask = [[-28 * UNIT; 21]; 3];
    let mut band_log = vec![0i32; 21 * channels];
    let mut input = vec![0i32; frame_size + overlap];
    let mut samples = vec![0.0f32; length];
    let mut frequency = vec![0i32; frequency_size];
    for channel in 0..channels {
        input[..overlap].copy_from_slice(&memory[channel * overlap..(channel + 1) * overlap]);
        for (index, sample) in samples.iter_mut().enumerate() {
            *sample = pcm[index * channels + channel].coding();
        }
        celt_preemphasis_fixed(
            &samples,
            &mut input[overlap..],
            frame_size,
            1,
            upsample,
            &mode.pre_emphasis,
            &mut preemphasis[channel],
            false,
        );
        let mut band_energy = [0i32; 21];
        for frame in 0..frame_size / frequency_size {
            clt_mdct_forward_fixed(
                &mdct,
                &input[frequency_size * frame..],
                &mut frequency,
                &window,
                overlap,
                mode.max_lm - lm,
                1,
            );
            if upsample != 1 {
                let bound = frequency_size / upsample;
                for bin in &mut frequency[..bound] {
                    *bin = bin.wrapping_mul(upsample as i32);
                }
                frequency[bound..].fill(0);
            }
            let mut energy = [0i32; 21];
            compute_band_energies_fixed(mode, &frequency, &mut energy, 21, 1, lm);
            for (band, energy) in band_energy.iter_mut().zip(energy) {
                *band = (*band).max(energy);
            }
        }
        let logs = &mut band_log[channel * 21..(channel + 1) * 21];
        amp2_log2_fixed(mode, 21, 21, &band_energy, logs, 1);
        for i in 1..21 {
            logs[i] = logs[i].max(logs[i - 1] - UNIT);
        }
        for i in (0..20).rev() {
            logs[i] = logs[i].max(logs[i + 1] - 2 * UNIT);
        }
        for i in 0..21 {
            match positions[channel] {
                1 => mask[0][i] = surround_log_sum_fixed(mask[0][i], logs[i]),
                3 => mask[2][i] = surround_log_sum_fixed(mask[2][i], logs[i]),
                2 => {
                    mask[0][i] = surround_log_sum_fixed(mask[0][i], logs[i] - UNIT / 2);
                    mask[2][i] = surround_log_sum_fixed(mask[2][i], logs[i] - UNIT / 2);
                }
                _ => {}
            }
        }
        memory[channel * overlap..(channel + 1) * overlap].copy_from_slice(&input[frame_size..]);
    }
    for i in 0..21 {
        mask[1][i] = mask[0][i].min(mask[2][i]);
    }
    let channel_offset = i32::from(celt_log2_q10(32768 / (channels - 1) as i32) >> 1);
    for side in &mut mask {
        for band in side {
            *band += channel_offset;
        }
    }
    for channel in 0..channels {
        for i in 0..21 {
            if positions[channel] == 0 {
                band_log[channel * 21 + i] = 0;
            } else {
                band_log[channel * 21 + i] -= mask[positions[channel] - 1][i];
            }
        }
    }
    band_log
}

fn extract_channel<T: Copy>(
    input: &[T],
    input_channels: usize,
    channel: usize,
    frame_size: usize,
    output: &mut [T],
    output_stride: usize,
    output_offset: usize,
) -> Result<(), OpusMultistreamEncoderError> {
    for i in 0..frame_size {
        let src = i
            .checked_mul(input_channels)
            .and_then(|base| base.checked_add(channel))
            .ok_or(OpusMultistreamEncoderError::BadArgument)?;
        let dst = output_offset
            .checked_add(
                i.checked_mul(output_stride)
                    .ok_or(OpusMultistreamEncoderError::BadArgument)?,
            )
            .ok_or(OpusMultistreamEncoderError::BadArgument)?;
        output[dst] = *input
            .get(src)
            .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    }
    Ok(())
}

pub fn opus_multistream_encode_with_options(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
    options: OpusMultistreamEncodeOptions<'_>,
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_native_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        options,
        opus_encode_with_options,
        None,
    )
}

pub(crate) struct MultistreamAnalysisInput<'a> {
    pub pcm: &'a [f32],
    pub depth: i32,
}

pub(crate) fn opus_multistream_encode_float_with_analysis(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
    analysis: MultistreamAnalysisInput<'_>,
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_native_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        OpusMultistreamEncodeOptions::default(),
        opus_encode_float_with_options,
        Some(analysis),
    )
}

#[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
pub(crate) fn opus_multistream_encode24_with_analysis(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
    analysis: MultistreamAnalysisInput<'_>,
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_native_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        OpusMultistreamEncodeOptions::default(),
        opus_encode24_with_options,
        Some(analysis),
    )
}

fn encode_stream_with_analysis<T: MultistreamPcm>(
    encoder: &mut OpusEncoder<'_>,
    pcm: &[T],
    frame_size: usize,
    data: &mut [u8],
    options: OpusEncodeOptions<'_>,
    analysis: OpusAnalysisInput<'_>,
) -> Result<usize, OpusEncodeError> {
    #[cfg(all(feature = "fixed_point", feature = "enable_res24"))]
    {
        let coding: Vec<i32> = pcm.iter().map(|&sample| sample.native24()).collect();
        crate::opus_encoder::opus_encode24_with_analysis(
            encoder, &coding, frame_size, data, options, analysis,
        )
    }
    #[cfg(not(all(feature = "fixed_point", feature = "enable_res24")))]
    {
        let coding: Vec<f32> = pcm.iter().map(|&sample| sample.coding()).collect();
        opus_encode_float_with_analysis(encoder, &coding, frame_size, data, options, analysis)
    }
}

fn opus_multistream_encode_native_with_options<T: MultistreamPcm>(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[T],
    frame_size: usize,
    data: &mut [u8],
    options: OpusMultistreamEncodeOptions<'_>,
    encode: fn(
        &mut OpusEncoder<'_>,
        &[T],
        usize,
        &mut [u8],
        OpusEncodeOptions<'_>,
    ) -> Result<usize, OpusEncodeError>,
    analysis: Option<MultistreamAnalysisInput<'_>>,
) -> Result<usize, OpusMultistreamEncoderError> {
    let channels = encoder.layout.nb_channels;
    if channels == 0 || frame_size == 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let required_pcm = channels
        .checked_mul(frame_size)
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    if pcm.len() < required_pcm {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    let original_analysis: Vec<f32> = if analysis.is_none() {
        pcm.iter().map(|&sample| sample.normalized()).collect()
    } else {
        Vec::new()
    };
    let analysis = Some(analysis.unwrap_or(MultistreamAnalysisInput {
        pcm: &original_analysis,
        depth: T::DEPTH,
    }));
    let frame_size = frame_size_select(
        i32::try_from(frame_size).map_err(|_| OpusMultistreamEncoderError::BadArgument)?,
        encoder.variable_duration,
        encoder.sample_rate,
    )
    .ok_or(OpusMultistreamEncoderError::BadArgument)? as usize;

    let nb_streams = encoder.layout.nb_streams;
    if nb_streams == 0 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }
    if let Some(masks) = options.stream_energy_masks
        && masks.len() != nb_streams
    {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }

    let mut smallest_packet = nb_streams
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    let hundred_ms = encoder.sample_rate as usize / frame_size == 10;
    if hundred_ms {
        smallest_packet += nb_streams;
    }
    if data.len() < smallest_packet {
        return Err(OpusMultistreamEncoderError::BufferTooSmall);
    }

    #[cfg(feature = "fixed_point")]
    let fixed_surround_masks =
        if encoder.mapping_type == MappingType::Surround && encoder.application != 2052 {
            Some(surround_analysis_fixed(
                pcm,
                channels,
                frame_size,
                encoder.sample_rate,
                &mut encoder.surround_fixed_window,
                &mut encoder.surround_fixed_preemphasis,
            ))
        } else {
            None
        };
    #[cfg(feature = "fixed_point")]
    let surround_masks: Option<Vec<f32>> = fixed_surround_masks.as_ref().map(|mask| {
        mask.iter()
            .map(|&value| value as f32 / 16777216.0)
            .collect()
    });
    #[cfg(not(feature = "fixed_point"))]
    let surround_masks =
        if encoder.mapping_type == MappingType::Surround && encoder.application != 2052 {
            Some(surround_analysis(
                pcm,
                channels,
                frame_size,
                encoder.sample_rate,
                &mut encoder.surround_window,
                &mut encoder.surround_preemphasis,
            ))
        } else {
            None
        };

    let mut rates = vec![0i32; nb_streams];
    let rate_sum = rate_allocation(
        &encoder.layout,
        encoder.mapping_type,
        encoder.bitrate_bps,
        encoder.lfe_stream,
        frame_size,
        encoder.sample_rate,
        &mut rates,
    )
    .ok_or(OpusMultistreamEncoderError::BadArgument)?;
    let mut vbr = false;
    opus_encoder_ctl(
        &mut encoder.encoders[0],
        OpusEncoderCtlRequest::GetVbr(&mut vbr),
    )?;
    let mut max_data_bytes = data.len();
    if !vbr && encoder.bitrate_bps != OPUS_BITRATE_MAX {
        let bitrate = if encoder.bitrate_bps == OPUS_AUTO {
            rate_sum
        } else {
            encoder.bitrate_bps
        };
        let budget = ((i64::from(bitrate) * frame_size as i64 / i64::from(encoder.sample_rate) + 4)
            / 8) as usize;
        max_data_bytes = max_data_bytes.min(if encoder.bitrate_bps == OPUS_AUTO {
            budget
        } else {
            budget.max(smallest_packet)
        });
    }

    let mut total_written = 0usize;
    for stream in 0..nb_streams {
        let self_delimited = stream + 1 != nb_streams;
        let remaining_streams = nb_streams - stream - 1;
        let mut reserve_min = remaining_streams
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .unwrap_or(0);
        if hundred_ms {
            reserve_min += remaining_streams;
        }
        let available = max_data_bytes
            .checked_sub(total_written)
            .and_then(|value| value.checked_sub(reserve_min))
            .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?
            .min(MS_FRAME_TMP);

        let size_overhead = if self_delimited {
            if available > 253 { 2 } else { 1 }
        } else {
            0
        };
        if available <= size_overhead {
            return Err(OpusMultistreamEncoderError::BufferTooSmall);
        }
        let mut tmp = vec![0u8; available - size_overhead];

        if let Some(mut rate) = rates.get(stream).copied() {
            if !vbr && !self_delimited {
                rate = (tmp.len() as i64 * 8 * i64::from(encoder.sample_rate) / frame_size as i64)
                    as i32;
            }
            opus_encoder_ctl(
                encoder
                    .encoders
                    .get_mut(stream)
                    .ok_or(OpusMultistreamEncoderError::InternalError)?,
                OpusEncoderCtlRequest::SetBitrate(rate),
            )?;
        }
        if encoder.mapping_type == MappingType::Surround {
            let mut equivalent_rate = encoder.bitrate_bps;
            if frame_size * 50 < encoder.sample_rate as usize {
                equivalent_rate -=
                    60 * (encoder.sample_rate / frame_size as i32 - 50) * channels as i32;
            }
            let bandwidth = if equivalent_rate > 10_000 * channels as i32 {
                1105
            } else if equivalent_rate > 7_000 * channels as i32 {
                1104
            } else if equivalent_rate > 5_000 * channels as i32 {
                1103
            } else {
                1101
            };
            opus_encoder_ctl(
                &mut encoder.encoders[stream],
                OpusEncoderCtlRequest::SetBandwidth(bandwidth),
            )?;
            if stream < encoder.layout.nb_coupled_streams {
                opus_encoder_ctl(
                    &mut encoder.encoders[stream],
                    OpusEncoderCtlRequest::SetForceMode(1002),
                )?;
                opus_encoder_ctl(
                    &mut encoder.encoders[stream],
                    OpusEncoderCtlRequest::SetForceChannels(2),
                )?;
            }
        } else if encoder.mapping_type == MappingType::Ambisonics {
            opus_encoder_ctl(
                &mut encoder.encoders[stream],
                OpusEncoderCtlRequest::SetForceMode(1002),
            )?;
        }
        let mut stream_mask = Vec::new();
        if let Some(masks) = &surround_masks {
            if stream < encoder.layout.nb_coupled_streams {
                let left = get_left_channel(&encoder.layout, stream, None).unwrap();
                let right = get_right_channel(&encoder.layout, stream, None).unwrap();
                stream_mask.extend_from_slice(&masks[left * 21..(left + 1) * 21]);
                stream_mask.extend_from_slice(&masks[right * 21..(right + 1) * 21]);
            } else {
                let channel = get_mono_channel(&encoder.layout, stream, None).unwrap();
                stream_mask.extend_from_slice(&masks[channel * 21..(channel + 1) * 21]);
            }
        }
        #[cfg(feature = "fixed_point")]
        {
            let mask = if options
                .stream_energy_masks
                .and_then(|masks| masks.get(stream))
                .copied()
                .flatten()
                .is_some()
            {
                None
            } else {
                fixed_surround_masks.as_ref().map(|masks| {
                    if stream < encoder.layout.nb_coupled_streams {
                        let left = get_left_channel(&encoder.layout, stream, None).unwrap();
                        let right = get_right_channel(&encoder.layout, stream, None).unwrap();
                        masks[left * 21..(left + 1) * 21]
                            .iter()
                            .chain(&masks[right * 21..(right + 1) * 21])
                            .copied()
                            .collect()
                    } else {
                        let channel = get_mono_channel(&encoder.layout, stream, None).unwrap();
                        masks[channel * 21..(channel + 1) * 21].to_vec()
                    }
                })
            };
            encoder.encoders[stream].set_fixed_energy_mask(mask);
        }
        let encode_options = OpusEncodeOptions {
            energy_masking: options
                .stream_energy_masks
                .and_then(|masks| masks.get(stream))
                .copied()
                .flatten()
                .or_else(|| (!stream_mask.is_empty()).then_some(stream_mask.as_slice())),
        };

        if stream < encoder.layout.nb_coupled_streams {
            let left = get_left_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let right = get_right_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;

            let mut coupled_pcm = vec![T::default(); frame_size * 2];
            extract_channel(pcm, channels, left, frame_size, &mut coupled_pcm, 2, 0)?;
            extract_channel(pcm, channels, right, frame_size, &mut coupled_pcm, 2, 1)?;

            let encoder_state = encoder
                .encoders
                .get_mut(stream)
                .ok_or(OpusMultistreamEncoderError::InternalError)?;
            let encoded = if let Some(analysis) = &analysis {
                encode_stream_with_analysis(
                    encoder_state,
                    &coupled_pcm,
                    frame_size,
                    &mut tmp,
                    encode_options,
                    OpusAnalysisInput {
                        pcm: analysis.pcm,
                        channels: channels as i32,
                        c1: left as i32,
                        c2: right as i32,
                        depth: analysis.depth,
                    },
                )
            } else {
                encode(
                    encoder_state,
                    &coupled_pcm,
                    frame_size,
                    &mut tmp,
                    encode_options,
                )
            };

            #[cfg(feature = "fixed_point")]
            encoder_state.set_fixed_energy_mask(None);
            let len = encoded?;
            let written = write_stream_packet(
                &tmp[..len],
                self_delimited,
                !vbr && !self_delimited,
                &mut data[total_written..max_data_bytes],
            )
            .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?;
            total_written += written;
        } else {
            let chan = get_mono_channel(&encoder.layout, stream, None)
                .ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let mut mono_pcm = vec![T::default(); frame_size];
            extract_channel(pcm, channels, chan, frame_size, &mut mono_pcm, 1, 0)?;

            let encoder_state = encoder
                .encoders
                .get_mut(stream)
                .ok_or(OpusMultistreamEncoderError::InternalError)?;
            let encoded = if let Some(analysis) = &analysis {
                encode_stream_with_analysis(
                    encoder_state,
                    &mono_pcm,
                    frame_size,
                    &mut tmp,
                    encode_options,
                    OpusAnalysisInput {
                        pcm: analysis.pcm,
                        channels: channels as i32,
                        c1: chan as i32,
                        c2: -1,
                        depth: analysis.depth,
                    },
                )
            } else {
                encode(
                    encoder_state,
                    &mono_pcm,
                    frame_size,
                    &mut tmp,
                    encode_options,
                )
            };

            #[cfg(feature = "fixed_point")]
            encoder_state.set_fixed_energy_mask(None);
            let len = encoded?;
            let written = write_stream_packet(
                &tmp[..len],
                self_delimited,
                !vbr && !self_delimited,
                &mut data[total_written..max_data_bytes],
            )
            .ok_or(OpusMultistreamEncoderError::BufferTooSmall)?;
            total_written += written;
        }
    }

    Ok(total_written)
}

pub fn opus_multistream_encode(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i16],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        OpusMultistreamEncodeOptions::default(),
    )
}

fn write_stream_packet(
    stream_packet: &[u8],
    self_delimited: bool,
    pad: bool,
    out: &mut [u8],
) -> Option<usize> {
    let mut repacketizer = OpusRepacketizer::new();
    repacketizer
        .opus_repacketizer_cat(stream_packet, stream_packet.len())
        .ok()?;
    repacketizer
        .opus_repacketizer_out_range_impl(
            0,
            repacketizer.opus_repacketizer_get_nb_frames(),
            out,
            out.len(),
            self_delimited,
            pad,
            &[],
        )
        .ok()
}

pub fn opus_multistream_encode_float_with_options(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusMultistreamEncodeOptions<'_>,
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_native_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        options,
        opus_encode_float_with_options,
        None,
    )
}

pub fn opus_multistream_encode_float(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[f32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_float_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        OpusMultistreamEncodeOptions::default(),
    )
}

pub fn opus_multistream_encode24_with_options(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
    options: OpusMultistreamEncodeOptions<'_>,
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode_native_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        options,
        opus_encode24_with_options,
        None,
    )
}

pub fn opus_multistream_encode24(
    encoder: &mut OpusMultistreamEncoder<'_>,
    pcm: &[i32],
    frame_size: usize,
    data: &mut [u8],
) -> Result<usize, OpusMultistreamEncoderError> {
    opus_multistream_encode24_with_options(
        encoder,
        pcm,
        frame_size,
        data,
        OpusMultistreamEncodeOptions::default(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusMultistreamSurroundLayout {
    pub streams: usize,
    pub coupled_streams: usize,
    pub mapping: Vec<u8>,
}

pub fn opus_multistream_surround_encoder_get_size(
    channels: usize,
    mapping_family: u8,
) -> Option<usize> {
    let (streams, coupled_streams, _) = surround_layout(channels, mapping_family).ok()?;
    let mut size = opus_multistream_encoder_get_size(streams, coupled_streams)?;
    if channels > 2 {
        size = size.checked_add(
            channels.checked_mul(
                120usize
                    .checked_mul(core::mem::size_of::<crate::celt::OpusVal32>())?
                    .checked_add(core::mem::size_of::<crate::celt::OpusVal32>())?,
            )?,
        )?;
    }
    Some(size)
}

pub fn opus_multistream_surround_encoder_create(
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<
    (
        OpusMultistreamEncoder<'static>,
        OpusMultistreamSurroundLayout,
    ),
    OpusMultistreamEncoderError,
> {
    let (streams, coupled_streams, mapping) = surround_layout(channels, mapping_family)?;
    let mut encoder = opus_multistream_encoder_create(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        application,
    )?;

    let (mapping_type, lfe_stream) = surround_mapping_type(channels, mapping_family, streams);
    encoder.mapping_type = mapping_type;
    encoder.lfe_stream = lfe_stream;
    if let Some(stream) = lfe_stream {
        opus_encoder_ctl(
            &mut encoder.encoders[stream],
            OpusEncoderCtlRequest::SetLfe(true),
        )?;
    }

    Ok((
        encoder,
        OpusMultistreamSurroundLayout {
            streams,
            coupled_streams,
            mapping,
        },
    ))
}

pub fn opus_multistream_surround_encoder_init(
    encoder: &mut OpusMultistreamEncoder<'_>,
    sample_rate: i32,
    channels: usize,
    mapping_family: u8,
    application: i32,
) -> Result<OpusMultistreamSurroundLayout, OpusMultistreamEncoderError> {
    let (streams, coupled_streams, mapping) = surround_layout(channels, mapping_family)?;
    encoder.init(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        application,
    )?;

    let (mapping_type, lfe_stream) = surround_mapping_type(channels, mapping_family, streams);
    encoder.mapping_type = mapping_type;
    encoder.lfe_stream = lfe_stream;
    if let Some(stream) = lfe_stream {
        opus_encoder_ctl(
            &mut encoder.encoders[stream],
            OpusEncoderCtlRequest::SetLfe(true),
        )?;
    }

    Ok(OpusMultistreamSurroundLayout {
        streams,
        coupled_streams,
        mapping,
    })
}

fn surround_mapping_type(
    channels: usize,
    mapping_family: u8,
    streams: usize,
) -> (MappingType, Option<usize>) {
    match mapping_family {
        1 if channels > 2 => (
            MappingType::Surround,
            if channels >= 6 {
                streams.checked_sub(1)
            } else {
                None
            },
        ),
        2 => (MappingType::Ambisonics, None),
        _ => (MappingType::None, None),
    }
}

fn surround_layout(
    channels: usize,
    mapping_family: u8,
) -> Result<(usize, usize, Vec<u8>), OpusMultistreamEncoderError> {
    if channels == 0 || channels > 255 {
        return Err(OpusMultistreamEncoderError::BadArgument);
    }

    let mut mapping = Vec::with_capacity(channels);
    match mapping_family {
        0 => match channels {
            1 => {
                mapping.push(0);
                Ok((1, 0, mapping))
            }
            2 => {
                mapping.extend_from_slice(&[0, 1]);
                Ok((1, 1, mapping))
            }
            _ => Err(OpusMultistreamEncoderError::Unimplemented),
        },
        1 => {
            if !(1..=8).contains(&channels) {
                return Err(OpusMultistreamEncoderError::Unimplemented);
            }
            let layout = &VORBIS_MAPPINGS[channels - 1];
            mapping.extend_from_slice(&layout.mapping[..channels]);
            Ok((layout.nb_streams, layout.nb_coupled_streams, mapping))
        }
        255 => {
            mapping.extend((0..channels).map(|v| v as u8));
            Ok((channels, 0, mapping))
        }
        2 => {
            let (streams, coupled_streams) =
                validate_ambisonics(channels).ok_or(OpusMultistreamEncoderError::BadArgument)?;
            let uncoupled_streams = streams - coupled_streams;
            for i in 0..uncoupled_streams {
                mapping.push((i + coupled_streams * 2) as u8);
            }
            for i in 0..coupled_streams * 2 {
                mapping.push(i as u8);
            }
            Ok((streams, coupled_streams, mapping))
        }
        _ => Err(OpusMultistreamEncoderError::Unimplemented),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::opus_decoder::{opus_decode, opus_decoder_create};
    use crate::opus_encoder::opus_encode;
    use alloc::vec;

    #[test]
    #[cfg(not(feature = "fixed_point"))]
    fn short_multistream_packets_match_reference() {
        // Scalar C commit 503d81b138d76621aae4b12786e90de48aa8db3a.
        // The second packet rejects a spurious very short pitch period.
        let expected = [
            (
                "e0137d306a1f010986624dd0881223df8292429893e07d306a1eb15ea6e99fbf1b736c287231a631",
                101062229u32,
            ),
            (
                "e0137bdd960a50cf7b4c7c1a8cf5b7ce2e32000007e071613593c521158745e40cbf5ca8e0360001",
                447884544u32,
            ),
            (
                "e0137a0b7899110181fb571db41ab98a72ece0010de07d5aa7c513e9996f96494e9347024ae7eaff",
                49976576u32,
            ),
        ];
        let mut encoder = opus_multistream_encoder_create(48_000, 2, 2, 0, &[0, 1], 2051).unwrap();
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetBitrate(128_000),
        )
        .unwrap();
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetVbr(false),
        )
        .unwrap();
        let mut packet = [0u8; 40];
        for (frame, (hex, range)) in expected.iter().enumerate() {
            let signal: Vec<f32> = (0..240)
                .map(|i| {
                    let channel = i % 2;
                    let time = frame * 120 + i / 2;
                    (((time * (127 + channel * 73) + channel * 199) % 65536) as i32 - 32768) as f32
                        * 211.0
                        / 8388608.0
                })
                .collect();
            let length =
                opus_multistream_encode_float(&mut encoder, &signal, 120, &mut packet).unwrap();
            let expected_packet: Vec<u8> = hex
                .as_bytes()
                .chunks_exact(2)
                .map(|bytes| u8::from_str_radix(core::str::from_utf8(bytes).unwrap(), 16).unwrap())
                .collect();
            assert_eq!(&packet[..length], expected_packet);
            let mut actual_range = 0;
            opus_multistream_encoder_ctl(
                &mut encoder,
                OpusMultistreamEncoderCtlRequest::GetFinalRange(&mut actual_range),
            )
            .unwrap();
            assert_eq!(actual_range, *range);
        }
    }

    #[test]
    #[ignore = "requires pinned C reference built by tools/reference/build.sh"]
    fn surround_mask_matches_pinned_reference() {
        use std::fs;
        use std::path::PathBuf;
        use std::process::Command;
        use std::string::ToString;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut directory = "target/reference".to_string();
        if cfg!(feature = "fixed_point") {
            directory.push_str("-fixed");
            if cfg!(feature = "enable_res24") {
                directory.push_str("-res24");
            }
        }
        if cfg!(feature = "enable_qext") {
            directory.push_str("-qext");
        }
        if cfg!(feature = "pfa") {
            directory.push_str("-pfa");
        }
        let reference_dir = root.join(directory);
        let reference = reference_dir.join("source");
        let executable = reference_dir.join("surround-mask-reference");
        let mut compiler = Command::new("cc");
        compiler.args(["-O2", "-std=c99", "-DOPUS_BUILD", "-DHAVE_LRINTF"]);
        if cfg!(feature = "enable_qext") {
            compiler.arg("-DENABLE_QEXT");
        }
        if cfg!(feature = "pfa") {
            compiler.arg("-DENABLE_PFA");
        }
        if cfg!(feature = "fixed_point") {
            compiler.arg("-DFIXED_POINT");
            if cfg!(feature = "enable_res24") {
                compiler.arg("-DENABLE_RES24");
            }
        }
        for path in ["include", "celt", "src", "silk"] {
            compiler.arg("-I").arg(reference.join(path));
        }
        assert!(
            compiler
                .arg(root.join("rust/tests/multistream_reference.c"))
                .arg(reference.join("libopus.a"))
                .arg("-lm")
                .arg("-o")
                .arg(&executable)
                .status()
                .unwrap()
                .success()
        );
        let mut rates = vec![8000, 12000, 16000, 24000, 48000];
        if cfg!(feature = "enable_qext") {
            rates.push(96000);
        }
        for channels in 3..=8 {
            for &rate in &rates {
                for duration in [1, 2, 4, 8, 16, 24, 32, 40, 48] {
                    let length = rate as usize / 400 * duration;
                    let signal: Vec<f32> = (0..length * channels * 3)
                        .map(|i| {
                            let channel = i % channels;
                            let time = i / channels;
                            (((time * (127 + channel * 73) + channel * 199) % 65536) as i32 - 32768)
                                as f32
                                / 131072.0
                        })
                        .collect();
                    let path = reference_dir.join("surround-mask-pcm.bin");
                    let bytes: Vec<u8> = signal
                        .iter()
                        .flat_map(|sample| sample.to_le_bytes())
                        .collect();
                    fs::write(&path, bytes).unwrap();
                    let args = [
                        "mask".to_string(),
                        rate.to_string(),
                        channels.to_string(),
                        length.to_string(),
                        "3".to_string(),
                        "f32".to_string(),
                        "2051".to_string(),
                        "1".to_string(),
                        "64000".to_string(),
                    ];
                    let output = Command::new(&executable)
                        .args(&args)
                        .arg(&path)
                        .output()
                        .unwrap();
                    assert!(output.status.success());
                    let expected = std::string::String::from_utf8(output.stdout).unwrap();
                    let mut memory = Vec::new();
                    let mut preemphasis = Vec::new();
                    for (frame, reference) in expected.lines().enumerate() {
                        #[cfg(feature = "fixed_point")]
                        let analysis = surround_analysis_fixed::<f32>;
                        #[cfg(not(feature = "fixed_point"))]
                        let analysis = surround_analysis::<f32>;
                        let actual = analysis(
                            &signal[frame * length * channels..],
                            channels,
                            length,
                            rate,
                            &mut memory,
                            &mut preemphasis,
                        );
                        let expected: Vec<u32> = reference
                            .split_whitespace()
                            .map(|bits| u32::from_str_radix(bits, 16).unwrap())
                            .collect();
                        for (band, (&actual, &expected)) in actual.iter().zip(&expected).enumerate()
                        {
                            #[cfg(feature = "fixed_point")]
                            let actual_bits = actual as u32;
                            #[cfg(not(feature = "fixed_point"))]
                            let actual_bits = actual.to_bits();
                            assert_eq!(
                                actual_bits, expected,
                                "channels={channels}, rate={rate}, length={length}, frame={frame}, band={band}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn self_delimited_streams_preserve_multiple_frame_boundaries() {
        for packet in [
            &[0x81, 0xaa, 0xbb][..],
            &[0x82, 1, 0xaa, 0xbb, 0xcc][..],
            &[0x83, 3, 0xaa, 0xbb, 0xcc][..],
            &[0x83, 0xc3, 2, 0, 1, 0xaa, 0xbb, 0xcc, 0, 0][..],
        ] {
            let original = opus_packet_parse_impl(packet, packet.len(), false).unwrap();
            let mut output = [0u8; 100];
            let len = write_stream_packet(packet, true, false, &mut output).unwrap();
            output[len] = 0x80; // A following stream must not enter the current frame.
            let parsed = opus_packet_parse_impl(&output, len + 1, true).unwrap();
            assert_eq!(parsed.packet_offset, len);
            assert_eq!(parsed.frame_count, original.frame_count);
            assert_eq!(
                &parsed.frames[..parsed.frame_count],
                &original.frames[..original.frame_count]
            );
        }
    }

    #[test]
    fn multistream_float_preserves_sub_i16_precision() {
        let pcm: Vec<f32> = (0..960)
            .map(|i| ((i % 41) as f32 - 20.0) / 1_000_000.0)
            .collect();
        let mut mono = opus_encoder_create(48_000, 1, 2051).unwrap();
        opus_encoder_ctl(&mut mono, OpusEncoderCtlRequest::SetBitrate(64_000)).unwrap();
        let mut multi = opus_multistream_encoder_create(48_000, 1, 1, 0, &[0], 2051).unwrap();
        opus_multistream_encoder_ctl(
            &mut multi,
            OpusMultistreamEncoderCtlRequest::SetBitrate(64_000),
        )
        .unwrap();
        let mut expected = [0; 1276];
        let mut actual = [0; 1276];
        let expected_len = opus_encode_float_with_options(
            &mut mono,
            &pcm,
            960,
            &mut expected,
            OpusEncodeOptions::default(),
        )
        .unwrap();
        let actual_len = opus_multistream_encode_float(&mut multi, &pcm, 960, &mut actual).unwrap();
        assert_eq!(&actual[..actual_len], &expected[..expected_len]);
    }

    #[test]
    fn multistream_int24_preserves_low_bits() {
        let pcm: Vec<i32> = (0..960).map(|i| i % 255 - 127).collect();
        let mut mono = opus_encoder_create(48_000, 1, 2051).unwrap();
        opus_encoder_ctl(&mut mono, OpusEncoderCtlRequest::SetBitrate(64_000)).unwrap();
        let mut multi = opus_multistream_encoder_create(48_000, 1, 1, 0, &[0], 2051).unwrap();
        opus_multistream_encoder_ctl(
            &mut multi,
            OpusMultistreamEncoderCtlRequest::SetBitrate(64_000),
        )
        .unwrap();
        let mut expected = [0; 1276];
        let mut actual = [0; 1276];
        let expected_len = opus_encode24_with_options(
            &mut mono,
            &pcm,
            960,
            &mut expected,
            OpusEncodeOptions::default(),
        )
        .unwrap();
        let actual_len = opus_multistream_encode24(&mut multi, &pcm, 960, &mut actual).unwrap();
        assert_eq!(&actual[..actual_len], &expected[..expected_len]);
    }

    fn layout_from_mapping(
        nb_channels: usize,
        nb_streams: usize,
        nb_coupled_streams: usize,
        mapping_slice: &[u8],
    ) -> ChannelLayout {
        let mut mapping = [u8::MAX; 256];
        let count = mapping_slice.len().min(mapping.len());
        mapping[..count].copy_from_slice(&mapping_slice[..count]);
        ChannelLayout {
            nb_channels,
            nb_streams,
            nb_coupled_streams,
            mapping,
        }
    }

    #[test]
    fn accepts_valid_layout() {
        let layout = layout_from_mapping(3, 2, 1, &[0, 1, 2]);
        assert!(validate_layout(&layout));
    }

    #[test]
    fn rejects_out_of_range_stream_indices() {
        let layout = layout_from_mapping(1, 1, 0, &[1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn rejects_when_stream_count_exceeds_byte_limit() {
        let layout = layout_from_mapping(2, 200, 60, &[0, 1]);
        assert!(!validate_layout(&layout));
    }

    #[test]
    fn iterates_all_channels_for_a_stream() {
        let layout = layout_from_mapping(4, 3, 1, &[0, 0, 1, 2]);

        assert_eq!(get_left_channel(&layout, 0, None), Some(0));
        assert_eq!(get_left_channel(&layout, 0, Some(0)), Some(1));
        assert_eq!(get_left_channel(&layout, 0, Some(1)), None);

        assert_eq!(get_right_channel(&layout, 0, None), Some(2));
        assert_eq!(get_right_channel(&layout, 0, Some(2)), None);

        assert_eq!(get_mono_channel(&layout, 1, None), Some(3));
        assert_eq!(get_mono_channel(&layout, 1, Some(3)), None);
    }

    #[test]
    fn validate_encoder_layout_rejects_missing_channel() {
        let layout = layout_from_mapping(2, 1, 1, &[0, u8::MAX]);
        assert!(!validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_encoder_layout_accepts_complete_mapping() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        assert!(validate_encoder_layout(&layout));
    }

    #[test]
    fn validate_ambisonics_computes_stream_and_coupling_counts() {
        assert_eq!(validate_ambisonics(1), Some((1, 0)));
        assert_eq!(validate_ambisonics(6), Some((5, 1)));
        assert_eq!(validate_ambisonics(7), None);
        assert_eq!(validate_ambisonics(228), None);
    }

    #[test]
    fn rate_allocation_handles_stereo_surround_defaults() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates = [0; 1];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 120_000);
        assert_eq!(rates, [120_000]);
    }

    #[test]
    fn rate_allocation_accounts_for_lfe_stream() {
        let layout = layout_from_mapping(6, 4, 2, &[0, 4, 1, 2, 3, 5]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Surround,
            256_000,
            Some(3),
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 255_995);
        assert_eq!(rates, [95_120, 95_120, 57_560, 8_195]);
    }

    #[test]
    fn rate_allocation_splits_evenly_for_ambisonics() {
        let layout = layout_from_mapping(4, 4, 0, &[0, 1, 2, 3]);
        let mut rates = [0; 4];

        let sum = rate_allocation(
            &layout,
            MappingType::Ambisonics,
            OPUS_AUTO,
            None,
            960,
            48_000,
            &mut rates,
        )
        .expect("allocation");

        assert_eq!(sum, 264_000);
        assert_eq!(rates, [66_000; 4]);
    }

    #[test]
    fn rate_allocation_rejects_insufficient_rate_storage() {
        let layout = layout_from_mapping(2, 1, 1, &[0, 1]);
        let mut rates: [i32; 0] = [];

        assert!(
            rate_allocation(
                &layout,
                MappingType::Surround,
                64_000,
                None,
                960,
                48_000,
                &mut rates
            )
            .is_none()
        );
    }

    #[test]
    fn multistream_decoder_size_matches_aligned_components() {
        let coupled = opus_decoder_get_size(2).expect("coupled size");
        let mono = opus_decoder_get_size(1).expect("mono size");
        let expected =
            align(core::mem::size_of::<OpusMsDecoderLayout>()) + align(coupled) + align(mono);
        let reported = opus_multistream_decoder_get_size(2, 1).expect("reported size");

        assert_eq!(reported, expected);
        assert!(opus_multistream_decoder_get_size(0, 0).is_none());
        assert!(opus_multistream_decoder_get_size(2, 3).is_none());
    }

    #[test]
    fn multistream_decoder_creation_validates_arguments() {
        let mapping = [0u8, 1];
        let err = opus_multistream_decoder_create(48_000, 0, 1, 0, &mapping).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::BadArgument);

        let short_map = [0u8];
        let err = opus_multistream_decoder_create(48_000, 2, 1, 1, &short_map).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::BadArgument);
    }

    #[test]
    fn multistream_decoder_ctl_round_trips_gain_and_complexity() {
        let mapping = [0u8, 1];
        let mut decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("decoder");

        opus_multistream_decoder_ctl(&mut decoder, OpusMultistreamDecoderCtlRequest::SetGain(-12))
            .unwrap();

        let mut gain = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetGain(&mut gain),
        )
        .unwrap();
        assert_eq!(gain, -12);

        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::SetComplexity(5),
        )
        .unwrap();
        let mut complexity = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetComplexity(&mut complexity),
        )
        .unwrap();
        assert_eq!(complexity, 5);

        let mut fs = 0;
        opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::GetSampleRate(&mut fs),
        )
        .unwrap();
        assert_eq!(fs, 48_000);

        let err = opus_multistream_decoder_ctl(
            &mut decoder,
            OpusMultistreamDecoderCtlRequest::SetComplexity(11),
        )
        .unwrap_err();
        assert_eq!(
            err,
            OpusMultistreamDecoderError::DecoderCtl(OpusDecoderCtlError::BadArgument)
        );
    }

    #[test]
    fn multistream_decoder_ignore_extensions_reaches_every_stream() {
        let mapping = [0u8, 1, 2];
        let mut decoder = opus_multistream_decoder_create(48_000, 3, 2, 1, &mapping).unwrap();
        for enabled in [false, true, false] {
            opus_multistream_decoder_ctl(
                &mut decoder,
                OpusMultistreamDecoderCtlRequest::SetIgnoreExtensions(enabled),
            )
            .unwrap();
            let mut actual = !enabled;
            opus_multistream_decoder_ctl(
                &mut decoder,
                OpusMultistreamDecoderCtlRequest::GetIgnoreExtensions(&mut actual),
            )
            .unwrap();
            assert_eq!(actual, enabled);
            for stream in &mut decoder.decoders {
                let mut actual = !enabled;
                opus_decoder_ctl(
                    stream,
                    OpusDecoderCtlRequest::GetIgnoreExtensions(&mut actual),
                )
                .unwrap();
                assert_eq!(actual, enabled);
            }
        }
    }

    #[test]
    fn packet_validation_accepts_self_delimited_streams() {
        // First stream is self-delimited (size=1), second stream is the tail packet.
        let packet = [0x00, 0x01, 0xAA, 0x00, 0xBB];
        let samples =
            opus_multistream_packet_validate(&packet, packet.len(), 2, 48_000).expect("packet");
        assert_eq!(samples, 480);
    }

    #[test]
    fn packet_validation_rejects_mismatched_sample_counts() {
        // Second stream advertises a 60 ms frame, which does not match the first stream.
        let packet = [0x00, 0x01, 0xAA, 0x18, 0xBB];
        let err = opus_multistream_packet_validate(&packet, packet.len(), 2, 48_000).unwrap_err();
        assert_eq!(err, OpusMultistreamDecoderError::InvalidPacket);
    }

    #[test]
    fn decode_accepts_valid_single_stream_packet() {
        let mapping = [0u8, 1];
        let mut decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("decoder");
        let packet = [0x00, 0xAA];
        let mut pcm = vec![0i16; 2 * 960];

        let decoded =
            opus_multistream_decode(&mut decoder, &packet, packet.len(), &mut pcm, 960, false)
                .expect("decode");
        assert!(decoded > 0);
    }

    #[test]
    fn multistream_decode_matches_single_stream_decoder_for_stereo() {
        let mapping = [0u8, 1];
        let mut encoder = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 1500];
        let len = opus_encode(&mut encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 2, 1, 1, &mapping).expect("ms decoder");
        let mut single = opus_decoder_create(48_000, 2).expect("single decoder");

        let mut ms_out = vec![0i16; 2 * 960];
        let mut single_out = vec![0i16; 2 * 960];

        let ms_decoded =
            opus_multistream_decode(&mut ms_decoder, &packet, len, &mut ms_out, 960, false)
                .expect("ms decode");
        let single_decoded = opus_decode(
            &mut single,
            Some(&packet[..len]),
            len,
            &mut single_out,
            960,
            false,
        )
        .expect("single decode");
        assert_eq!(ms_decoded, single_decoded);
        assert_eq!(ms_out[..ms_decoded * 2], single_out[..single_decoded * 2]);
    }

    #[test]
    fn multistream_decode_routes_two_coupled_streams() {
        let mapping = [0u8, 1, 2, 3];
        let mut ms_encoder =
            opus_multistream_encoder_create(48_000, 4, 2, 2, &mapping, 2048).expect("ms encoder");
        let pcm_in = vec![0i16; 4 * 960];
        let mut packet = vec![0u8; 4000];
        let len =
            opus_multistream_encode(&mut ms_encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 4, 2, 2, &mapping).expect("ms decoder");

        let mut ms_out = vec![0i16; 4 * 960];
        let decoded =
            opus_multistream_decode(&mut ms_decoder, &packet, len, &mut ms_out, 960, false)
                .expect("ms decode");
        assert!(decoded > 0);

        let mut offset = 0usize;
        let mut remaining = len;
        let mut expected = vec![0i16; 4 * decoded];
        for stream in 0..2 {
            let self_delimited = stream == 0;
            let parsed = opus_packet_parse_impl(&packet[offset..], remaining, self_delimited)
                .expect("parse");
            let packet_offset = parsed.packet_offset;
            let mut dec = opus_decoder_create(48_000, 2).expect("decoder");
            let mut stream_out = vec![0i16; 2 * decoded];
            let stream_packet = if self_delimited {
                let toc = packet[offset];
                let payload_start = offset + parsed.payload_offset;
                let payload_end = offset + parsed.packet_offset;
                let mut reconstructed = vec![0u8; 1 + (payload_end - payload_start)];
                reconstructed[0] = toc;
                reconstructed[1..].copy_from_slice(&packet[payload_start..payload_end]);
                reconstructed
            } else {
                packet[offset..offset + packet_offset].to_vec()
            };

            let stream_decoded = opus_decode(
                &mut dec,
                Some(&stream_packet),
                stream_packet.len(),
                &mut stream_out,
                decoded,
                false,
            )
            .expect("decode");
            assert_eq!(stream_decoded, decoded);

            for i in 0..decoded {
                expected[i * 4 + stream * 2] = stream_out[i * 2];
                expected[i * 4 + stream * 2 + 1] = stream_out[i * 2 + 1];
            }

            offset += packet_offset;
            remaining -= packet_offset;
        }

        assert_eq!(ms_out[..decoded * 4], expected[..decoded * 4]);
    }

    #[test]
    fn multistream_decode_zero_fills_muted_channel() {
        let mapping = [0u8, 1, u8::MAX];
        let mut encoder = opus_encoder_create(48_000, 2, 2048).expect("encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 1500];
        let len = opus_encode(&mut encoder, &pcm_in, 960, &mut packet).expect("encode");

        let mut ms_decoder =
            opus_multistream_decoder_create(48_000, 3, 1, 1, &mapping).expect("ms decoder");
        let mut out = vec![0i16; 3 * 960];
        let decoded = opus_multistream_decode(&mut ms_decoder, &packet, len, &mut out, 960, false)
            .expect("decode");

        for i in 0..decoded {
            assert_eq!(out[i * 3 + 2], 0);
        }
    }

    #[test]
    fn multistream_encode_respects_minimum_packet_size() {
        let mapping = [0u8, 1];
        let mut ms_encoder =
            opus_multistream_encoder_create(48_000, 2, 2, 0, &mapping, 2048).expect("ms encoder");
        let pcm_in = vec![0i16; 2 * 960];
        let mut packet = vec![0u8; 2];
        let err = opus_multistream_encode(&mut ms_encoder, &pcm_in, 960, &mut packet).unwrap_err();
        assert_eq!(err, OpusMultistreamEncoderError::BufferTooSmall);
    }

    #[test]
    fn multistream_encoder_ctl_round_trips_lsb_and_prediction() {
        let mapping = [0u8, 1];
        let mut encoder =
            opus_multistream_encoder_create(48_000, 2, 2, 0, &mapping, 2048).expect("ms encoder");

        let mut lsb = 0;
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::GetLsbDepth(&mut lsb),
        )
        .unwrap();
        assert!(lsb >= 16);

        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetLsbDepth(12),
        )
        .unwrap();
        let mut updated = 0;
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::GetLsbDepth(&mut updated),
        )
        .unwrap();
        assert_eq!(updated, 12);

        let mut pred = false;
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(&mut pred),
        )
        .unwrap();
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetPredictionDisabled(!pred),
        )
        .unwrap();
        let mut pred_after = false;
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::GetPredictionDisabled(&mut pred_after),
        )
        .unwrap();
        assert_eq!(pred_after, !pred);

        {
            let stream =
                opus_multistream_encoder_get_encoder_state(&mut encoder, 1).expect("stream");
            let mut stream_lsb = 0;
            opus_encoder_ctl(stream, OpusEncoderCtlRequest::GetLsbDepth(&mut stream_lsb)).unwrap();
            assert_eq!(stream_lsb, 12);
        }

        assert_eq!(
            opus_multistream_encoder_get_encoder_state(&mut encoder, 2).unwrap_err(),
            OpusMultistreamEncoderError::BadArgument
        );
    }

    #[cfg(feature = "enable_qext")]
    #[test]
    fn multistream_encoder_qext_control_reaches_every_stream() {
        let mapping = [0u8, 1, 2];
        let mut encoder = opus_multistream_encoder_create(48_000, 3, 2, 1, &mapping, 2049).unwrap();
        for enabled in [false, true, false] {
            opus_multistream_encoder_ctl(
                &mut encoder,
                OpusMultistreamEncoderCtlRequest::SetQext(enabled),
            )
            .unwrap();
            let mut actual = !enabled;
            opus_multistream_encoder_ctl(
                &mut encoder,
                OpusMultistreamEncoderCtlRequest::GetQext(&mut actual),
            )
            .unwrap();
            assert_eq!(actual, enabled);
            for stream in &mut encoder.encoders {
                let mut actual = !enabled;
                opus_encoder_ctl(stream, OpusEncoderCtlRequest::GetQext(&mut actual)).unwrap();
                assert_eq!(actual, enabled);
            }
        }
    }

    #[cfg(feature = "dred")]
    #[test]
    fn multistream_encoder_ctl_round_trips_dred_duration() {
        let mapping = [0u8, 1];
        let mut encoder =
            opus_multistream_encoder_create(48_000, 2, 2, 0, &mapping, 2048).expect("ms encoder");

        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::SetDredDuration(8),
        )
        .unwrap();
        let mut duration = 0;
        opus_multistream_encoder_ctl(
            &mut encoder,
            OpusMultistreamEncoderCtlRequest::GetDredDuration(&mut duration),
        )
        .unwrap();
        assert_eq!(duration, 8);
    }

    #[test]
    fn surround_encoder_layout_matches_vorbis_mapping() {
        let (encoder, layout) =
            opus_multistream_surround_encoder_create(48_000, 6, 1, 2048).expect("encoder");

        assert_eq!(layout.streams, 4);
        assert_eq!(layout.coupled_streams, 2);
        assert_eq!(layout.mapping, vec![0, 4, 1, 2, 3, 5]);
        assert_eq!(encoder.layout.nb_channels, 6);
        assert_eq!(encoder.layout.nb_streams, 4);
        assert_eq!(encoder.layout.nb_coupled_streams, 2);

        // LFE is the last stream for 5.1+ layouts.
        assert_eq!(encoder.lfe_stream, Some(3));
        assert_eq!(encoder.mapping_type, MappingType::Surround);
    }

    #[test]
    fn surround_encoder_ambisonics_layout_matches_reference_ordering() {
        let (_encoder, layout) =
            opus_multistream_surround_encoder_create(48_000, 6, 2, 2048).expect("encoder");
        assert_eq!(layout.streams, 5);
        assert_eq!(layout.coupled_streams, 1);
        assert_eq!(layout.mapping, vec![2, 3, 4, 5, 0, 1]);
    }

    #[test]
    fn surround_encoder_get_size_matches_reference_overhead() {
        let size = opus_multistream_surround_encoder_get_size(6, 1).expect("size");
        let base = opus_multistream_encoder_get_size(4, 2).expect("base");
        let overhead = 6
            * (120 * core::mem::size_of::<crate::celt::OpusVal32>()
                + core::mem::size_of::<crate::celt::OpusVal32>());
        assert_eq!(size, base + overhead);
    }
}
