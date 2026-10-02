//! Opus Custom CELT modes with explicit, safe mode lifetimes.
//!
//! Custom packets require the same sample rate and frame size at both ends.
//! Keep the owned [`Mode`] and its [`ModeView`] alive while using an encoder or
//! decoder. This separates mode tables from mutable codec state without raw
//! pointers or self-referential allocations.
//!
//! ```
//! use opus_rs::custom::Mode;
//! let mode = Mode::new(48_000, 512)?;
//! let view = mode.view();
//! let mut encoder = view.encoder(1)?;
//! let mut decoder = view.decoder(1)?;
//! let mut packet = [0u8; 80];
//! let length = encoder.encode(&[0i16; 512], &mut packet)?;
//! let mut output = [0i16; 512];
//! decoder.decode(Some(&packet[..length]), &mut output)?;
//! # Ok::<(), opus_rs::custom::Error>(())
//! ```

use crate::celt::*;

/// Invalid configuration, buffer, or custom packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidMode,
    UnsupportedTransform,
    InvalidChannels,
    InvalidArgument,
    InvalidPacket,
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidMode => "unsupported custom sample rate or frame size",
            Self::UnsupportedTransform => "custom transform has an unsupported prime factor",
            Self::InvalidChannels => "custom CELT requires one or two channels",
            Self::InvalidArgument => "invalid custom codec argument or buffer",
            Self::InvalidPacket => "invalid custom CELT packet",
        })
    }
}
impl core::error::Error for Error {}

/// Owns the immutable tables of a custom CELT mode.
#[derive(Debug, Clone)]
pub struct Mode {
    inner: OwnedOpusCustomMode,
    frame_size: usize,
}

impl Mode {
    /// Constructs a CELT mode for an 8–96 kHz sample rate and a 40–1024
    /// sample frame (up to 2048 samples with `enable_qext`).
    /// CELT additionally constrains the duration, short blocks,
    /// band widths and transform factorization; invalid modes return an error.
    pub fn new(sample_rate: u32, frame_size: usize) -> Result<Self, Error> {
        let rate = i32::try_from(sample_rate).map_err(|_| Error::InvalidMode)?;
        let inner = opus_custom_mode_create(rate, frame_size).map_err(|error| match error {
            ModeError::UnsupportedTransform => Error::UnsupportedTransform,
            _ => Error::InvalidMode,
        })?;
        Ok(Self { inner, frame_size })
    }

    pub fn view(&self) -> ModeView<'_> {
        ModeView {
            inner: self.inner.mode(),
            frame_size: self.frame_size,
        }
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
    pub fn sample_rate(&self) -> u32 {
        self.inner.mode().sample_rate as u32
    }
}

/// Borrows mode tables and creates codec states that cannot outlive them.
pub struct ModeView<'mode> {
    inner: OpusCustomMode<'mode>,
    frame_size: usize,
}

impl ModeView<'_> {
    pub fn encoder(&self, channels: usize) -> Result<Encoder<'_>, Error> {
        if !(1..=2).contains(&channels) {
            return Err(Error::InvalidChannels);
        }
        let inner = opus_custom_encoder_create(&self.inner, self.inner.sample_rate, channels, 0)
            .map_err(|_| Error::InvalidArgument)?;
        Ok(Encoder {
            inner,
            frame_size: self.frame_size,
            channels,
        })
    }

    pub fn decoder(&self, channels: usize) -> Result<Decoder<'_>, Error> {
        if !(1..=2).contains(&channels) {
            return Err(Error::InvalidChannels);
        }
        // C accepts some modes whose frames leave no room for the decoder's
        // pitch history. Its postfilter then reads before its allocation.
        // Retain mode construction, but reject an invalid decoder state.
        if self.frame_size > crate::celt::mode_decode_buffer_size(&self.inner) / 2 {
            return Err(Error::InvalidMode);
        }
        let inner = opus_custom_decoder_create(&self.inner, channels)
            .map_err(|_| Error::InvalidArgument)?;
        Ok(Decoder {
            inner,
            frame_size: self.frame_size,
        })
    }
}

/// A custom CELT encoder borrowing its mode view.
pub struct Encoder<'mode> {
    inner: OwnedCeltEncoder<'mode>,
    frame_size: usize,
    channels: usize,
}

impl Encoder<'_> {
    fn validate(&self, samples: usize, bytes: usize) -> Result<(), Error> {
        if samples != self.frame_size * self.channels || bytes < 2 {
            Err(Error::InvalidArgument)
        } else {
            Ok(())
        }
    }

    pub fn encode(&mut self, pcm: &[i16], packet: &mut [u8]) -> Result<usize, Error> {
        self.validate(pcm.len(), packet.len())?;
        let capacity = packet.len();
        opus_custom_encode(&mut self.inner, pcm, self.frame_size, packet, capacity)
            .map_err(|_| Error::InvalidArgument)
    }

    pub fn encode_float(&mut self, pcm: &[f32], packet: &mut [u8]) -> Result<usize, Error> {
        self.validate(pcm.len(), packet.len())?;
        let capacity = packet.len();
        opus_custom_encode_float(&mut self.inner, pcm, self.frame_size, packet, capacity)
            .map_err(|_| Error::InvalidArgument)
    }

    pub fn encode_24(&mut self, pcm: &[i32], packet: &mut [u8]) -> Result<usize, Error> {
        self.validate(pcm.len(), packet.len())?;
        let capacity = packet.len();
        opus_custom_encode24(&mut self.inner, pcm, self.frame_size, packet, capacity)
            .map_err(|_| Error::InvalidArgument)
    }

    /// Enables the scalable QEXT extension when the selected mode supports it.
    #[cfg(feature = "enable_qext")]
    pub fn set_qext(&mut self, enabled: bool) -> Result<(), Error> {
        opus_custom_encoder_ctl(&mut self.inner, EncoderCtlRequest::SetQext(enabled))
            .map_err(|_| Error::InvalidArgument)
    }

    /// Reports whether the scalable QEXT extension is enabled.
    #[cfg(feature = "enable_qext")]
    pub fn qext(&mut self) -> Result<bool, Error> {
        let mut enabled = false;
        opus_custom_encoder_ctl(&mut self.inner, EncoderCtlRequest::GetQext(&mut enabled))
            .map_err(|_| Error::InvalidArgument)?;
        Ok(enabled)
    }

    pub fn set_complexity(&mut self, value: u8) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetComplexity(i32::from(value)),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_bitrate(&mut self, value: u32) -> Result<(), Error> {
        let bitrate = i32::try_from(value).map_err(|_| Error::InvalidArgument)?;
        opus_custom_encoder_ctl(&mut self.inner, EncoderCtlRequest::SetBitrate(bitrate))
            .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_vbr(&mut self, enabled: bool) -> Result<(), Error> {
        opus_custom_encoder_ctl(&mut self.inner, EncoderCtlRequest::SetVbr(enabled))
            .map_err(|_| Error::InvalidArgument)
    }

    /// Limits VBR fluctuations to the decoder's buffering budget.
    pub fn set_vbr_constraint(&mut self, enabled: bool) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetVbrConstraint(enabled),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    /// Sets prediction: 0 disables it, 1 permits inter-frame prediction,
    /// and 2 also permits the pitch prefilter.
    pub fn set_prediction(&mut self, value: u8) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetPrediction(i32::from(value)),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_packet_loss_percent(&mut self, value: u8) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetPacketLossPerc(i32::from(value)),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_lsb_depth(&mut self, value: u8) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetLsbDepth(i32::from(value)),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<(), Error> {
        opus_custom_encoder_ctl(
            &mut self.inner,
            EncoderCtlRequest::SetPhaseInversionDisabled(disabled),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn reset(&mut self) {
        let _ = opus_custom_encoder_ctl(&mut self.inner, EncoderCtlRequest::ResetState);
    }

    pub fn final_range(&self) -> u32 {
        self.inner.rng
    }
}

/// A custom CELT decoder. Passing `None` conceals a lost packet.
pub struct Decoder<'mode> {
    inner: OwnedCeltDecoder<'mode>,
    frame_size: usize,
}

impl Decoder<'_> {
    fn validate(&self, packet: Option<&[u8]>, samples: usize) -> Result<(), Error> {
        if samples < self.frame_size * self.inner.channels
            || (packet.is_none_or(|bytes| bytes.len() < 2)
                && self.frame_size > self.inner.history_size() / 2)
        {
            Err(Error::InvalidArgument)
        } else {
            Ok(())
        }
    }

    pub fn decode(&mut self, packet: Option<&[u8]>, pcm: &mut [i16]) -> Result<usize, Error> {
        self.validate(packet, pcm.len())?;
        opus_custom_decode(&mut self.inner, packet, pcm, self.frame_size)
            .map_err(|_| Error::InvalidPacket)
    }

    pub fn decode_float(&mut self, packet: Option<&[u8]>, pcm: &mut [f32]) -> Result<usize, Error> {
        self.validate(packet, pcm.len())?;
        opus_custom_decode_float(&mut self.inner, packet, pcm, self.frame_size)
            .map_err(|_| Error::InvalidPacket)
    }

    pub fn decode_24(&mut self, packet: Option<&[u8]>, pcm: &mut [i32]) -> Result<usize, Error> {
        self.validate(packet, pcm.len())?;
        opus_custom_decode24(&mut self.inner, packet, pcm, self.frame_size)
            .map_err(|_| Error::InvalidPacket)
    }

    pub fn set_complexity(&mut self, value: u8) -> Result<(), Error> {
        opus_custom_decoder_ctl(
            &mut self.inner,
            DecoderCtlRequest::SetComplexity(i32::from(value)),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<(), Error> {
        opus_custom_decoder_ctl(
            &mut self.inner,
            DecoderCtlRequest::SetPhaseInversionDisabled(disabled),
        )
        .map_err(|_| Error::InvalidArgument)
    }

    pub fn reset(&mut self) {
        let _ = opus_custom_decoder_ctl(&mut self.inner, DecoderCtlRequest::ResetState);
    }

    pub fn final_range(&self) -> u32 {
        self.inner.rng
    }
}
