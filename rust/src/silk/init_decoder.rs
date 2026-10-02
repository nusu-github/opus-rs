//! Port of `silk/init_decoder.c`.
//!
//! The reference implementation exposes two helpers:
//! - `silk_reset_decoder` clears per-channel coding state while preserving
//!   bandwidth-extension history, matching the C reset boundary.
//! - `silk_init_decoder` fully reinitialises a channel by zeroing the struct
//!   before delegating to `silk_reset_decoder`.
//!
//! These wrappers keep the Rust decoder state in sync with the behaviour
//! expected by the C API entry points.

use crate::celt::opus_select_arch;
use crate::silk::decoder_state::DecoderState;
use crate::silk::errors::SilkError;

/// Mirrors `silk_reset_decoder`.
pub fn reset_decoder(state: &mut DecoderState) -> Result<(), SilkError> {
    // The C BBWE state precedes SILK_DECODER_STATE_RESET_START. A reset keeps
    // this history, while a full initialization clears it along with the rest
    // of the channel allocation.
    #[cfg(feature = "osce")]
    let osce_bwe = core::mem::take(&mut state.osce_bwe);
    *state = DecoderState::default();
    #[cfg(feature = "osce")]
    {
        state.osce_bwe = osce_bwe;
    }

    let lpc_order = state.sample_rate.lpc_order;
    state.cng_state.reset(lpc_order);
    state.plc_state.reset(state.sample_rate.frame_length);
    state.arch = opus_select_arch();

    Ok(())
}

/// Mirrors `silk_init_decoder`.
pub fn init_decoder(state: &mut DecoderState) -> Result<(), SilkError> {
    *state = DecoderState::default();
    reset_decoder(state)
}

#[cfg(test)]
mod tests {
    use super::{init_decoder, reset_decoder};
    use crate::celt::opus_select_arch;
    use crate::silk::decoder_state::DecoderState;

    #[test]
    fn reset_decoder_initialises_architecture() {
        let mut state = DecoderState::default();
        reset_decoder(&mut state).unwrap();
        assert_eq!(state.arch, opus_select_arch());
    }

    #[test]
    fn init_decoder_primes_cng_and_plc() {
        let mut state = DecoderState::default();
        init_decoder(&mut state).unwrap();
        assert!(
            state.cng_state.smoothed_nlsf_q15()[0] > 0,
            "CNG reset should prime the smoothed NLSF grid"
        );
        assert_eq!(state.plc_state.prev_gain_q16, [1 << 16; 2]);
    }
}
