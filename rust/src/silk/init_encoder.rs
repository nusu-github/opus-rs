//! Port of `silk/init_encoder.c`.
//!
//! The reference helper clears the per-channel encoder state, re-primes the
//! adaptive high-pass smoother, and reinitialises the fixed-point VAD before
//! encoding resumes. This translation mirrors that behaviour for the current
//! Rust `EncoderChannelState`.

use crate::silk::encoder::state::ActiveEncoderChannelState as EncoderChannelState;
use crate::silk::errors::SilkError;
use crate::silk::lin2log::lin2log;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

/// Mirrors `silk_init_encoder`.
pub fn init_encoder(state: &mut EncoderChannelState, arch: i32) -> Result<(), SilkError> {
    *state = EncoderChannelState::default();

    let hp_log_q15 = lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    {
        let common = state.common_mut();
        // Initialization precedes sample-rate setup in the reference. Leaving
        // the convenient standalone defaults here skips the first resampler
        // and NSQ reset when the requested rate happens to be 16 kHz.
        common.fs_khz = 0;
        common.api_sample_rate_hz = 0;
        common.prev_api_sample_rate_hz = 0;
        common.max_internal_sample_rate_hz = 0;
        common.min_internal_sample_rate_hz = 0;
        common.desired_internal_sample_rate_hz = 0;
        common.packet_size_ms = 0;
        common.frame_length = 0;
        common.n_frames_per_packet = 0;
        common.nb_subfr = 0;
        common.subfr_length = 0;
        common.ltp_mem_length = 0;
        common.la_pitch = 0;
        common.la_shape = 0;
        common.shape_win_length = 0;
        common.max_pitch_lag = 0;
        common.pitch_lpc_win_length = 0;
        common.predict_lpc_order = 0;
        common.n_channels_api = 0;
        common.n_channels_internal = 0;
        common.nsq_state.prev_gain_q16 = 0;
        common.arch = arch;
        common.variable_hp_smth1_q15 = hp_log_q15;
        common.variable_hp_smth2_q15 = hp_log_q15;
        common.first_frame_after_reset = true;
    }
    state.vad_mut().reset();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::init_encoder;
    use crate::silk::encoder::state::{ActiveEncoderChannelState as EncoderChannelState, VadState};
    use crate::silk::lin2log::lin2log;
    use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

    #[test]
    fn init_encoder_resets_channel_state() {
        let mut state = EncoderChannelState::default();
        {
            let common = state.common_mut();
            common.fs_khz = 24;
            common.variable_hp_smth1_q15 = 0;
            common.variable_hp_smth2_q15 = 0;
            common.first_frame_after_reset = false;
        }
        state.common_mut().input_buf[0] = 123;
        state.vad_mut().counter = -5;

        init_encoder(&mut state, 7).unwrap();

        assert_eq!(state.common().arch, 7);
        assert_eq!(
            state.common().variable_hp_smth1_q15,
            lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8
        );
        assert_eq!(
            state.common().variable_hp_smth2_q15,
            state.common().variable_hp_smth1_q15
        );
        assert_eq!(state.common().fs_khz, 0);
        assert_eq!(state.common().api_sample_rate_hz, 0);
        assert_eq!(state.common().packet_size_ms, 0);
        assert!(state.common().first_frame_after_reset);
        assert_eq!(state.vad(), &VadState::default());
        assert!(state.common().input_buf.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn init_encoder_updates_architecture_flag() {
        let mut state = EncoderChannelState::default();
        init_encoder(&mut state, 3).unwrap();
        assert_eq!(state.common().arch, 3);

        init_encoder(&mut state, 5).unwrap();
        assert_eq!(state.common().arch, 5);
    }
}
