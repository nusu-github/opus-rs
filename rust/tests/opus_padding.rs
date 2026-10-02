//! Port of the padding overflow regression test from
//! `opus-c/tests/test_opus_padding.c`, adapted to the current Rust decoder API.

use opus_rs::{Channels, Decoder, OpusDecodeError, opus_get_version_string};

// 16,909,318 bytes mirrors the pathological packet length from the C test.
const PACKET_SIZE: usize = 16_909_318;
// A full 120 ms mono output buffer lets the complete decoder inspect padding.
const OUTPUT_SAMPLES: usize = 5760;

#[test]
fn padding_overflow_packet_is_rejected() {
    let version = opus_get_version_string();
    assert!(
        !version.is_empty(),
        "version string should be available for diagnostics"
    );

    let mut packet = vec![0xffu8; PACKET_SIZE];
    packet[1] = 0x41;
    packet[PACKET_SIZE - 1] = 0x0b;

    let mut decoder = Decoder::new(48_000, Channels::Mono).expect("decoder init");
    let mut output = vec![0i16; OUTPUT_SAMPLES];

    let err = decoder
        .decode(&packet, &mut output, false)
        .expect_err("invalid padded packet must be rejected");
    assert!(
        matches!(err, OpusDecodeError::InvalidPacket),
        "the complete packet parser must reject the padding overflow"
    );
}
