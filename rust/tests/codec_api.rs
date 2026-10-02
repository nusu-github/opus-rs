use opus_rs::{
    Application, Bitrate, Channels, Decoder, Encoder, FrameDuration, OpusDecodeError,
    OpusEncodeError, Signal,
};

const FRAME_SIZE: usize = 960;
const SAMPLE_RATE: u32 = 48_000;
const MAX_FRAME_SIZE: usize = 6 * 960;
const MAX_PACKET_SIZE: usize = 3 * 1276;

#[cfg(feature = "enable_qext")]
#[test]
fn high_level_quality_extension_configuration_at_96khz() {
    let mut encoder = Encoder::builder(96_000, Channels::Stereo, Application::Audio)
        .qext(true)
        .build()
        .unwrap();
    assert!(encoder.qext().unwrap());
    encoder.set_qext(false).unwrap();
    assert!(!encoder.qext().unwrap());
    let mut decoder = Decoder::builder(96_000, Channels::Stereo)
        .ignore_extensions(true)
        .build()
        .unwrap();
    assert!(decoder.ignore_extensions().unwrap());
    decoder.set_ignore_extensions(false).unwrap();
    assert!(!decoder.ignore_extensions().unwrap());
}

#[cfg(feature = "enable_qext")]
#[test]
fn malformed_quality_extension_payloads_do_not_panic() {
    let mut random = 0x254e97a1u32;
    for rate in [48_000, 96_000] {
        for channels in [Channels::Mono, Channels::Stereo] {
            let frame_size = rate as usize / 400;
            let pcm: Vec<i16> = (0..frame_size * channels.count())
                .map(|i| ((i * 997 % 30000) as i16) - 15000)
                .collect();
            let mut encoder = Encoder::new(rate, channels, Application::LowDelay).unwrap();
            let mut base = vec![0; 1024];
            let base_length = encoder.encode(&pcm, &mut base).unwrap();
            assert_eq!(base[0] & 3, 0);
            for length in [0, 1, 2, 3, 7, 31, 127] {
                for trial in 0..16 {
                    let mut packet = vec![(base[0] & !3) | 3, 0x41, (length + 1) as u8];
                    packet.extend_from_slice(&base[1..base_length]);
                    packet.push(124 << 1);
                    for _ in 0..length {
                        random ^= random << 13;
                        random ^= random >> 17;
                        random ^= random << 5;
                        packet.push(if trial == 0 {
                            0
                        } else if trial == 1 {
                            255
                        } else {
                            random as u8
                        });
                    }
                    let mut decoder = Decoder::new(rate, channels).unwrap();
                    let mut output = vec![0; pcm.len()];
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        decoder.decode(&packet, &mut output, false)
                    }));
                    assert!(
                        result.is_ok(),
                        "QEXT decoder panicked at {rate} Hz, {channels:?}, payload {length}, trial {trial}"
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_packet_payloads_do_not_panic() {
    // Exercise every TOC mode and both encoded channel layouts with sparse,
    // saturated, and pseudorandom payloads. An error is acceptable; a panic is
    // not, because these bytes can come directly from an untrusted transport.
    let mut random = 0x8d37ab91u32;
    for toc in 0u8..=255 {
        for length in [1, 2, 17, 127] {
            let mut packet = vec![toc];
            for _ in 1..length {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                packet.push(random as u8);
            }
            let mut decoder = Decoder::new(8000, Channels::Mono).unwrap();
            let mut output = [0i16; 960];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                decoder.decode(&packet, &mut output, false)
            }));
            assert!(result.is_ok(), "decoder panicked for packet {packet:02x?}");
        }
    }
}

#[test]
fn high_level_round_trip_and_configuration() {
    let mut encoder = Encoder::builder(SAMPLE_RATE, Channels::Stereo, Application::Audio)
        .bitrate(Bitrate::Bits(64_000))
        .complexity(7)
        .vbr(false)
        .signal(Signal::Music)
        .frame_duration(FrameDuration::Ms20)
        .build()
        .expect("encoder");

    assert_eq!(encoder.bitrate().expect("bitrate"), Bitrate::Bits(64_000));
    assert_eq!(encoder.complexity().expect("complexity"), 7);
    assert!(!encoder.vbr().expect("vbr"));
    assert_eq!(encoder.signal().expect("signal"), Signal::Music);
    assert_eq!(
        encoder.frame_duration().expect("frame duration"),
        FrameDuration::Ms20
    );

    let mut decoder = Decoder::builder(SAMPLE_RATE, Channels::Stereo)
        .gain(256)
        .complexity(6)
        .build()
        .expect("decoder");

    assert_eq!(decoder.gain().expect("gain"), 256);
    assert_eq!(decoder.complexity().expect("complexity"), 6);

    let channels = Channels::Stereo.count();
    let mut input = vec![0i16; FRAME_SIZE * channels];
    for (idx, sample) in input.iter_mut().enumerate() {
        *sample = ((idx as i32 * 31) % i16::MAX as i32) as i16;
    }

    let mut packet = vec![0u8; MAX_PACKET_SIZE];
    let packet_len = encoder.encode(&input, &mut packet).expect("encode");
    assert!(packet_len > 0);
    assert_eq!(
        decoder
            .packet_samples(&packet[..packet_len])
            .expect("packet samples"),
        FRAME_SIZE
    );

    let mut output = vec![0i16; MAX_FRAME_SIZE * channels];
    let decoded = decoder
        .decode(&packet[..packet_len], &mut output, false)
        .expect("decode");
    assert_eq!(decoded, FRAME_SIZE);
    assert!(
        decoder
            .last_packet_duration()
            .expect("last packet duration")
            > 0
    );
    assert!(decoder.final_range().expect("final range") > 0);
}

#[test]
fn high_level_encode_rejects_partial_frame_slices() {
    let mut encoder =
        Encoder::new(SAMPLE_RATE, Channels::Stereo, Application::Audio).expect("encoder");
    let pcm = [0i16; FRAME_SIZE * 2 - 1];
    let mut packet = [0u8; MAX_PACKET_SIZE];

    let err = encoder
        .encode(&pcm, &mut packet)
        .expect_err("partial stereo frame should fail");
    assert_eq!(err, OpusEncodeError::BadArgument);
}

#[test]
fn high_level_decode_rejects_partial_frame_buffers() {
    let mut decoder = Decoder::new(SAMPLE_RATE, Channels::Stereo).expect("decoder");
    let packet = [0u8; 1];
    let mut pcm = [0i16; MAX_FRAME_SIZE * 2 - 1];

    let err = decoder
        .decode(&packet, &mut pcm, false)
        .expect_err("partial stereo buffer should fail");
    assert_eq!(err, OpusDecodeError::BadArgument);
}
