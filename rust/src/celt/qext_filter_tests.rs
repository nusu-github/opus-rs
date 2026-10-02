//! Pinned C arithmetic for QEXT's higher precision preemphasis.
use alloc::{vec, vec::Vec};

#[test]
fn qext_preemphasis_matches_pinned_c_vectors() {
    let mut words = include_bytes!("../../tests/fixtures/reference/fixed-qext-preemphasis.bin")
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()));
    let mut cases = 0;
    while let Some(rate) = words.next() {
        let upsample = words.next().unwrap() as usize;
        let channels = words.next().unwrap() as usize;
        let input: Vec<_> = (0..384)
            .map(|i| (((i * 631 + 811) % 60001 - 30000) as f32) / 32768.0)
            .collect();
        let coef = super::modes::compute_preemphasis(rate);
        let mut output = vec![0; 192];
        let mut memory = -123456;
        super::celt_encoder::celt_preemphasis_fixed(
            &input,
            &mut output,
            192,
            channels,
            upsample,
            &coef,
            &mut memory,
            false,
        );
        for (i, sample) in output.into_iter().enumerate() {
            assert_eq!(
                sample,
                words.next().unwrap(),
                "rate{rate} up{upsample} channels{channels} sample{i}"
            );
        }
        assert_eq!(
            memory,
            words.next().unwrap(),
            "final state rate{rate} up{upsample} channels{channels}"
        );
        cases += 1;
    }
    assert_eq!(cases, 42);
}
