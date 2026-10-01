//! Exact FFT/MDCT checks against independent pinned C profiles.
use crate::celt::KissFftCpx;
use crate::celt::{MdctLookup, clt_mdct_backward, clt_mdct_forward};
use alloc::{vec, vec::Vec};
struct Words<'a>(&'a [u8]);
impl Words<'_> {
    fn next(&mut self) -> u32 {
        let (word, rest) = self.0.split_at(4);
        self.0 = rest;
        u32::from_le_bytes(word.try_into().unwrap())
    }
}
fn sample(index: usize, pattern: u32) -> f32 {
    match pattern {
        0 => 0.0,
        1 => {
            if index == 3 {
                2.0
            } else {
                0.0
            }
        }
        2 => {
            ((index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                & 16_777_215) as f32
                * (1.0 / 16_777_216.0)
                - 0.5
        }
        _ => {
            if index & 1 != 0 {
                -8.0
            } else {
                7.9999995
            }
        }
    }
}
#[test]
fn float_fft_matches_pinned_c_transform_vectors() {
    for &(base_size, data) in fft_fixtures() {
        let mut words = Words(data);
        let lookup = MdctLookup::new(base_size * 2, 3);
        let mut cases = 0;
        while !words.0.is_empty() {
            let shift = words.next();
            let pattern = words.next();
            let n = words.next() as usize;
            let state = lookup.forward_plan(shift as usize);
            let input: Vec<_> = (0..n)
                .map(|i| {
                    KissFftCpx::new(
                        sample(2 * i, pattern) * (1.0 / 16.0),
                        sample(2 * i + 1, pattern) * (1.0 / 16.0),
                    )
                })
                .collect();
            let mut forward = vec![KissFftCpx::default(); n];
            let mut inverse = forward.clone();
            state.fft(&input, &mut forward);
            state.ifft(&input, &mut inverse);
            for i in 0..n {
                for (kind, value) in [
                    ("forward real", forward[i].r),
                    ("forward imaginary", forward[i].i),
                    ("inverse real", inverse[i].r),
                    ("inverse imaginary", inverse[i].i),
                ] {
                    assert_eq!(
                        value.to_bits(),
                        words.next(),
                        "{kind} shift={shift} pattern={pattern} index={i}"
                    );
                }
            }
            cases += 1;
        }
        assert_eq!(cases, 16);
    }
}
#[test]
fn float_mdct_matches_pinned_c_transform_vectors() {
    for &(base_size, data) in mdct_fixtures() {
        let mut words = Words(data);
        let overlap = words.next() as usize;
        let window: Vec<_> = (0..overlap).map(|_| f32::from_bits(words.next())).collect();
        let lookup = MdctLookup::new(base_size * 2, 3);
        let mut cases = 0;
        while !words.0.is_empty() {
            let shift = words.next() as usize;
            let stride = words.next() as usize;
            let pattern = words.next();
            let forward_len = words.next() as usize;
            let backward_len = words.next() as usize;
            let n = base_size >> shift;
            let input: Vec<_> = (0..n + overlap).map(|i| sample(i, pattern)).collect();
            let frequency: Vec<_> = (0..n * stride).map(|i| sample(i, pattern)).collect();
            let mut spectrum = vec![0x13579 as f32; forward_len];
            let mut time: Vec<_> = (0..backward_len)
                .map(|i| sample(i + 11, pattern) * (1.0 / 16.0))
                .collect();
            clt_mdct_forward(
                &lookup,
                &input,
                &mut spectrum,
                &window,
                overlap,
                shift,
                stride,
            );
            clt_mdct_backward(
                &lookup, &frequency, &mut time, &window, overlap, shift, stride,
            );
            for (kind, values) in [("forward", &spectrum), ("inverse", &time)] {
                for (i, &value) in values.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        words.next(),
                        "{kind} shift={shift} stride={stride} pattern={pattern} index={i}"
                    );
                }
            }
            cases += 1;
        }
        assert_eq!(cases, 32);
    }
}
fn fft_fixtures() -> &'static [(usize, &'static [u8])] {
    #[cfg(all(not(feature = "enable_qext"), not(feature = "pfa")))]
    {
        &[(
            960,
            include_bytes!("../../tests/fixtures/reference/float-48000-fft.bin"),
        )]
    }

    #[cfg(all(not(feature = "enable_qext"), feature = "pfa"))]
    {
        &[(
            960,
            include_bytes!("../../tests/fixtures/reference/pfa-fft.bin"),
        )]
    }
    #[cfg(all(feature = "enable_qext", not(feature = "pfa")))]
    {
        &[
            (
                960,
                include_bytes!("../../tests/fixtures/reference/qext-48000-fft.bin"),
            ),
            (
                1920,
                include_bytes!("../../tests/fixtures/reference/qext-96000-fft.bin"),
            ),
        ]
    }
    #[cfg(all(feature = "enable_qext", feature = "pfa"))]
    {
        &[
            (
                960,
                include_bytes!("../../tests/fixtures/reference/qext-pfa-48000-fft.bin"),
            ),
            (
                1920,
                include_bytes!("../../tests/fixtures/reference/qext-pfa-96000-fft.bin"),
            ),
        ]
    }
}
fn mdct_fixtures() -> &'static [(usize, &'static [u8])] {
    #[cfg(all(not(feature = "enable_qext"), not(feature = "pfa")))]
    {
        &[(
            960,
            include_bytes!("../../tests/fixtures/reference/float-48000-mdct.bin"),
        )]
    }

    #[cfg(all(not(feature = "enable_qext"), feature = "pfa"))]
    {
        &[(
            960,
            include_bytes!("../../tests/fixtures/reference/pfa-mdct.bin"),
        )]
    }
    #[cfg(all(feature = "enable_qext", not(feature = "pfa")))]
    {
        &[
            (
                960,
                include_bytes!("../../tests/fixtures/reference/qext-48000-mdct.bin"),
            ),
            (
                1920,
                include_bytes!("../../tests/fixtures/reference/qext-96000-mdct.bin"),
            ),
        ]
    }
    #[cfg(all(feature = "enable_qext", feature = "pfa"))]
    {
        &[
            (
                960,
                include_bytes!("../../tests/fixtures/reference/qext-pfa-48000-mdct.bin"),
            ),
            (
                1920,
                include_bytes!("../../tests/fixtures/reference/qext-pfa-96000-mdct.bin"),
            ),
        ]
    }
}
