//! Checked C-oracle traces verify the actual entropy coder used by the codec.

use opus_rs::entropy::{RangeDecoder, RangeEncoder};
use std::fmt::Write;

const ICDF: [u8; 4] = [192, 128, 64, 0];
const ICDF16: [u16; 4] = [30000, 20000, 10000, 0];

fn compare(script: &str, expected: &str) {
    let mut lines = script.lines();
    let size: usize = lines
        .next()
        .unwrap()
        .strip_prefix("size ")
        .unwrap()
        .parse()
        .unwrap();
    let operations: Vec<_> = lines
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next().unwrap();
            let mut values = [0u32; 3];
            for (i, field) in fields.enumerate() {
                values[i] = field.parse().unwrap();
            }
            (name, values)
        })
        .collect();
    let mut buffer = vec![0; size];
    let mut encoder = RangeEncoder::new(&mut buffer);
    let mut actual = String::new();
    for (i, &(name, [a, b, c])) in operations.iter().enumerate() {
        match name {
            "uint" => encoder.encode_uint(a, b),
            "bits" => encoder.encode_bits(a, b),
            "bit" => encoder.encode_bit_logp(a != 0, b),
            "icdf" => encoder.encode_icdf(a as usize, &ICDF, 8),
            "icdf16" => encoder.encode_icdf16(a as usize, &ICDF16, 15),
            "encode" => encoder.encode(a, b, c),
            "bin" => encoder.encode_bin(a, b, c),
            "patch" => encoder.patch_initial_bits(a, b),
            "shrink" => encoder.shrink(a as usize),
            _ => panic!("unknown operation {name}"),
        }
        writeln!(
            actual,
            "E\t{i}\t{}\t{}\t{}\t{}",
            encoder.tell(),
            encoder.tell_frac(),
            encoder.range(),
            encoder.error()
        )
        .unwrap();
    }
    encoder.finish();
    write!(
        actual,
        "B\t{}\t{}\t",
        encoder.buffer().len(),
        encoder.error()
    )
    .unwrap();
    for &byte in encoder.buffer() {
        write!(actual, "{byte:02x}").unwrap();
    }
    actual.push('\n');
    let mut decoder = RangeDecoder::new(encoder.buffer());
    for (i, &(name, [a, b, c])) in operations.iter().enumerate() {
        let value = match name {
            "uint" => decoder.decode_uint(b),
            "bits" => decoder.decode_bits(b),
            "bit" => u32::from(decoder.decode_bit_logp(b)),
            "icdf" => decoder.decode_icdf(&ICDF, 8) as u32,
            "icdf16" => decoder.decode_icdf16(&ICDF16, 15) as u32,
            "encode" => {
                let value = decoder.decode(c);
                decoder.update(a, b, c);
                value
            }
            "bin" => {
                let value = decoder.decode_bin(c);
                decoder.update(a, b, 1 << c);
                value
            }
            "patch" | "shrink" => continue,
            _ => panic!("unknown operation {name}"),
        };
        writeln!(
            actual,
            "D\t{i}\t{value}\t{}\t{}\t{}\t{}",
            decoder.tell(),
            decoder.tell_frac(),
            decoder.range(),
            decoder.error()
        )
        .unwrap();
    }
    // Compare line by line so a failure identifies the first divergent state.
    for (i, (actual, expected)) in actual.lines().zip(expected.lines()).enumerate() {
        assert_eq!(
            actual,
            expected,
            "reference mismatch at trace line {}",
            i + 1
        );
    }
    assert_eq!(actual.lines().count(), expected.lines().count());
}

macro_rules! fixture {
    ($name:ident) => {
        #[test]
        fn $name() {
            compare(
                include_str!(concat!(
                    "fixtures/reference/entropy/",
                    stringify!($name),
                    ".script"
                )),
                include_str!(concat!(
                    "fixtures/reference/entropy/",
                    stringify!($name),
                    ".tsv"
                )),
            );
        }
    };
}

fixture!(mixed);
fixture!(uint_boundaries);
fixture!(raw_boundaries);
fixture!(empty);
fixture!(zero_storage);
fixture!(overflow);
fixture!(corrupted_uint);
fixture!(overlap);
fixture!(patch_error);
fixture!(patch_0);
fixture!(patch_5);
fixture!(patch_16);
fixture!(patch_64);
