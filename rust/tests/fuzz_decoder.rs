#[path = "common/mod.rs"]
mod common;

#[test]
fn fuzz_decoder_seed_inputs() {
    common::fuzz_decoder(&[]);
    common::fuzz_decoder(common::TINY_OGG);
}

#[test]
fn reference_ogg_decodes_all_frames_with_the_complete_decoder() {
    assert_eq!(common::decode_stream(common::TINY_OGG), Ok(3));
}
