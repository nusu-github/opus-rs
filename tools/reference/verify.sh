#!/usr/bin/env bash
# Reproduce the supported scalar reference profiles used by CI.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
suite=${1:-baseline}
profile=${2:-float}
case "$profile" in
    float) features=; custom_profile=float ;;
    fixed) features=fixed_point; custom_profile=fixed ;;
    fixed-res24) features=enable_res24; custom_profile=fixed24 ;;
    *) printf '%s\n' 'Profile must be float, fixed, or fixed-res24.' >&2; exit 2 ;;
esac
reference_profile=$profile
case "$suite" in
    baseline) ;;
    pfa) features=${features:+$features,}pfa; reference_profile=${profile/float/pfa};
         if [[ "$profile" != float ]]; then reference_profile=$profile-pfa; fi ;;
    custom) features=${features:+$features,}custom_modes ;;
    custom-pfa) features=${features:+$features,}custom_modes,pfa; custom_profile=$custom_profile-pfa ;;
    *) printf '%s\n' 'Usage: verify.sh baseline|pfa|custom|custom-pfa [float|fixed|fixed-res24]' >&2; exit 2 ;;
esac
export CARGO_INCREMENTAL=${CARGO_INCREMENTAL:-0}
export CARGO_PROFILE_TEST_OPT_LEVEL=${CARGO_PROFILE_TEST_OPT_LEVEL:-2}
export CARGO_PROFILE_TEST_DEBUG=${CARGO_PROFILE_TEST_DEBUG:-0}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$root/target/verify/$suite-$profile"}
case "$CARGO_TARGET_DIR" in /*) ;; *) CARGO_TARGET_DIR="$root/$CARGO_TARGET_DIR" ;; esac
output=${OPUS_VERIFY_OUTPUT:-"$root/target/verification/$suite-$profile"}
mkdir -p "$output"
cargo_features=()
if [[ -n "$features" ]]; then cargo_features=(--features "$features"); fi
failures=0
run() {
    local name=$1
    shift
    printf 'Running %s (%s/%s)\n' "$name" "$suite" "$profile"
    if [[ ${OPUS_VERIFY_DRY_RUN:-0} == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@" 2>&1 | tee "$output/$name.log"
    fi
}
check() {
    if ! run "$@"; then failures=$((failures+1)); fi
}
# Ambient diagnostic controls must not change a recorded baseline profile.
for name in "${!OPUS_ORACLE_@}"; do unset "$name"; done
printf 'suite=%s\nprofile=%s\nfeatures=%s\nreference_revision=%s\n' \
    "$suite" "$profile" "$features" 503d81b138d76621aae4b12786e90de48aa8db3a > "$output/profile.txt"
if [[ "$suite" == custom* ]]; then
    run build-reference bash tools/reference/build_custom.sh "$custom_profile"
    check regenerate-fixtures python3 tools/reference/generate_custom_vectors.py "$custom_profile"
    check fixture-drift git diff --exit-code -- rust/tests/fixtures/reference
    check mode-tables cargo test --locked "${cargo_features[@]}" --lib dynamic_mode_tables_match_pinned_c
    check custom-codec cargo test --locked "${cargo_features[@]}" --test custom_reference
else
    suffix=-$reference_profile
    if [[ "$reference_profile" == float ]]; then suffix=; fi
    reference=${OPUS_VERIFY_REFERENCE:-"$root/target/reference$suffix/opus-reference"}
    candidate=${OPUS_VERIFY_CANDIDATE:-"$CARGO_TARGET_DIR/release/opus-rs-oracle"}
    if [[ -z ${OPUS_VERIFY_REFERENCE:-} ]]; then run build-reference bash tools/reference/build.sh "$reference_profile"; fi
    if [[ -z ${OPUS_VERIFY_CANDIDATE:-} ]]; then run build-candidate cargo build --locked --release "${cargo_features[@]}" --bin opus-rs-oracle; fi
    if [[ "$profile" != float ]]; then
        check regenerate-downmix python3 tools/reference/generate_downmix_vectors.py "$reference_profile"
        check downmix-fixture-drift git diff --exit-code -- "rust/tests/fixtures/reference/$reference_profile-downmix.tsv"
        check downmix-reference cargo test --locked "${cargo_features[@]}" --test codec_downmix_reference
    fi
    modes=()
    if [[ "$suite" == pfa ]]; then
        modes=(--mode celt)
        generator_flags=()
        if [[ "$profile" != float ]]; then generator_flags=(--fixed); fi
        check regenerate-transforms python3 tools/reference/generate_pfa_transform_vectors.py "${generator_flags[@]}"
        check fixture-drift git diff --exit-code -- rust/tests/fixtures/reference
        check transforms cargo test --locked "${cargo_features[@]}" --lib matches_pinned_c_transform_vectors
        if [[ "$profile" != float ]]; then
            check plc-state cargo test --locked "${cargo_features[@]}" --lib decoder_plc_iir_matches_ctest_vectors
            check integer-fade cargo test --locked "${cargo_features[@]}" --lib smooth_fade_matches_pinned_integer_c
            check plc-reference cargo test --locked "${cargo_features[@]}" --test fixed_plc_reference
        fi
    fi
    check codec python3 tools/reference/compare_codec.py --reference "$reference" --candidate "$candidate" \
        --profile "$reference_profile" --record "$output/codec.json" "${modes[@]}"
    check decode python3 tools/reference/compare_decode.py --reference "$reference" --candidate "$candidate" --loss "${modes[@]}"
    check sequences python3 tools/reference/compare_sequences.py --reference "$reference" --candidate "$candidate"
    if [[ "$suite" == baseline ]]; then
        if [[ "$profile" != float ]]; then
            check unit-tests cargo test --locked "${cargo_features[@]}" --lib
            check plc-reference cargo test --locked "${cargo_features[@]}" --test fixed_plc_reference
        fi
        check controls python3 tools/reference/compare_controls.py --reference "$reference" --candidate "$candidate" --record "$output/controls.json"
        check fec-controls env OPUS_ORACLE_FEC=1 OPUS_ORACLE_LOSS=15 python3 tools/reference/compare_controls.py \
            --reference "$reference" --candidate "$candidate" --record "$output/fec-controls.json"
        check encoder-edges python3 tools/reference/compare_encoder_edges.py --reference "$reference" --candidate "$candidate" --record "$output/encoder-edges.json"
        if [[ ${OPUS_VERIFY_VECTORS:-1} == 1 ]]; then
            run fetch-vectors python3 tools/reference/fetch_vectors.py
            check official-vectors python3 tools/reference/compare_vectors.py --reference "$reference" --candidate "$candidate" \
                --profile "$reference_profile" --rates 8000 12000 16000 24000 48000 --record "$output/rfc8251.json"
        else
            printf '%s\n' 'Official vectors explicitly disabled by OPUS_VERIFY_VECTORS=0.' | tee "$output/official-vectors-skipped.log"
        fi
        if [[ "$profile" == float ]]; then
            check packet-reference cargo test --locked --test packet_reference -- --ignored
            check multistream-reference cargo test --locked --test multistream_reference -- --ignored
            check surround-reference cargo test --locked --lib surround_mask_matches_pinned_reference -- --ignored
        fi
    fi
fi
printf '%s/%s: %s failing stages\n' "$suite" "$profile" "$failures"
test "$failures" -eq 0
