# QEXT transform and filter oracle fixtures

All expected values come from the scalar C implementation at
`503d81b138d76621aae4b12786e90de48aa8db3a`. Rust never generates expected
transform or filter outputs. Reference builds disable floating-point
contraction, fast math, SIMD, and compiler vectorization.

The QEXT coefficient tables are extracted from the pinned `static_modes`
headers by `tools/reference/extract_qext_tables.py`. Fixed coefficients use
Q31 precision. Canonical 48 kHz and 96 kHz transforms use the original static
tables; custom modes retain the C runtime table-generation formulas.

Generate FFT and MDCT vectors for all four arithmetic profiles:

```sh
for profile in qext qext-pfa fixed-qext fixed-qext-pfa; do
    sh tools/reference/build.sh "$profile"
    python3 tools/reference/generate_qext_transform_vectors.py "$profile"
done
```

Each profile has 32 FFT cases and 64 MDCT cases across the 48 kHz/960-sample
and 96 kHz/1920-sample canonical modes. Cases cover all four transform shifts,
four deterministic input patterns, forward and inverse transforms, and MDCT
strides one and three. Values are little-endian IEEE-754 bits or signed
32-bit fixed values. QEXT normalizes forward MDCT coefficients after the FFT,
which differs numerically from the standard profile's earlier normalization.

Regenerate the filter fixtures after building the scalar float and fixed
QEXT references above:

```sh
sh tools/reference/build_custom.sh fixed-qext
python3 tools/reference/generate_qext_filter_vectors.py
```

`qext_comb_vectors.c` generates the fixed Q31 comb fixtures against
`target/reference-fixed-qext/source/libopus.a`, with `FIXED_POINT`,
`ENABLE_QEXT`, and `OPUS_BUILD` defined. Run it with an output filename for
144 ordinary comb cases; append `96` for 72 cases covering the independent
even and odd sample filters at 96 kHz. Building the same driver without
`FIXED_POINT` against the float QEXT library produces the corresponding
float output bits. Cases cover all tapset pairs, positive/negative/zero gains,
windowed transitions, and both in-place and separate input/output buffers.

`qext_preemphasis_vectors.c` links against the fixed QEXT Custom library
built by `tools/reference/build_custom.sh fixed-qext`. Compile it with
`FIXED_POINT`, `ENABLE_QEXT`, `CUSTOM_MODES`, and `OPUS_BUILD` defined.
Its output contains 42 preemphasis cases across seven sample rates, three
upsampling factors, and mono/stereo input, including the final filter state.
This exercises the Newton–Raphson coefficient correction and the ordinary
single-coefficient path.

The unit tests `matches_pinned_c_transform_vectors`,
`comb_filter_matches_pinned_c_vectors`, and
`qext_preemphasis_matches_pinned_c_vectors` compare every result exactly.
