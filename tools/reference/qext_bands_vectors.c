/* Stateful QEXT band quantization oracle; every integer is little-endian. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bands.h"
#include "modes.h"
#include "vq.h"

static void word(uint32_t value) {
    int i;
    for (i = 0; i < 4; i++) putchar((value >> (8*i)) & 255);
}
static void scalar(celt_norm value) {
#ifdef FIXED_POINT
    word((uint32_t)value);
#else
    uint32_t bits;
    memcpy(&bits, &value, 4);
    word(bits);
#endif
}
static void scalars(const celt_norm *values, int length) {
    int i;
    for (i = 0; i < length; i++) scalar(values[i]);
}
static void state(ec_ctx *base, ec_ctx *ext, uint32_t seed) {
    word(base->rng); word(ext->rng);
    word(ec_tell(base)); word(ec_tell(ext));
    word(ec_tell_frac(base)); word(ec_tell_frac(ext)); word(seed);
}

int main(void) {
    int lm, short_blocks, depth_index, layout, variant, case_index = 0;
    const int depths[] = {0, 2, 6};
    word(0x51424e44); word(288);
    for (lm = 0; lm < 4; lm++) for (short_blocks = 0; short_blocks < 2; short_blocks++)
    for (depth_index = 0; depth_index < 3; depth_index++) for (layout = 0; layout < 3; layout++)
    for (variant = 0; variant < 4; variant++) {
        CELTMode *owner, extra_mode;
        const CELTMode *mode;
        int error, i, c, rate = layout == 2 ? 96000 : 48000;
        int channels = variant == 0 ? 1 : 2, depth = depths[depth_index];
        int m = 1 << lm, count, bands, end, start = 0, base_sum = 0, ext_sum = 0;
        int pulses[32] = {0}, extra[32] = {0}, caps[32] = {0}, tf[32] = {0};
        int base_cap, ext_cap, dual = variant == 2, intensity, complexity = variant == 3 ? 4 : 10;
        celt_norm input[3840] = {0}, encoded[3840], decoded[3840] = {0};
        celt_ener energies[64];
        unsigned char masks[64] = {0}, decoded_masks[64] = {0};
        unsigned char packet[1275] = {0}, extension[4096] = {0};
        uint32_t enc_seed = 0x12345678, dec_seed = 0x12345678;
        ec_enc enc, ext_enc;
        ec_dec dec, ext_dec;
        owner = opus_custom_mode_create(rate, rate / 50, &error);
        if (!owner || error) return 2;
        mode = owner;
        if (layout) { compute_qext_mode(&extra_mode, owner); mode = &extra_mode; }
        bands = mode->nbEBands; end = mode->effEBands;
        count = mode->shortMdctSize * m;
        intensity = variant == 3 ? end / 2 : end;
        if (!layout) init_caps(mode, caps, lm, channels);
        for (i = 0; i < bands; i++) {
            int n = m * (mode->eBands[i+1] - mode->eBands[i]);
            pulses[i] = channels * ((n - 1) * 2 + 12) * 8;
            if (!layout && pulses[i] > caps[i]) pulses[i] = caps[i];
            extra[i] = channels * (n - 1) * depth * 8;
            if (i < end) { base_sum += pulses[i]; ext_sum += extra[i]; }
            tf[i] = lm > 0 && i % 3 == 1 ? (short_blocks ? -1 : 1) : 0;
            for (c = 0; c < channels; c++) {
#ifdef FIXED_POINT
                energies[c*bands+i] = 16777216;
#else
                energies[c*bands+i] = 1.f;
#endif
            }
        }
        for (c = 0; c < channels; c++) for (i = start; i < end; i++) {
            int j, n = m * (mode->eBands[i+1] - mode->eBands[i]);
            celt_norm *band = input + c*count + m*mode->eBands[i];
            for (j = 0; j < n; j++) {
                int raw = (((j*7919 + i*104729 + c*15427 + lm*541) % 65536) - 32768) * 128;
#ifdef FIXED_POINT
                band[j] = raw;
#else
                band[j] = raw * (1.f / 16777216.f);
#endif
            }
#ifdef FIXED_POINT
            renormalise_vector(band, n, 2147483647, 0);
#else
            renormalise_vector(band, n, 1.f, 0);
#endif
        }
        base_cap = (base_sum + 63) / 64 + 8;
        ext_cap = depth ? (ext_sum + 63) / 64 + 8 : 0;
        if (base_cap > 1275 || ext_cap > 4096 || count*channels > 3840) return 3;
        memcpy(encoded, input, count*channels*sizeof(*input));
        ec_enc_init(&enc, packet, base_cap); ec_enc_init(&ext_enc, extension, ext_cap);
        quant_all_bands(1, mode, start, end, encoded, channels == 2 ? encoded+count : NULL,
            masks, energies, pulses, short_blocks, variant, dual, intensity, tf,
            base_cap*64, 0, &enc, lm, end, &enc_seed, complexity, 0, 0,
            &ext_enc, extra, ext_cap*64, layout ? NULL : caps);
        word(case_index++); word(rate); word(lm); word(channels); word(short_blocks);
        word(depth); word(layout != 0); word(variant); word(dual); word(intensity);
        word(complexity); word(start); word(end); word(bands); word(count);
        word(base_cap); word(ext_cap);
        for (i = 0; i < bands; i++) { word(pulses[i]); word(extra[i]); word(caps[i]); word(tf[i]); }
        scalars(input, count*channels); scalars(encoded, count*channels);
        state(&enc, &ext_enc, enc_seed);
        for (i = 0; i < bands*channels; i++) word(masks[i]);
        ec_enc_done(&enc); ec_enc_done(&ext_enc);
        ec_dec_init(&dec, packet, base_cap); ec_dec_init(&ext_dec, extension, ext_cap);
        quant_all_bands(0, mode, start, end, decoded, channels == 2 ? decoded+count : NULL,
            decoded_masks, energies, pulses, short_blocks, variant, dual, intensity, tf,
            base_cap*64, 0, &dec, lm, end, &dec_seed, complexity, 0, 0,
            &ext_dec, extra, ext_cap*64, layout ? NULL : caps);
        scalars(decoded, count*channels); state(&dec, &ext_dec, dec_seed);
        for (i = 0; i < bands*channels; i++) word(decoded_masks[i]);
        if (enc.error || ext_enc.error || dec.error || ext_dec.error || enc.rng != dec.rng || ext_enc.rng != ext_dec.rng) {
            fprintf(stderr, "Invalid coding state in band case %d: %d %d %d %d\n", case_index-1, enc.error, ext_enc.error, dec.error, ext_dec.error);
            return 4;
        }
        fwrite(packet, 1, base_cap, stdout); fwrite(extension, 1, ext_cap, stdout);
        /* The reference uses statically allocated standard modes. */
    }
    return ferror(stdout) ? 5 : 0;
}
