/* Independent QEXT vector quantization, refinement, and cubic coding oracle. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vq.h"

static uint32_t scalar_bits(celt_norm value) {
#ifdef FIXED_POINT
    return (uint32_t)value;
#else
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
#endif
}

int main(int argc, char **argv) {
    FILE *input;
    char operation[20], name[96];
    int n, k, spread, blocks, gain_q15, resynth, extra_bits, base_cap, ext_cap, i;
    if (argc != 2 || !(input = fopen(argv[1], "r"))) return 2;
    while (fscanf(input, "%19s %95s %d %d %d %d %d %d %d %d %d", operation, name,
                  &n, &k, &spread, &blocks, &gain_q15, &resynth, &extra_bits,
                  &base_cap, &ext_cap) == 11) {
        celt_norm x[64] = {0}, y[64] = {0};
        opus_val32 gain;
        unsigned char packet[256] = {0}, extension[256] = {0};
        ec_enc encoder, ext_encoder;
        ec_dec decoder, ext_decoder;
        unsigned mask_enc, mask_dec, base_range, ext_range, base_frac, ext_frac;
        int base_tell, ext_tell;
        if (n < 2 || n > 64 || base_cap < 1 || base_cap > 256 || ext_cap < 1 || ext_cap > 256) return 2;
        for (i = 0; i < n; i++) {
            int value;
            if (fscanf(input, "%d", &value) != 1) return 2;
#ifdef FIXED_POINT
            x[i] = value;
#else
            x[i] = value * (1.f / 16777216.f);
#endif
        }
#ifdef FIXED_POINT
        gain = gain_q15 == 32768 ? INT32_MAX : gain_q15 * 65536;
#else
        gain = gain_q15 * (1.f / 32768.f);
#endif
        ec_enc_init(&encoder, packet, base_cap);
        ec_enc_init(&ext_encoder, extension, ext_cap);
        if (!strcmp(operation, "alg"))
            mask_enc = alg_quant(x, n, k, spread, blocks, &encoder, gain, resynth, &ext_encoder, extra_bits, 0);
        else if (!strcmp(operation, "cubic"))
            mask_enc = cubic_quant(x, n, k, blocks, &encoder, gain, resynth);
        else return 2;
        base_range = encoder.rng; ext_range = ext_encoder.rng;
        base_tell = ec_tell(&encoder); ext_tell = ec_tell(&ext_encoder);
        base_frac = ec_tell_frac(&encoder); ext_frac = ec_tell_frac(&ext_encoder);
        ec_enc_done(&encoder); ec_enc_done(&ext_encoder);
        ec_dec_init(&decoder, packet, base_cap);
        ec_dec_init(&ext_decoder, extension, ext_cap);
        if (!strcmp(operation, "alg"))
            mask_dec = alg_unquant(y, n, k, spread, blocks, &decoder, gain, &ext_decoder, extra_bits);
        else mask_dec = cubic_unquant(y, n, k, blocks, &decoder, gain);
        if (encoder.error || ext_encoder.error || decoder.error || ext_decoder.error ||
            base_range != decoder.rng || ext_range != ext_decoder.rng) {
            fprintf(stderr, "Invalid coding state in %s: errors %d %d %d %d\n", name,
                    encoder.error, ext_encoder.error, decoder.error, ext_decoder.error);
            return 3;
        }
        printf("%s %u %u %u %u %d %d %u %u %u %u %d %d", name,
               mask_enc, mask_dec, base_range, ext_range, base_tell, ext_tell,
               base_frac, ext_frac, decoder.rng, ext_decoder.rng,
               ec_tell(&decoder), ec_tell(&ext_decoder));
        for (i = 0; i < n; i++) printf(" %08x", scalar_bits(x[i]));
        for (i = 0; i < n; i++) printf(" %08x", scalar_bits(y[i]));
        putchar(' ');
        for (i = 0; i < base_cap; i++) printf("%02x", packet[i]);
        putchar(' ');
        for (i = 0; i < ext_cap; i++) printf("%02x", extension[i]);
        putchar('\n');
    }
    fclose(input);
    return 0;
}
