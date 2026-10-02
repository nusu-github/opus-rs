/* Test-only access to static helpers in the pinned C fixed-point VQ source. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vq.c"

int main(int argc, char **argv) {
    FILE *input;
    char op[20], name[80];
    int n, a, b, c, d, e, i;
    if (argc != 2 || !(input = fopen(argv[1], "r"))) return 2;
    while (fscanf(input, "%19s %79s %d %d %d %d %d %d", op, name, &n, &a, &b, &c, &d, &e) == 8) {
        celt_norm x[64] = {0}, y[64] = {0};
        int pulses[64] = {0};
        unsigned char packet[256] = {0};
        int scalar1 = 0, scalar2 = 0;
        unsigned range = 0;
        if (n < 1 || n > 64) return 2;
        for (i = 0; i < n; ++i) if (fscanf(input, "%d", &x[i]) != 1) return 2;
        if (!strcmp(op, "normalise") || !strcmp(op, "normrot")) {
            for (i = 0; i < n; ++i) pulses[i] = x[i];
            normalise_residual(pulses, x, n, a, b, 0);
            if (!strcmp(op, "normrot")) exp_rotation(x, n, c, d, e, SPREAD_NORMAL);
        } else if (!strcmp(op, "rotate")) exp_rotation(x, n, a, b, c, d);
        else if (!strcmp(op, "rotate1")) exp_rotation1(x, n, a, b, c);
        else if (!strcmp(op, "renorm")) renormalise_vector(x, n, a, 0);
        else if (!strcmp(op, "search")) {
            scalar1 = op_pvq_search_c(x, pulses, a, n, 0);
            for (i = 0; i < n; ++i) y[i] = pulses[i];
        } else if (!strcmp(op, "alg")) {
            ec_enc encoder;
            ec_dec decoder;
            ec_enc_init(&encoder, packet, sizeof(packet));
            scalar1 = alg_quant(x, n, a, b, c, &encoder, d, e, 0);
            range = encoder.rng;
            ec_enc_done(&encoder);
            ec_dec_init(&decoder, packet, sizeof(packet));
            scalar2 = alg_unquant(y, n, a, b, c, &decoder, d);
            if (range != decoder.rng || encoder.error || decoder.error) return 3;
        } else return 2;
        printf("%s %d %d %u", name, scalar1, scalar2, range);
        for (i = 0; i < n; ++i) printf(" %d", x[i]);
        for (i = 0; i < n; ++i) printf(" %d", y[i]);
        putchar(' ');
        if (!strcmp(op, "alg")) for (i = 0; i < 256; ++i) printf("%02x", packet[i]);
        else putchar('-');
        putchar('\n');
    }
    fclose(input);
    return 0;
}
