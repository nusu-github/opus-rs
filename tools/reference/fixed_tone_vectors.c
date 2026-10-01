/* Test-only access to the pinned fixed CELT encoder's static tone helpers. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "celt_encoder.c"

int main(int argc, char **argv) {
    char op[20], name[80];
    int n, a, b, i;
    FILE *input;
    if (argc != 2 || !(input = fopen(argv[1], "r"))) return 2;
    while (fscanf(input, "%19s %79s %d %d %d", op, name, &n, &a, &b) == 5) {
        opus_val16 samples[1920];
        celt_sig signal[1920];
        if (n < 1 || n > 1920) return 2;
        for (i = 0; i < n; ++i) {
            int value;
            if (fscanf(input, "%d", &value) != 1) return 2;
            signal[i] = value;
            samples[i] = (opus_val16)value;
        }
        printf("%s", name);
        if (!strcmp(op, "normalize")) {
            normalize_tone_input(samples, n);
            for (i = 0; i < n; ++i) printf(" %d", samples[i]);
        } else if (!strcmp(op, "acos")) {
            for (i = 0; i < n; ++i) printf(" %d", acos_approx(signal[i]));
        } else if (!strcmp(op, "lpc")) {
            opus_val32 lpc[2] = {0, 0};
            int fail = tone_lpc(samples, n, a, lpc);
            printf(" %d %d %d", fail, lpc[0], lpc[1]);
        } else if (!strcmp(op, "detect")) {
            opus_val32 toneishness;
            opus_val16 frequency = tone_detect(signal, a, n/a, &toneishness, b);
            printf(" %d %d", frequency, toneishness);
        } else return 2;
        putchar('\n');
    }
    fclose(input);
    return 0;
}
