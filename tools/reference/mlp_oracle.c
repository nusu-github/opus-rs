/* Independent scalar C vectors for the Opus tonality-analysis network. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mlp.h"

static void print_bits(const float *values, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        printf("%08x%c", bits, i + 1 == count ? '\n' : ' ');
    }
}
int main(void) {
    uint32_t seed = 0x12345678;
    float state[24] = {0};
    int frame, i;
    for (frame = 0; frame < 16; ++frame) {
        float input[25], dense[32], output[2];
        for (i = 0; i < 25; ++i) {
            seed = seed * 1664525U + 1013904223U;
            input[i] = (float)((int)(seed >> 16) - 32768) / 8192.f;
        }
        analysis_compute_dense(&layer0, dense, input);
        analysis_compute_gru(&layer1, state, dense);
        analysis_compute_dense(&layer2, output, state);
        print_bits(dense, 32);
        print_bits(state, 24);
        print_bits(output, 2);
    }
    return 0;
}
