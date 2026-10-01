/* Exact stateful 96 kHz SILK resampler vectors from the pinned C source. */
#include <stdint.h>
#include <stdio.h>
#include "resampler_private.h"

int main(void) {
    const int rates[][3] = {{96000, 16000, 1}, {8000, 96000, 0},
        {12000, 96000, 0}, {16000, 96000, 0}};
    for (int c = 0; c < 4; c++) {
        silk_resampler_state_struct state;
        int in_rate = rates[c][0], out_rate = rates[c][1];
        if (silk_resampler_init(&state, in_rate, out_rate, rates[c][2])) return 1;
        uint32_t seed = UINT32_C(0x12345678);
        for (int frame = 0; frame < 6; frame++) {
            int duration = frame % 2 ? 20 : 10;
            int in_count = in_rate / 1000 * duration;
            int out_count = out_rate / 1000 * duration;
            opus_int16 input[1920], output[1920];
            for (int i = 0; i < in_count; i++) {
                seed = UINT32_C(1664525) * seed + UINT32_C(1013904223);
                input[i] = (opus_int16)(seed >> 16);
            }
            silk_resampler(&state, output, input, in_count);
            printf("%d\t%d\t%d\t%d\t%d\t", in_rate, out_rate, rates[c][2], frame, duration);
            for (int i = 0; i < out_count; i++) printf("%s%d", i ? "," : "", output[i]);
            printf("\n");
        }
    }
    return 0;
}
