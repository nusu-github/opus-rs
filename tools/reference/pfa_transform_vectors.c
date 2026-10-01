/* Exact test-only floating-point FFT/MDCT vectors from the pinned C codec. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opus_custom.h"
#include "modes.h"
#include "mdct.h"
#include "kiss_fft.h"

static void word(FILE *file, uint32_t value) {
    int i;
    for (i = 0; i < 4; ++i) fputc((int)((value >> (8 * i)) & 255), file);
}

static float sample(unsigned index, unsigned pattern) {
    switch (pattern) {
    case 0: return 0;
    case 1: return index == 3 ? 2.f : 0;
    case 2: return ((int32_t)((index * UINT32_C(1664525) + UINT32_C(1013904223)) & UINT32_C(16777215)) - 8388608) / 16777216.f;
    default: return index & 1 ? -8.f : 7.9999995f;
    }
}

static uint32_t bits(float value) { uint32_t result; memcpy(&result, &value, 4); return result; }

int main(int argc, char **argv) {
    int error, shift, stride, pattern, i;
    int rate = argc == 5 ? atoi(argv[3]) : 48000;
    int size = argc == 5 ? atoi(argv[4]) : 960;
    const OpusCustomMode *mode = opus_custom_mode_create(rate, size, &error);
    FILE *mdct, *fft;
    if ((argc != 3 && argc != 5) || !mode || error) return 2;
    mdct = fopen(argv[1], "wb");
    fft = fopen(argv[2], "wb");
    if (!mdct || !fft) return 2;
    word(mdct, mode->overlap);
    for (i = 0; i < mode->overlap; ++i) word(mdct, bits(mode->window[i]));
    for (shift = 0; shift <= 3; ++shift) {
        int n = (mode->mdct.n / 2) >> shift;
        for (pattern = 0; pattern < 4; ++pattern) {
            const kiss_fft_state *state = mode->mdct.kfft[shift];
            kiss_fft_cpx in[960], forward[960], backward[960];
            for (i = 0; i < state->nfft; ++i) {
                in[i].r = sample(2 * i, pattern) * (1.f / 16.f);
                in[i].i = sample(2 * i + 1, pattern) * (1.f / 16.f);
            }
            opus_fft_c(state, in, forward);
            opus_ifft_c(state, in, backward);
            word(fft, shift); word(fft, pattern); word(fft, state->nfft);
            for (i = 0; i < state->nfft; ++i) {
                word(fft, bits(forward[i].r)); word(fft, bits(forward[i].i));
                word(fft, bits(backward[i].r)); word(fft, bits(backward[i].i));
            }
            for (stride = 1; stride <= 3; stride += 2) {
                float in[2160], freq[5760], spectrum[5760], time[2040];
                for (i = 0; i < n + mode->overlap; ++i) in[i] = sample(i, pattern);
                for (i = 0; i < n * stride; ++i) {
                    freq[i] = sample(i, pattern);
                    spectrum[i] = 0x13579;
                }
                for (i = 0; i < n + mode->overlap/2; ++i) time[i] = sample(i + 11, pattern) * (1.f / 16.f);
                clt_mdct_forward_c(&mode->mdct, in, spectrum, mode->window, mode->overlap, shift, stride, 0);
                clt_mdct_backward_c(&mode->mdct, freq, time, mode->window, mode->overlap, shift, stride, 0);
                word(mdct, shift); word(mdct, stride); word(mdct, pattern);
                word(mdct, n * stride); word(mdct, n + mode->overlap/2);
                for (i = 0; i < n * stride; ++i) word(mdct, bits(spectrum[i]));
                for (i = 0; i < n + mode->overlap/2; ++i) word(mdct, bits(time[i]));
            }
        }
    }
    fclose(mdct); fclose(fft);
    /* This reference build returns the process-lifetime canonical static mode. */
    return 0;
}
