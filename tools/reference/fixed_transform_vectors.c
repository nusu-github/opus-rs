/* Exact test-only fixed-point FFT/MDCT vectors from the pinned C codec. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "opus_custom.h"
#include "modes.h"
#include "mdct.h"
#include "kiss_fft.h"

static void word(FILE *file, uint32_t value) {
    int i;
    for (i = 0; i < 4; ++i) fputc((int)((value >> (8 * i)) & 255), file);
}

static int32_t sample(unsigned index, unsigned pattern) {
    switch (pattern) {
    case 0: return 0;
    case 1: return index == 3 ? 33554432 : 0;
    case 2: return (int32_t)((index * UINT32_C(1664525) + UINT32_C(1013904223)) & UINT32_C(16777215)) - 8388608;
    default: return index & 1 ? -134217728 : 134217727;
    }
}

int main(int argc, char **argv) {
    int error, shift, stride, pattern, i;
    const OpusCustomMode *mode = opus_custom_mode_create(argc>3?atoi(argv[3]):48000, argc>4?atoi(argv[4]):960, &error);
    FILE *mdct, *fft;
    if ((argc != 3 && argc != 5) || !mode || error) return 2;
    mdct = fopen(argv[1], "wb");
    fft = fopen(argv[2], "wb");
    if (!mdct || !fft) return 2;
    word(mdct, mode->overlap);
    for (i = 0; i < mode->overlap; ++i) word(mdct, (uint32_t)mode->window[i]);
    for (shift = 0; shift <= 3; ++shift) {
        int n = (mode->mdct.n/2) >> shift;
        for (pattern = 0; pattern < 4; ++pattern) {
            const kiss_fft_state *state = mode->mdct.kfft[shift];
            kiss_fft_cpx in[960], forward[960], backward[960];
            for (i = 0; i < state->nfft; ++i) {
                in[i].r = sample(2 * i, pattern) >> 4;
                in[i].i = sample(2 * i + 1, pattern) >> 4;
            }
            opus_fft_c(state, in, forward);
            opus_ifft_c(state, in, backward);
            word(fft, shift); word(fft, pattern); word(fft, state->nfft);
            for (i = 0; i < state->nfft; ++i) {
                word(fft, (uint32_t)forward[i].r); word(fft, (uint32_t)forward[i].i);
                word(fft, (uint32_t)backward[i].r); word(fft, (uint32_t)backward[i].i);
            }
            for (stride = 1; stride <= 3; stride += 2) {
                int32_t in[2160], freq[5760], spectrum[5760], time[2040];
                for (i = 0; i < n + mode->overlap; ++i) in[i] = sample(i, pattern);
                for (i = 0; i < n * stride; ++i) {
                    freq[i] = sample(i, pattern);
                    spectrum[i] = 0x13579;
                }
                for (i = 0; i < n + mode->overlap/2; ++i) time[i] = sample(i + 11, pattern) >> 4;
                clt_mdct_forward_c(&mode->mdct, in, spectrum, mode->window, mode->overlap, shift, stride, 0);
                clt_mdct_backward_c(&mode->mdct, freq, time, mode->window, mode->overlap, shift, stride, 0);
                word(mdct, shift); word(mdct, stride); word(mdct, pattern);
                word(mdct, n * stride); word(mdct, n + mode->overlap/2);
                for (i = 0; i < n * stride; ++i) word(mdct, (uint32_t)spectrum[i]);
                for (i = 0; i < n + mode->overlap/2; ++i) word(mdct, (uint32_t)time[i]);
            }
        }
    }
    fclose(mdct); fclose(fft);
    /* This reference build returns the process-lifetime canonical static mode. */
    return 0;
}
