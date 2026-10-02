/* Access current C PLC state for the inherited decoder regression scenarios. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "celt_decoder.c"

static unsigned hash16(const opus_int16 *p, int n) {
    unsigned h = 2166136261U;
    int i;
    for (i = 0; i < n; ++i) {
        unsigned v = (unsigned short)p[i];
        h = (h ^ (v & 255)) * 16777619U;
        h = (h ^ (v >> 8)) * 16777619U;
    }
    return h;
}

static unsigned hash32(const opus_int32 *p, int n) {
    unsigned h = 2166136261U;
    int i, b;
    for (i = 0; i < n; ++i) for (b = 0; b < 4; ++b)
        h = (h ^ (((unsigned)p[i] >> (8*b)) & 255)) * 16777619U;
    return h;
}

static void emit(const char *name, CELTDecoder *st, opus_int16 *pcm) {
    int c, stride = DECODE_BUFFER_SIZE + st->overlap;
    celt_glog *energy = (celt_glog *)(st->_decode_mem + stride * st->channels);
    opus_val16 *lpc = (opus_val16 *)(energy + 8 * st->mode->nbEBands);
    unsigned tail = 2166136261U;
    for (c = 0; c < st->channels; ++c) {
        tail ^= hash32(st->_decode_mem + c*stride + DECODE_BUFFER_SIZE - 960, 960 + st->overlap);
        tail *= 16777619U;
    }
    printf("%s %u %u %u\n", name, hash16(lpc, st->channels * CELT_LPC_ORDER), tail, hash16(pcm, 960 * st->channels));
}

static void ratio_vectors(void) {
    int i;
    opus_val32 s1 = 0, s2 = 0;
    for (i = 0; i < 144; ++i) {
        int pos = i%72, pos2 = (i+11)%72;
        int a = ((2*(pos < 36 ? pos : 72-pos)-36)*7000)/36;
        int b = -((2*(pos2 < 36 ? pos2 : 72-pos2)-36)*12000)/36;
        s1 += (a*a)>>11;
        s2 += (b*b)>>11;
    }
    printf("ratio_a %d %d %d\n", s1, s2, celt_sqrt(frac_div32((s1>>1)+1, s2+1)));
    printf("ratio_b %d %d %d\n", s2, s1, celt_sqrt(frac_div32((s2>>1)+1, s1+1)));
    printf("ratio_zero 0 0 %d\n", celt_sqrt(frac_div32(1, 1)));
}

int main(int argc, char **argv) {
    FILE *input;
    int channels, error;
    CELTMode *mode = opus_custom_mode_create(48000, 960, &error);
    if (argc != 2 || !mode || error || !(input = fopen(argv[1], "r"))) return 2;
    for (channels = 1; channels <= 2; ++channels) {
        char hex[2][4001], name[40];
        opus_int16 pcm[1920];
        CELTDecoder *st = opus_custom_decoder_create(mode, channels, &error);
        int frame, pitch = 0;
        if (!st || error) return 2;
        for (frame = 0; frame < 2; ++frame) if (fscanf(input, "%4000s", hex[frame]) != 1) return 2;
        for (frame = 0; frame < 2 && pitch <= 0; ++frame) {
            unsigned char packet[2000];
            int i, len = (int)strlen(hex[frame])/2;
            for (i = 0; i < len; ++i) { unsigned v; sscanf(hex[frame]+2*i, "%2x", &v); packet[i] = (unsigned char)v; }
            if (opus_custom_decode(st, packet, len, pcm, 960) != 960) return 3;
            opus_custom_decoder_ctl(st, OPUS_GET_PITCH(&pitch));
        }
        if (pitch <= 0) return 4;
        for (frame = 0; frame < 2; ++frame) {
            if (opus_custom_decode(st, NULL, 0, pcm, 960) != 960) return 3;
            snprintf(name, sizeof(name), "plc_%d_%d", channels, frame);
            emit(name, st, pcm);
        }
        if (channels == 1) {
            opus_custom_decoder_ctl(st, OPUS_RESET_STATE);
            opus_custom_decoder_ctl(st, CELT_SET_START_BAND(17));
            if (opus_custom_decode(st, NULL, 0, pcm, 960) != 960) return 3;
            emit("noise_reset", st, pcm);
        }
        opus_custom_decoder_destroy(st);
    }
    fclose(input);
    ratio_vectors();
    return 0;
}
