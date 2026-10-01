/* Independent fixed-point frontend vectors from the pinned Opus C source. */
#include <stdint.h>
#include <stdio.h>
#include "opus_encoder.c"

static int16_t sample(int frame, int i, int pattern) {
    uint32_t value = (uint32_t)(i + 1) * 2654435761u + (uint32_t)frame * 2246822519u;
    if (pattern == 0) return (int16_t)(value >> 16);
    if (pattern == 1) return (int16_t)((i & 1) ? 2000 : 27000);
    return (int16_t)((i & 2) ? -12345 : 12345);
}
static uint64_t hash_samples(const opus_res *pcm, int n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    int i, j;
    for (i = 0; i < n; ++i) {
        uint32_t value = (uint32_t)(int32_t)pcm[i];
        for (j = 0; j < 4; ++j) { hash ^= (value >> (8*j)) & 255; hash *= UINT64_C(1099511628211); }
    }
    return hash;
}
int main(void) {
    const int rates[] = {8000,12000,16000,24000,48000
#ifdef ENABLE_QEXT
        ,96000
#endif
    };
    const int rate_count = sizeof(rates)/sizeof(rates[0]);
    const int gains[] = {0,8192,16384,24576,32767};
    opus_res pcm[5760];
    const CELTMode *mode = opus_custom_mode_create(48000,960,NULL);
    int r, pattern, frame, i, a, b, channels;
    for (r = 0; r < rate_count; ++r) for (pattern = 0; pattern < 3; ++pattern) {
        StereoWidthState state = {0};
        int n = rates[r]/50;
        for (frame = 0; frame < 16; ++frame) {
            int width;
            for (i = 0; i < 2*n; ++i) pcm[i] = INT16TORES(sample(frame,i,pattern));
            width = compute_stereo_width(pcm,n,rates[r],&state);
            printf("W %d %d %d %d %d %d %d %d %d\n",rates[r],pattern,frame,width,state.XX,state.XY,state.YY,state.smoothed_width,state.max_follower);
        }
    }
    for (r = 0; r < rate_count; ++r) for (a = 0; a < 5; ++a) for (b = 0; b < 5; ++b) for (channels = 1; channels <= 2; ++channels) {
        int n = rates[r]/100;
        const CELTMode *fade_mode = mode;
#ifdef ENABLE_QEXT
        if (rates[r] == 96000) fade_mode = opus_custom_mode_create(96000,1920,NULL);
#endif
        for (i = 0; i < channels*n; ++i) pcm[i] = INT16TORES(sample(a,i,b%3));
        gain_fade(pcm,pcm,gains[a],gains[b],fade_mode->overlap,n,channels,fade_mode->window,rates[r]);
        printf("G %d %d %d %d %016llx\n",rates[r],a,b,channels,(unsigned long long)hash_samples(pcm,channels*n));
        if (channels == 2) {
            for (i = 0; i < channels*n; ++i) pcm[i] = INT16TORES(sample(a,i,b%3));
            stereo_fade(pcm,pcm,gains[a],gains[b],fade_mode->overlap,n,channels,fade_mode->window,rates[r]);
            printf("S %d %d %d %d %016llx\n",rates[r],a,b,channels,(unsigned long long)hash_samples(pcm,channels*n));
        }
    }
    {
        const int celt_rates[] = {0,1,1023,1024,16000,16383,32767,32768,35000,40000,64000,65535,65536,70000,100000,200000};
        for (i = 0; i < 16; ++i) {
            opus_val16 gain = Q15ONE - SHR32(celt_exp2(-celt_rates[i] * QCONST16(1.f/1024,10)),1);
            printf("H %d %d\n",celt_rates[i],gain);
        }
    }
#ifdef ENABLE_RES24
    for (channels = 1; channels <= 2; ++channels) for (a = 1; a <= 2; ++a) {
        celt_sig signal[64], memory = 123456;
        int n = 32*a;
        for (i = 0; i < 32*channels; ++i) pcm[i] = i*167111 - 18331119;
        celt_preemphasis(pcm,signal,n,channels,a,mode->preemph,&memory,1);
        printf("P %d %d %d %016llx\n",channels,a,memory,(unsigned long long)hash_samples(signal,n));
    }
#endif
    return 0;
}
