/* Exercise the pinned C decoder's actual static smooth_fade implementation. */
#include "opus_decoder.c"
#include <stdio.h>
#include <string.h>

int main(void)
{
    const float first[4] = {1.f, -1.f, .5f, -.5f};
    const float second[4] = {0.f, .5f, 1.f, -1.f};
    opus_res a[4], b[4], output[4];
#ifdef FIXED_POINT
    const celt_coef window[4] = {0, COEF_ONE / 2 + 1, COEF_ONE, COEF_ONE};
#else
    const celt_coef window[4] = {0.f, .5f, 1.f, 1.f};
#endif
    int i;
    for (i = 0; i < 4; ++i) {
        a[i] = FLOAT2RES(first[i]);
        b[i] = FLOAT2RES(second[i]);
    }
    smooth_fade(a, b, output, 2, 2, window, 48000);
    for (i = 0; i < 4; ++i) {
        float value = RES2FLOAT(output[i]);
        unsigned bits;
        memcpy(&bits, &value, sizeof(bits));
        printf("%d %08x\n", (int)output[i], bits);
    }
    return 0;
}
