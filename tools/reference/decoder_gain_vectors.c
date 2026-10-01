/* Exact integer decoder gain expressions from the pinned opus_decoder.c. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mathops.h"

int main(void) {
    static const int gains[] = {-32768, -4096, -768, -256, -255, -128, -120, -1,
                                0, 1, 120, 128, 255, 256, 768, 4096, 32767};
#ifdef ENABLE_RES24
    static const int samples[] = {-8388608, -8388607, -4194304, -32768, -32767,
                                  -129, -128, -1, 0, 1, 127, 128, 32767, 32768,
                                  4194304, 8388607};
#else
    static const int samples[] = {-32768, -32767, -16384, -257, -129, -128, -1, 0,
                                  1, 127, 128, 255, 256, 16384, 32766, 32767};
#endif
    unsigned i, j;
    for (i = 0; i < sizeof(gains)/sizeof(*gains); i++) {
        opus_val32 gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), gains[i]));
        for (j = 0; j < sizeof(samples)/sizeof(*samples); j++) {
            opus_res sample = samples[j];
            opus_val32 scaled = sample;
            float input = RES2FLOAT(sample), output;
            uint32_t input_bits, output_bits;
            if (gains[i]) {
#ifdef ENABLE_RES24
                scaled = MULT32_32_Q16(sample, gain);
#else
                scaled = MULT16_32_P16(sample, gain);
#endif
                scaled = SATURATE(scaled, 32767);
            }
            output = RES2FLOAT(scaled);
            memcpy(&input_bits, &input, 4); memcpy(&output_bits, &output, 4);
            printf("%d %d %d %d %08x %08x\n", gains[i], samples[j], gain, scaled,
                   input_bits, output_bits);
        }
    }
    return 0;
}
