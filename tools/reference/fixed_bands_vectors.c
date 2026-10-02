/* Current Q24 band normalization fixtures; no Rust-generated expected data. */
#include <stdio.h>
#include "opus_custom.h"
#include "modes.h"
#include "bands.h"

static void dump(const char *name, const opus_int32 *values, int len) {
    int i;
    printf("%s", name);
    for (i = 0; i < len; ++i) printf(" %d", values[i]);
    putchar('\n');
}

int main(void) {
    int error;
    const OpusCustomMode *canonical = opus_custom_mode_create(48000, 960, &error);
    CELTMode mode;
    opus_int16 bands[] = {0, 1, 2, 4};
    celt_sig frequency[] = {1000, -2000, 3000, -4000};
    celt_ener energy[] = {1 << 10, 1 << 20, 1 << 15};
    celt_norm normalized[4];
    celt_norm input[] = {1234, -2345, 3456, -4567};
    celt_glog logarithm[] = {-(18 << DB_SHIFT), 17 << DB_SHIFT, 10 << DB_SHIFT};
    if (!canonical || error) return 2;
    mode = *canonical;
    mode.eBands = bands;
    mode.nbEBands = 3;
    mode.effEBands = 3;
    mode.shortMdctSize = 4;
    mode.nbShortMdcts = 1;
    normalise_bands(&mode, frequency, normalized, energy, 3, 1, 1);
    dump("normalise", normalized, 4);
    denormalise_bands(&mode, input, frequency, logarithm, 0, 3, 1, 1, 0);
    dump("denormalise", frequency, 4);
    denormalise_bands(&mode, input, frequency, logarithm, 0, 3, 1, 2, 0);
    dump("downsample", frequency, 4);
    denormalise_bands(&mode, input, frequency, logarithm, 0, 3, 1, 1, 1);
    dump("silence", frequency, 4);
    return 0;
}
