/* Pinned scalar C vectors for the QEXT fixed-point energy math helpers.
 * Build with -DFIXED_POINT -DENABLE_QEXT -DOPUS_BUILD -DVAR_ARRAYS and the
 * pinned reference include paths, then redirect stdout to the binary fixture. */
#include <stdio.h>
#include <stdint.h>
#include "mathops.h"

static uint32_t state = UINT32_C(0x31e7a924);
static uint32_t random_word(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
static void word(int32_t value) {
    uint32_t bits = (uint32_t)value;
    int i;
    for (i = 0; i < 4; i++) putchar((int)((bits >> (8*i)) & 255));
}
static void vector(int32_t energy, int32_t exponent, int32_t fraction) {
    word(energy); word(celt_log2_db(energy));
    word(exponent); word(celt_exp2_db(exponent));
    word(fraction); word(celt_exp2_db_frac(fraction));
}
int main(void) {
    int bit, i;
    vector(0, -17 * (1 << 24), 0);
    vector(INT32_MAX, 15 * (1 << 24), (1 << 24) - 1);
    for (bit = 0; bit < 31; bit++) {
        int32_t energy = (int32_t)(UINT32_C(1) << bit);
        vector(energy - 1, (bit - 18) * (1 << 24), 1);
        vector(energy, (bit - 17) * (1 << 24) + 1, (1 << 23));
        vector(energy + 1, (bit - 16) * (1 << 24) - 1, (1 << 24) - 2);
    }
    for (i = 0; i < 2048; i++) {
        int32_t energy = (int32_t)(random_word() & INT32_MAX);
        int32_t fraction = (int32_t)(random_word() & ((1 << 24) - 1));
        int32_t exponent = ((int32_t)(random_word() % 37) - 20) * (1 << 24) + fraction;
        vector(energy, exponent, fraction);
    }
    return 0;
}
