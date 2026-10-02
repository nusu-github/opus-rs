/* Scalar fixed math regression vectors, emitted by the pinned C implementation. */
#include <stdint.h>
#include <stdio.h>
#include "mathops.h"

static void unary(const char *name, int32_t input, int32_t output) {
    printf("%s %d 0 %d\n", name, input, output);
}

static void sample(uint32_t value) {
    int32_t positive = (int32_t)(value & UINT32_C(2147483647));
    int32_t root = 536870912 + (positive % 1610612736);
    int32_t reciprocal = 1073741824 + (positive & 1073741823);
    int32_t angle = positive - 1073741824;
    int32_t denominator = (positive | 1);
    int32_t numerator = denominator / 3;
    unary("sqrt", positive, celt_sqrt(positive));
    unary("sqrt32", positive, celt_sqrt32(positive));
    unary("rsqrt", root >> 15, celt_rsqrt_norm(root >> 15));
    unary("rsqrt32", root, celt_rsqrt_norm32(root));
    unary("rcp32", reciprocal, celt_rcp_norm32(reciprocal));
    unary("rcp", denominator, celt_rcp(denominator));
    unary("cos32", angle, celt_cos_norm32(angle));
    unary("atan", angle, celt_atan_norm(angle));
    unary("cos", (int32_t)value, celt_cos_norm((int32_t)value));
    printf("atan2 %d %d %d\n", numerator, denominator, celt_atan2p_norm(numerator, denominator));
    printf("atan2 %d %d %d\n", denominator, numerator, celt_atan2p_norm(denominator, numerator));
    printf("div %d %d %d\n", numerator, denominator, frac_div32(numerator, denominator));
    printf("div %d %d %d\n", -numerator, denominator, frac_div32(-numerator, denominator));
    printf("div29 %d %d %d\n", numerator, denominator, frac_div32_q29(numerator, denominator));
    printf("div29 %d %d %d\n", -numerator, denominator, frac_div32_q29(-numerator, denominator));
}

int main(void) {
    uint32_t state = 0x9e3779b9;
    int i;
    sample(0);
    sample(2147483647);
    for (i = 0; i < 31; ++i) {
        uint32_t bit = UINT32_C(1) << i;
        sample(bit - 1);
        sample(bit);
        sample(bit + 1);
    }
    for (i = 0; i < 512; ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        sample(state);
    }
    unary("cos32", 1073741824, celt_cos_norm32(1073741824));
    unary("atan", 1073741824, celt_atan_norm(1073741824));
    printf("atan2 0 0 %d\n", celt_atan2p_norm(0, 0));
    return 0;
}
