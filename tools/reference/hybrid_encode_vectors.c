/* Test-only vectors for the settings in rust/tests/hybrid_encode.rs. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "opus.h"
#include "opus_private.h"

static void check(int result) {
    if (result < 0) {
        fprintf(stderr, "C reference error: %s\n", opus_strerror(result));
        exit(1);
    }
}

static void dump(const char *name, const unsigned char *packet, int bytes,
                 opus_uint32 range) {
    int i;
    printf("pub const %s_PACKET: [u8; %d] = [\n", name, bytes);
    for (i = 0; i < bytes; ++i) {
        printf("%u,%s", packet[i], i % 16 == 15 ? "\n" : " ");
    }
    printf("\n];\npub const %s_RANGE: u32 = %u;\n\n", name, range);
}

static void run(int application, int bitrate, int signal, int seed, int frames) {
    int error, frame, i, channel;
    opus_int16 pcm[960 * 2 * 2];
    unsigned char packet[1500];
    OpusEncoder *encoder = opus_encoder_create(48000, 2, application, &error);
    check(error);
    check(opus_encoder_ctl(encoder, OPUS_SET_FORCE_MODE(MODE_HYBRID)));
    check(opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_FULLBAND)));
    check(opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate)));
    check(opus_encoder_ctl(encoder, OPUS_SET_VBR(0)));
    check(opus_encoder_ctl(encoder, OPUS_SET_DTX(0)));
    check(opus_encoder_ctl(encoder, OPUS_SET_SIGNAL(signal)));
    for (i = 0; i < 960 * frames; ++i) {
        for (channel = 0; channel < 2; ++channel) {
            int value = 2000 + seed + (((i * 37 + channel * 13 + seed) % 400) - 200) * 10;
            if (value > 32767) value = 32767;
            if (value < -32768) value = -32768;
            pcm[2 * i + channel] = (opus_int16)value;
        }
    }
    for (frame = 0; frame < frames; ++frame) {
        opus_uint32 range;
        int bytes = opus_encode(encoder, pcm + 960 * 2 * frame, 960, packet, sizeof(packet));
        check(bytes);
        check(opus_encoder_ctl(encoder, OPUS_GET_FINAL_RANGE(&range)));
        if (frames == 2) dump(frame == 0 ? "HP_DELAY0" : "HP_DELAY1", packet, bytes, range);
        else dump("STEREO_WIDTH", packet, bytes, range);
    }
    opus_encoder_destroy(encoder);
}

int main(void) {
    puts("// Generated from pinned scalar C revision 503d81b138d76621aae4b12786e90de48aa8db3a.");
    puts("// Regenerate with tools/reference/generate_hybrid_vectors.py.");
    run(OPUS_APPLICATION_VOIP, 20000, OPUS_SIGNAL_VOICE, 0, 2);
    run(OPUS_APPLICATION_AUDIO, 12000, OPUS_SIGNAL_MUSIC, 42, 1);
    return 0;
}
