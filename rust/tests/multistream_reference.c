/* Test-only process oracle for native multistream/projection and masking. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_custom.h"
#include "opus_multistream.h"
#include "opus_projection.h"
#include "opus_private.h"
#include "modes.h"
#include "float_cast.h"

extern void surround_analysis(const CELTMode *, const void *, celt_glog *,
    opus_val32 *, opus_val32 *, int, int, int, int, opus_copy_channel_in_func, int);

static void copy_float(opus_res *dst, int dst_stride, const void *src,
    int src_stride, int channel, int frame_size, void *user_data) {
    int i;
    const float *pcm = (const float *)src;
    (void)user_data;
    for (i=0; i<frame_size; i++) dst[i*dst_stride] = FLOAT2RES(pcm[i*src_stride+channel]);
}

static uint64_t hash_bytes(const void *data, size_t len) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t i;
    for (i=0; i<len; i++) hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}

int main(int argc, char **argv) {
    const char *kind, *format;
    int rate, channels, frame_size, frames, application, vbr, bitrate;
    int streams, coupled, error, frame, i, bytes_per_sample;
    unsigned char mapping[256], output[65536];
    FILE *input;
    void *pcm;
    OpusMSEncoder *encoder = NULL;
    OpusProjectionEncoder *projection = NULL;
    OpusCustomMode *mode = NULL;
    OpusMSDecoder *decoders[3] = {NULL};
    OpusProjectionDecoder *projection_decoders[3] = {NULL};
    opus_int16 *decoded16;
    opus_int32 *decoded24;
    float *decoded_float;
    opus_val32 memory[8*240] = {0}, preemphasis[8] = {0};
    if (argc != 11) return 2;
    kind = argv[1]; rate = atoi(argv[2]); channels = atoi(argv[3]);
    frame_size = atoi(argv[4]); frames = atoi(argv[5]); format = argv[6];
    application = atoi(argv[7]); vbr = atoi(argv[8]); bitrate = atoi(argv[9]);
    bytes_per_sample = !strcmp(format, "s16") ? 2 : 4;
    input = fopen(argv[10], "rb");
    if (!input) return 3;
    pcm = calloc(frame_size*channels, bytes_per_sample);
    if (!strcmp(kind, "mask")) {
        mode = opus_custom_mode_create(rate == 96000 ? 96000 : 48000, rate == 96000 ? 1920 : 960, &error);
        if (!mode || channels > 8 || channels < 3) return 4;
    } else if (!strcmp(kind, "projection")) {
        projection = opus_projection_ambisonics_encoder_create(rate, channels, 3,
            &streams, &coupled, application, &error);
        if (!projection) return 5;
        opus_projection_encoder_ctl(projection, OPUS_SET_VBR(vbr));
        opus_projection_encoder_ctl(projection, OPUS_SET_BITRATE(bitrate));
#ifdef ENABLE_QEXT
        if (opus_projection_encoder_ctl(projection, OPUS_SET_QEXT(1))) return 11;
#endif
    } else {
        if (!strcmp(kind, "streams")) {
            streams = channels;
            coupled = 0;
            for (i=0; i<channels; i++) mapping[i] = i;
            encoder = opus_multistream_encoder_create(rate, channels, channels, 0,
                mapping, application, &error);
        } else {
            int family = !strcmp(kind, "surround") ? 1 : 2;
            encoder = opus_multistream_surround_encoder_create(rate, channels, family,
                &streams, &coupled, mapping, application, &error);
        }
        if (!encoder) return 6;
        opus_multistream_encoder_ctl(encoder, OPUS_SET_VBR(vbr));
        opus_multistream_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
#ifdef ENABLE_QEXT
        if (opus_multistream_encoder_ctl(encoder, OPUS_SET_QEXT(1))) return 12;
#endif
    }
    decoded16 = calloc(frame_size*channels, sizeof(*decoded16));
    decoded24 = calloc(frame_size*channels, sizeof(*decoded24));
    decoded_float = calloc(frame_size*channels, sizeof(*decoded_float));
    if (projection) {
        int matrix_size;
        unsigned char *matrix;
        opus_projection_encoder_ctl(projection, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE(&matrix_size));
        matrix = malloc(matrix_size);
        opus_projection_encoder_ctl(projection, OPUS_PROJECTION_GET_DEMIXING_MATRIX(matrix, matrix_size));
        for (i=0; i<3; i++) {
            projection_decoders[i] = opus_projection_decoder_create(rate, channels, streams, coupled, matrix, matrix_size, &error);
            if (!projection_decoders[i]) return 8;
        }
        free(matrix);
    } else if (encoder) {
        for (i=0; i<3; i++) {
            decoders[i] = opus_multistream_decoder_create(rate, channels, streams, coupled, mapping, &error);
            if (!decoders[i]) return 9;
        }
    }
    for (frame=0; frame<frames; frame++) {
        int length;
        opus_uint32 range = 0;
        if (fread(pcm, bytes_per_sample, frame_size*channels, input) != (size_t)(frame_size*channels)) return 7;
        if (mode) {
            celt_glog masks[8*21];
            surround_analysis(mode, pcm, masks, memory, preemphasis, frame_size,
                mode->overlap, channels, rate, copy_float, 0);
            for (i=0; i<channels*21; i++) {
                uint32_t bits;
                memcpy(&bits, &masks[i], sizeof(bits));
                printf("%08x ", bits);
            }
            putchar('\n');
            continue;
        }
        if (projection) {
            if (!strcmp(format, "f32")) length = opus_projection_encode_float(projection, pcm, frame_size, output, 65536);
            else if (!strcmp(format, "s24")) length = opus_projection_encode24(projection, pcm, frame_size, output, 65536);
            else length = opus_projection_encode(projection, pcm, frame_size, output, 65536);
            opus_projection_encoder_ctl(projection, OPUS_GET_FINAL_RANGE(&range));
        } else {
            if (!strcmp(format, "f32")) length = opus_multistream_encode_float(encoder, pcm, frame_size, output, 65536);
            else if (!strcmp(format, "s24")) length = opus_multistream_encode24(encoder, pcm, frame_size, output, 65536);
            else length = opus_multistream_encode(encoder, pcm, frame_size, output, 65536);
            opus_multistream_encoder_ctl(encoder, OPUS_GET_FINAL_RANGE(&range));
        }
        printf("%d %u ", length, range);
        for (i=0; i<length; i++) printf("%02x", output[i]);
        if (length > 0) {
            int samples16, samples24, samples_float;
            if (projection) {
                samples16 = opus_projection_decode(projection_decoders[0], output, length, decoded16, frame_size, 0);
                samples24 = opus_projection_decode24(projection_decoders[1], output, length, decoded24, frame_size, 0);
                samples_float = opus_projection_decode_float(projection_decoders[2], output, length, decoded_float, frame_size, 0);
            } else {
                samples16 = opus_multistream_decode(decoders[0], output, length, decoded16, frame_size, 0);
                samples24 = opus_multistream_decode24(decoders[1], output, length, decoded24, frame_size, 0);
                samples_float = opus_multistream_decode_float(decoders[2], output, length, decoded_float, frame_size, 0);
            }
            if (samples16 != frame_size || samples24 != frame_size || samples_float != frame_size) return 10;
            printf(" %016llx %016llx %016llx",
                (unsigned long long)hash_bytes(decoded16, samples16*channels*sizeof(*decoded16)),
                (unsigned long long)hash_bytes(decoded24, samples24*channels*sizeof(*decoded24)),
                (unsigned long long)hash_bytes(decoded_float, samples_float*channels*sizeof(*decoded_float)));
        }
        putchar('\n');
    }
    if (encoder) opus_multistream_encoder_destroy(encoder);
    if (projection) opus_projection_encoder_destroy(projection);
    for (i=0; i<3; i++) {
        if (decoders[i]) opus_multistream_decoder_destroy(decoders[i]);
        if (projection_decoders[i]) opus_projection_decoder_destroy(projection_decoders[i]);
    }
    /* The 48 kHz, 960-sample mode is static in this reference build. */
    fclose(input);
    free(pcm);
    free(decoded16);
    free(decoded24);
    free(decoded_float);
    return 0;
}
