/* Independent in-process scalar C benchmark; never linked into Rust. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opus.h"
#include "opus_private.h"

#define CORPUS 128
#define WARMUP 64
#define CAPACITY 24576

static void check(int result) {
    if (result < 0) { fprintf(stderr, "Opus error: %d\n", result); exit(2); }
}
static uint64_t now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}
static uint64_t cpu_time(void) {
    unsigned long long value = 0;
    FILE *file = fopen("/proc/thread-self/schedstat", "r");
    if (!file || fscanf(file, "%llu", &value) != 1) exit(2);
    fclose(file);
    return value;
}
static void word(uint32_t value) {
    unsigned char bytes[4] = { value, value >> 8, value >> 16, value >> 24 };
    fwrite(bytes, 1, 4, stdout);
}
int main(int argc, char **argv) {
    int rate, channels, frame, mode, bitrate, complexity, iterations, stride, error, i, size, loss;
    uint64_t start, elapsed, cpu_start, cpu_elapsed, checksum = 0;
    opus_uint32 range, ranges[CORPUS];
    OpusEncoder *encoder;
    OpusDecoder *decoder;
    FILE *file;
    opus_int16 *input, *output;
    unsigned char *packet, *packets[CORPUS];
    int lengths[CORPUS];
    if (argc != 10) return 2;
    rate = atoi(argv[2]); channels = atoi(argv[3]); frame = atoi(argv[4]);
    mode = atoi(argv[5]); bitrate = atoi(argv[6]); complexity = atoi(argv[7]);
    iterations = atoi(argv[8]); stride = frame * channels;
    loss = getenv("OPUS_BENCH_LOSS_EVERY") ? atoi(getenv("OPUS_BENCH_LOSS_EVERY")) : 0;
    input = calloc(CORPUS * stride, sizeof(*input)); output = calloc(stride, sizeof(*output));
    packet = calloc(CAPACITY, 1);
    if (!input || !output || !packet) return 2;
    file = fopen(argv[9], "rb");
    if (!file) return 2;
    for (i = 0; i < CORPUS * stride; i++) {
        int lo = fgetc(file), hi = fgetc(file);
        if (lo < 0 || hi < 0) return 2;
        input[i] = (opus_int16)(lo | hi << 8);
    }
    fclose(file);
    encoder = opus_encoder_create(rate, channels, OPUS_APPLICATION_AUDIO, &error); check(error);
    check(opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate)));
    check(opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(complexity)));
    check(opus_encoder_ctl(encoder, OPUS_SET_FORCE_MODE(mode)));
    check(opus_encoder_ctl(encoder, OPUS_SET_VBR(1)));
    check(opus_encoder_ctl(encoder, OPUS_SET_VBR_CONSTRAINT(1)));
#ifdef ENABLE_QEXT
    check(opus_encoder_ctl(encoder, OPUS_SET_QEXT(1)));
#endif
    if (mode == MODE_SILK_ONLY)
        check(opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(rate >= 16000 ? OPUS_BANDWIDTH_WIDEBAND : rate >= 12000 ? OPUS_BANDWIDTH_MEDIUMBAND : OPUS_BANDWIDTH_NARROWBAND)));
    if (mode == MODE_HYBRID)
        check(opus_encoder_ctl(encoder, OPUS_SET_BANDWIDTH(rate >= 48000 ? OPUS_BANDWIDTH_FULLBAND : OPUS_BANDWIDTH_SUPERWIDEBAND)));
    for (i = 0; i < WARMUP; i++) check(opus_encode(encoder, input + i * stride, frame, packet, CAPACITY));
    if (argv[1][0] == 'e') {
        cpu_start = cpu_time(); start = now();
        for (i = 0; i < iterations; i++) {
            size = opus_encode(encoder, input + (i % CORPUS) * stride, frame, packet, CAPACITY); check(size);
            checksum += (uint64_t)size + packet[0];
        }
        elapsed = now() - start;
        cpu_elapsed = cpu_time() - cpu_start;
        check(opus_encoder_ctl(encoder, OPUS_GET_FINAL_RANGE(&range)));
    } else {
        for (i = 0; i < CORPUS; i++) {
            size = opus_encode(encoder, input + i * stride, frame, packet, CAPACITY); check(size);
            lengths[i] = size; packets[i] = malloc(size);
            if (!packets[i]) return 2;
            for (int k = 0; k < size; k++) packets[i][k] = packet[k];
            check(opus_encoder_ctl(encoder, OPUS_GET_FINAL_RANGE(&ranges[i])));
        }
        decoder = opus_decoder_create(rate, channels, &error); check(error);
        for (i = 0; i < WARMUP; i++) check(opus_decode(decoder, loss && (i + 1) % loss == 0 ? NULL : packets[i], loss && (i + 1) % loss == 0 ? 0 : lengths[i], output, frame, 0));
        if (argv[1][0] == 'v') {
            for (i = 0; i < CORPUS; i++) {
                size = opus_decode(decoder, loss && (i + 1) % loss == 0 ? NULL : packets[i], loss && (i + 1) % loss == 0 ? 0 : lengths[i], output, frame, 0); check(size);
                check(opus_decoder_ctl(decoder, OPUS_GET_FINAL_RANGE(&range)));
                word(lengths[i]); fwrite(packets[i], 1, lengths[i], stdout); word(ranges[i]); word(size);
                for (int k = 0; k < size * channels; k++) {
                    unsigned char bytes[2] = { (uint16_t)output[k], (uint16_t)output[k] >> 8 };
                    fwrite(bytes, 1, 2, stdout);
                }
                word(range);
            }
            return 0;
        }
        cpu_start = cpu_time(); start = now();
        for (i = 0; i < iterations; i++) {
            int k = i % CORPUS, lost = loss && (i + 1) % loss == 0;
            size = opus_decode(decoder, lost ? NULL : packets[k], lost ? 0 : lengths[k], output, frame, 0); check(size);
            checksum += (uint64_t)size + (uint16_t)output[0];
        }
        elapsed = now() - start;
        cpu_elapsed = cpu_time() - cpu_start;
        check(opus_decoder_ctl(decoder, OPUS_GET_FINAL_RANGE(&range)));
        opus_decoder_destroy(decoder);
        for (i = 0; i < CORPUS; i++) free(packets[i]);
    }
    printf("{\"elapsed_ns\":%llu,\"cpu_ns\":%llu,\"iterations\":%d,\"checksum\":%llu,\"final_range\":%u}\n", (unsigned long long)elapsed, (unsigned long long)cpu_elapsed, iterations, (unsigned long long)checksum, range);
    opus_encoder_destroy(encoder); free(input); free(output); free(packet);
    return 0;
}
