/* Test-only process oracle. Never linked to or invoked by the Rust library. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_private.h"
#include "entenc.h"
#include "entdec.h"

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void hex(const unsigned char *data, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) printf("%02x", data[i]);
}

static void pcm16_hex(const opus_int16 *data, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) {
        uint16_t value = (uint16_t)data[i];
        printf("%02x%02x", value & 255, value >> 8);
    }
}

static void float_hex(const float *data, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) {
        uint32_t value;
        memcpy(&value, &data[i], sizeof(value));
        printf("%08x", value);
    }
}

static unsigned char *unhex(const char *text, size_t *len) {
    size_t i;
    unsigned char *data;
    *len = strlen(text) / 2;
    if (strlen(text) % 2) fail("Odd hexadecimal input length.");
    data = (unsigned char *)calloc(*len + 1, 1);
    if (!data) fail("Allocation failed.");
    for (i = 0; i < *len; ++i) {
        unsigned value;
        if (sscanf(text + 2 * i, "%2x", &value) != 1) fail("Invalid hexadecimal input.");
        data[i] = (unsigned char)value;
    }
    return data;
}

static void packet(const char *text, int rate) {
    size_t len;
    unsigned char *data = unhex(text, &len), toc = 0;
    const unsigned char *frames[48];
    opus_int16 sizes[48];
    int offset = 0, i;
    int count = opus_packet_parse(data, (opus_int32)len, &toc, frames, sizes, &offset);
    printf("P\t%d\t%u\t%d\t%d\t%d\t%d\t%d\t%d\n", count, toc, offset,
        len ? opus_packet_get_bandwidth(data) : OPUS_BAD_ARG,
        len ? opus_packet_get_nb_channels(data) : OPUS_BAD_ARG,
        len ? opus_packet_get_samples_per_frame(data, rate) : OPUS_BAD_ARG,
        opus_packet_get_nb_frames(data, (opus_int32)len),
        opus_packet_get_nb_samples(data, (opus_int32)len, rate));
    for (i = 0; i < count; ++i) printf("F\t%ld\t%d\n", (long)(frames[i] - data), sizes[i]);
    free(data);
}

typedef struct { char name[16]; unsigned a, b, c; } Operation;
static const unsigned char icdf[] = {192, 128, 64, 0};
static const opus_uint16 icdf16[] = {30000, 20000, 10000, 0};

static void entropy(const char *path) {
    FILE *file = fopen(path, "r");
    char line[128];
    Operation operations[4096];
    unsigned size, count = 0, i;
    unsigned char *data;
    ec_enc enc;
    ec_dec dec;
    if (!file) fail("Cannot open entropy script.");
    if (!fgets(line, sizeof(line), file) || sscanf(line, "size %u", &size) != 1 || size > 1048576 || !size)
        fail("An entropy script must start with size N (1 through 1048576).");
    data = (unsigned char *)calloc(size, 1);
    if (!data) fail("Allocation failed.");
    while (fgets(line, sizeof(line), file)) {
        Operation op;
        memset(&op, 0, sizeof(op));
        if (line[0] == '#' || line[0] == '\n') continue;
        if (count >= 4096 || sscanf(line, "%15s %u %u %u", op.name, &op.a, &op.b, &op.c) < 2)
            fail("Invalid entropy operation.");
        operations[count++] = op;
    }
    fclose(file);
    ec_enc_init(&enc, data, size);
    for (i = 0; i < count; ++i) {
        Operation *op = &operations[i];
        if (!strcmp(op->name, "uint")) ec_enc_uint(&enc, op->a, op->b);
        else if (!strcmp(op->name, "bits")) ec_enc_bits(&enc, op->a, op->b);
        else if (!strcmp(op->name, "bit")) ec_enc_bit_logp(&enc, op->a, op->b);
        else if (!strcmp(op->name, "icdf")) ec_enc_icdf(&enc, op->a, icdf, 8);
        else if (!strcmp(op->name, "icdf16")) ec_enc_icdf16(&enc, op->a, icdf16, 15);
        else if (!strcmp(op->name, "encode")) ec_encode(&enc, op->a, op->b, op->c);
        else if (!strcmp(op->name, "bin")) ec_encode_bin(&enc, op->a, op->b, op->c);
        else if (!strcmp(op->name, "patch")) ec_enc_patch_initial_bits(&enc, op->a, op->b);
        else if (!strcmp(op->name, "shrink")) ec_enc_shrink(&enc, op->a);
        else fail("Unknown entropy operation.");
        printf("E\t%u\t%d\t%u\t%u\t%d\n", i, ec_tell(&enc), ec_tell_frac(&enc), enc.rng, enc.error);
    }
    ec_enc_done(&enc);
    printf("B\t%u\t%d\t", enc.storage, enc.error);
    hex(data, enc.storage);
    putchar('\n');
    ec_dec_init(&dec, data, enc.storage);
    for (i = 0; i < count; ++i) {
        Operation *op = &operations[i];
        unsigned value;
        if (!strcmp(op->name, "uint")) value = ec_dec_uint(&dec, op->b);
        else if (!strcmp(op->name, "bits")) value = ec_dec_bits(&dec, op->b);
        else if (!strcmp(op->name, "bit")) value = ec_dec_bit_logp(&dec, op->b);
        else if (!strcmp(op->name, "icdf")) value = ec_dec_icdf(&dec, icdf, 8);
        else if (!strcmp(op->name, "icdf16")) value = ec_dec_icdf16(&dec, icdf16, 15);
        else if (!strcmp(op->name, "encode")) {
            value = ec_decode(&dec, op->c);
            ec_dec_update(&dec, op->a, op->b, op->c);
        } else if (!strcmp(op->name, "bin")) {
            value = ec_decode_bin(&dec, op->c);
            ec_dec_update(&dec, op->a, op->b, 1U << op->c);
        } else continue;
        printf("D\t%u\t%u\t%d\t%u\t%u\t%d\n", i, value, ec_tell(&dec), ec_tell_frac(&dec), dec.rng, dec.error);
    }
    free(data);
}

static void ctl_ok(int error) {
    if (error != OPUS_OK) fail(opus_strerror(error));
}

static int codec_env_int(const char *name, int default_value, int minimum, int maximum) {
    const char *value = getenv(name);
    char *end;
    long parsed;
    if (!value) return default_value;
    parsed = strtol(value, &end, 10);
    if (!*value || *end || parsed < minimum || parsed > maximum)
        fail("Invalid codec environment setting.");
    return (int)parsed;
}

static void configure_decoder(OpusDecoder *decoder) {
    const char *path = getenv("OPUS_ORACLE_DNN_BLOB");
    ctl_ok(opus_decoder_ctl(decoder, OPUS_SET_COMPLEXITY(codec_env_int("OPUS_ORACLE_DECODER_COMPLEXITY", 0, 0, 10))));
#ifdef ENABLE_OSCE
    ctl_ok(opus_decoder_ctl(decoder, OPUS_SET_OSCE_BWE(codec_env_int("OPUS_ORACLE_OSCE_BWE", 0, 0, 1))));
#endif
    if (path) {
        FILE *file = fopen(path, "rb");
        long length;
        unsigned char *bytes;
        if (!file || fseek(file, 0, SEEK_END)) fail("Cannot open DNN model blob.");
        length = ftell(file);
        if (length <= 0 || length > 64*1024*1024 || fseek(file, 0, SEEK_SET)) fail("Invalid DNN model blob length.");
        bytes = (unsigned char *)malloc((size_t)length);
        if (!bytes || fread(bytes,1,(size_t)length,file)!=(size_t)length) fail("Cannot read DNN model blob.");
        fclose(file);
        ctl_ok(opus_decoder_ctl(decoder, OPUS_SET_DNN_BLOB(bytes,(int)length)));
        /* The C model retains pointers into the blob for the decoder's lifetime. */
    }
}

static void codec(int argc, char **argv) {
    int rate, channels, frame_size, frame_count, mode, bitrate, error, f, i;
    FILE *input;
    OpusEncoder *enc;
    OpusDecoder *dec, *float_dec;
    opus_int16 *pcm, *decoded;
    opus_int32 *pcm24;
    float *pcm_float;
    float *decoded_float;
    unsigned char encoded[3826 * 6];
    if (argc != 9) fail("codec RATE CHANNELS FRAME_SIZE FRAME_COUNT MODE BITRATE INPUT_I16LE");
    rate = atoi(argv[2]); channels = atoi(argv[3]); frame_size = atoi(argv[4]);
    frame_count = atoi(argv[5]); mode = atoi(argv[6]); bitrate = atoi(argv[7]);
    if (frame_size < 1 || frame_size > rate * 120 / 1000 || frame_count < 1 || frame_count > 1000 || channels < 1 || channels > 2)
        fail("Invalid codec dimensions.");
    input = fopen(argv[8], "rb");
    if (!input) fail("Cannot open PCM input.");
    enc = opus_encoder_create(rate, channels, codec_env_int("OPUS_ORACLE_APPLICATION", OPUS_APPLICATION_AUDIO, 2048, 2051), &error);
    ctl_ok(error);
    dec = opus_decoder_create(rate, channels, &error); ctl_ok(error);
    float_dec = opus_decoder_create(rate, channels, &error); ctl_ok(error);
    configure_decoder(dec); configure_decoder(float_dec);
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate)));
#ifdef ENABLE_QEXT
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_QEXT(codec_env_int("OPUS_ORACLE_QEXT", 0, 0, 1))));
#endif
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_VBR(codec_env_int("OPUS_ORACLE_VBR", 0, 0, 1))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(codec_env_int("OPUS_ORACLE_COMPLEXITY", 10, 0, 10))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(codec_env_int("OPUS_ORACLE_FEC", 0, 0, 1))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(codec_env_int("OPUS_ORACLE_LOSS", 0, 0, 100))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_DTX(codec_env_int("OPUS_ORACLE_DTX", 0, 0, 1))));
    if (getenv("OPUS_ORACLE_DRED_DURATION")) ctl_ok(opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION(codec_env_int("OPUS_ORACLE_DRED_DURATION", 0, 0, 104))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_VBR_CONSTRAINT(codec_env_int("OPUS_ORACLE_CVBR", 1, 0, 1))));
    ctl_ok(opus_encoder_ctl(enc, OPUS_SET_SIGNAL(codec_env_int("OPUS_ORACLE_SIGNAL", OPUS_AUTO, OPUS_AUTO, OPUS_SIGNAL_MUSIC))));
    if (mode != OPUS_AUTO) ctl_ok(opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(mode)));
    if (mode == MODE_SILK_ONLY) ctl_ok(opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(rate >= 16000 ? OPUS_BANDWIDTH_WIDEBAND : rate >= 12000 ? OPUS_BANDWIDTH_MEDIUMBAND : OPUS_BANDWIDTH_NARROWBAND)));
    if (mode == MODE_HYBRID) ctl_ok(opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(rate >= 48000 ? OPUS_BANDWIDTH_FULLBAND : OPUS_BANDWIDTH_SUPERWIDEBAND)));
    pcm = (opus_int16 *)calloc((size_t)frame_size * channels, sizeof(*pcm));
    pcm24 = (opus_int32 *)calloc((size_t)frame_size * channels, sizeof(*pcm24));
    pcm_float = (float *)calloc((size_t)frame_size * channels, sizeof(*pcm_float));
    decoded = (opus_int16 *)calloc((size_t)rate * 120 / 1000 * channels, sizeof(*decoded));
    decoded_float = (float *)calloc((size_t)rate * 120 / 1000 * channels, sizeof(*decoded_float));
    if (!pcm || !pcm24 || !pcm_float || !decoded || !decoded_float) fail("Allocation failed.");
    if (!strcmp(argv[1], "codec_transition")) {
        ctl_ok(opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC(1)));
        ctl_ok(opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC(15)));
    }
    for (f = 0; f < frame_count; ++f) {
        if (f == codec_env_int("OPUS_ORACLE_RESET_AT", -1, -1, frame_count)) ctl_ok(opus_encoder_ctl(enc, OPUS_RESET_STATE));
        int bytes, samples, float_samples;
        opus_uint32 enc_range, dec_range, float_range;
        if (!strcmp(argv[1], "codec_transition")) {
            static const int modes[] = {MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY};
            mode = modes[(f / 4) % 5];
            ctl_ok(opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE(mode)));
            ctl_ok(opus_encoder_ctl(enc, OPUS_SET_BITRATE((mode == MODE_SILK_ONLY ? 24000 : 48000) * channels)));
            ctl_ok(opus_encoder_ctl(enc, OPUS_SET_BANDWIDTH(mode == MODE_SILK_ONLY ? OPUS_BANDWIDTH_WIDEBAND : rate >= 48000 ? OPUS_BANDWIDTH_FULLBAND : OPUS_BANDWIDTH_SUPERWIDEBAND)));
        }
        for (i = 0; i < frame_size * channels; ++i) {
            int lo = fgetc(input), hi = fgetc(input);
            if (lo == EOF || hi == EOF) fail("PCM input is shorter than requested.");
            pcm[i] = (opus_int16)((unsigned)lo | (unsigned)hi << 8);
            if (!strcmp(argv[1], "codec_float") || !strcmp(argv[1], "codec24")) {
                int b2 = fgetc(input), b3 = fgetc(input);
                uint32_t word;
                if (b2 == EOF || b3 == EOF) fail("PCM input is shorter than requested.");
                word = (uint32_t)lo | (uint32_t)hi << 8 | (uint32_t)b2 << 16 | (uint32_t)b3 << 24;
                pcm24[i] = (opus_int32)word;
                memcpy(&pcm_float[i], &word, sizeof(word));
            }
        }
        if (!strcmp(argv[1], "codec_float"))
            bytes = opus_encode_float(enc, pcm_float, frame_size, encoded, sizeof(encoded));
        else if (!strcmp(argv[1], "codec24"))
            bytes = opus_encode24(enc, pcm24, frame_size, encoded, sizeof(encoded));
        else
            bytes = opus_encode(enc, pcm, frame_size, encoded, sizeof(encoded));
        if (bytes < 0) fail(opus_strerror(bytes));
        samples = opus_decode(dec, encoded, bytes, decoded, rate * 120 / 1000, 0);
        if (samples < 0) fail(opus_strerror(samples));
        float_samples = opus_decode_float(float_dec, encoded, bytes, decoded_float, rate * 120 / 1000, 0);
        if (samples != float_samples) fail("Decoder sample counts differ.");
        ctl_ok(opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&enc_range)));
        ctl_ok(opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&dec_range)));
        ctl_ok(opus_decoder_ctl(float_dec, OPUS_GET_FINAL_RANGE(&float_range)));
        printf("C\t%d\t%d\t%d\t%u\t%u\t%u\t", f, bytes, samples, enc_range, dec_range, float_range);
        hex(encoded, (size_t)bytes); putchar('\t');
        pcm16_hex(decoded, (size_t)samples * channels); putchar('\t');
        float_hex(decoded_float, (size_t)samples * channels); putchar('\n');
    }
    if (fgetc(input) != EOF) fail("PCM input is longer than requested.");
    fclose(input);
    free(pcm); free(pcm24); free(pcm_float); free(decoded); free(decoded_float);
    opus_encoder_destroy(enc); opus_decoder_destroy(dec); opus_decoder_destroy(float_dec);
}

static void decode_packets(int argc, char **argv) {
    int rate, channels, error, frame = 0;
    FILE *input;
    OpusDecoder *dec, *float_dec;
    OpusDecoder *dec24 = NULL;
    opus_int32 *pcm24 = NULL;
    OpusDREDDecoder *dred_decoder = NULL;
    OpusDRED *dred = NULL;
    int dred_mode = !strcmp(argv[1], "dred_decode");
    opus_int16 *pcm;
    float *float_pcm;
    char line[32768], text[32000];
    if (argc != 5) fail("decode RATE CHANNELS PACKETS_FILE");
    rate = atoi(argv[2]); channels = atoi(argv[3]);
    input = fopen(argv[4], "r");
    if (!input) fail("Cannot open packet input.");
    dec = opus_decoder_create(rate, channels, &error); ctl_ok(error);
    float_dec = opus_decoder_create(rate, channels, &error); ctl_ok(error);
    configure_decoder(dec); configure_decoder(float_dec);
    if (dred_mode) {
        dred_decoder = opus_dred_decoder_create(&error); ctl_ok(error);
        dred = opus_dred_alloc(&error); ctl_ok(error);
        dec24 = opus_decoder_create(rate, channels, &error); ctl_ok(error);
        configure_decoder(dec24);
        pcm24 = (opus_int32 *)calloc((size_t)rate * 120 / 1000 * channels, sizeof(*pcm24));
        if (!pcm24) fail("Allocation failed.");
    }
    pcm = (opus_int16 *)calloc((size_t)rate * 120 / 1000 * channels, sizeof(*pcm));
    float_pcm = (float *)calloc((size_t)rate * 120 / 1000 * channels, sizeof(*float_pcm));
    if (!pcm || !float_pcm) fail("Allocation failed.");
    while (fgets(line, sizeof(line), input)) {
        int frame_size, fec, samples, float_samples, samples24 = 0;
        size_t len = 0;
        unsigned char *packet = NULL;
        opus_uint32 range, float_range;
        if (!strncmp(line, "reset", 5)) {
            ctl_ok(opus_decoder_ctl(dec, OPUS_RESET_STATE));
            ctl_ok(opus_decoder_ctl(float_dec, OPUS_RESET_STATE));
            if (dec24) ctl_ok(opus_decoder_ctl(dec24, OPUS_RESET_STATE));
            continue;
        }
        if (sscanf(line, "gain %d", &frame_size) == 1) {
            ctl_ok(opus_decoder_ctl(dec, OPUS_SET_GAIN(frame_size)));
            ctl_ok(opus_decoder_ctl(float_dec, OPUS_SET_GAIN(frame_size)));
            if (dec24) ctl_ok(opus_decoder_ctl(dec24, OPUS_SET_GAIN(frame_size)));
            continue;
        }
        if (sscanf(line, "%d %d %31999s", &frame_size, &fec, text) != 3) fail("Invalid packet input line.");
        if (frame_size < 1 || frame_size > rate * 120 / 1000) fail("Invalid decode frame size.");
        if (strcmp(text, "-")) packet = unhex(text, &len);
        if (dred_mode && fec >= 0) {
            int dred_end = 0;
            int defer = codec_env_int("OPUS_ORACLE_DRED_DEFER", 0, 0, 1);
            int available = opus_dred_parse(dred_decoder, dred, packet, (opus_int32)len, rate * 104 / 100, rate, &dred_end, defer);
            if (available <= 0) fail("Recovery packet contains no decodable DRED redundancy.");
            if (defer) ctl_ok(opus_dred_process(dred_decoder, dred, dred));
            printf("R\t%d\t%d\t%d\n", frame, available, dred_end);
            samples = opus_decoder_dred_decode(dec, dred, fec, pcm, frame_size);
            float_samples = opus_decoder_dred_decode_float(float_dec, dred, fec, float_pcm, frame_size);
            samples24 = opus_decoder_dred_decode24(dec24, dred, fec, pcm24, frame_size);
        } else {
            samples = opus_decode(dec, packet, (opus_int32)len, pcm, frame_size, dred_mode ? 0 : fec);
            float_samples = opus_decode_float(float_dec, packet, (opus_int32)len, float_pcm, frame_size, dred_mode ? 0 : fec);
            if (dec24) samples24 = opus_decode24(dec24, packet, (opus_int32)len, pcm24, frame_size, 0);
        }
        free(packet);
        if (samples != float_samples) fail("Decoder return values differ.");
        ctl_ok(opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE(&range)));
        ctl_ok(opus_decoder_ctl(float_dec, OPUS_GET_FINAL_RANGE(&float_range)));
        printf("D\t%d\t%d\t%u\t%u\t", frame++, samples, range, float_range);
        if (samples > 0) pcm16_hex(pcm, (size_t)samples * channels);
        putchar('\t');
        if (samples > 0) float_hex(float_pcm, (size_t)samples * channels);
        putchar('\n');
        if (dec24) {
            int i;
            opus_uint32 range24;
            if (samples24 != samples) fail("24-bit decoder return value differs.");
            ctl_ok(opus_decoder_ctl(dec24, OPUS_GET_FINAL_RANGE(&range24)));
            printf("Q\t%d\t%d\t%u\t", frame - 1, samples24, range24);
            for (i = 0; i < samples24 * channels; i++) printf("%08x", (unsigned)pcm24[i]);
            putchar('\n');
        }
    }
    fclose(input);
    free(pcm); free(float_pcm);
    opus_decoder_destroy(dec); opus_decoder_destroy(float_dec);
    if (dec24) opus_decoder_destroy(dec24);
    free(pcm24);
    if (dred) opus_dred_free(dred);
    if (dred_decoder) opus_dred_decoder_destroy(dred_decoder);
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "packet")) packet(argv[2], atoi(argv[3]));
    else if (argc == 3 && !strcmp(argv[1], "entropy")) entropy(argv[2]);
    else if (argc >= 2 && (!strcmp(argv[1], "codec") || !strcmp(argv[1], "codec_float") || !strcmp(argv[1], "codec24") || !strcmp(argv[1], "codec_transition"))) codec(argc, argv);
    else if (argc >= 2 && (!strcmp(argv[1], "decode") || !strcmp(argv[1], "dred_decode"))) decode_packets(argc, argv);
    else fail("Usage: opus-reference packet HEX RATE | entropy SCRIPT | codec RATE CHANNELS FRAME_SIZE FRAME_COUNT MODE BITRATE INPUT_I16LE");
    return 0;
}
