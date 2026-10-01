/* Test-only process oracle. No C code is linked into the Rust library. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_multistream.h"
#include "opus_private.h"

static int unhex(int c) {
    return c <= '9' ? c - '0' : c - 'a' + 10;
}

int main(int argc, char **argv) {
    char line[131072];
    unsigned char input[65536], output[65536];
    FILE *source;
    if (argc != 2 || !(source = fopen(argv[1], "r"))) return 2;
    while (fgets(line, sizeof(line), source)) {
        char operation, *hex;
        int parameter, streams, offset, len = 0, result, i;
        if (sscanf(line, "%c %d %d %n", &operation, &parameter, &streams, &offset) != 3) return 3;
        hex = line + offset;
        while (hex[0] && hex[0] != '\n' && hex[1] && hex[1] != '\n') {
            input[len++] = (unsigned char)(16 * unhex(hex[0]) + unhex(hex[1]));
            hex += 2;
        }
        if (operation == 'p') {
            unsigned char toc;
            const unsigned char *frames[48], *padding;
            opus_int16 sizes[48];
            opus_int32 packet_offset, padding_len;
            int payload_offset;
            result = opus_packet_parse_impl(input, len, parameter, &toc, frames, sizes,
                &payload_offset, &packet_offset, &padding, &padding_len);
            printf("%d", result);
            if (result >= 0) {
                printf(" %u %d %d %d", toc, payload_offset, packet_offset, padding_len);
                for (i = 0; i < result; i++) printf(" %d:%d", (int)(frames[i] - input), sizes[i]);
            }
        } else if (operation == 'e') {
            opus_extension_data extensions[4096];
            opus_int32 count = 4096;
            result = opus_packet_extensions_parse(input, len, extensions, &count, parameter);
            printf("%d", result);
            if (result >= 0) {
                printf(" %d", count);
                for (i = 0; i < count; i++) printf(" %d:%d:%d:%d", extensions[i].id,
                    extensions[i].frame, (int)(extensions[i].data-input), extensions[i].len);
            }
        } else {
            memcpy(output, input, len);
            if (operation == 'r') {
                OpusRepacketizer *rp = opus_repacketizer_create();
                result = opus_repacketizer_cat(rp, input, len);
                if (!result) result = opus_repacketizer_out(rp, output, parameter);
                opus_repacketizer_destroy(rp);
            } else if (operation == 'a') {
                result = opus_packet_pad(output, len, parameter);
                if (!result) result = parameter;
            } else if (operation == 'u') result = opus_packet_unpad(output, len);
            else if (operation == 'm') {
                result = opus_multistream_packet_pad(output, len, parameter, streams);
                if (!result) result = parameter;
            } else if (operation == 'n') result = opus_multistream_packet_unpad(output, len, streams);
            else return 4;
            printf("%d", result);
            if (result >= 0) {
                printf(":");
                for (i = 0; i < result; i++) printf("%02x", output[i]);
            }
        }
        putchar('\n');
    }
    fclose(source);
    return 0;
}
