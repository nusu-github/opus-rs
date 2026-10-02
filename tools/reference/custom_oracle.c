/* Test-only numerical oracle for dynamically constructed CELT modes. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opus_custom.h"
#include "modes.h"
#include "mdct.h"
#include "kiss_fft.h"

static unsigned bits(float value) { unsigned result; memcpy(&result, &value, 4); return result; }

int main(int argc, char **argv) {
    int rate, size, channels, frames, bytes, error, i, frame;
#ifdef ENABLE_QEXT
    const int max_size = 2048;
#else
    const int max_size = 1024;
#endif
    OpusCustomMode *mode;
    if (argc == 2 && !strcmp(argv[1], "sweep")) {
        const int rates[] = {7999,8000,11025,12000,16000,22050,24000,32000,40000,44100,48000,88200,96000,96001};
        unsigned r;
        for (r=0;r<sizeof(rates)/sizeof(rates[0]);r++) {
            for (size=0;size<=1030;size++) {
                int factor=size/2;
                if(factor>0 && size>=40 && size<=max_size && !(size&1)) {
                    while(!(factor%2))factor/=2;
                    while(!(factor%3))factor/=3;
                    while(!(factor%5))factor/=5;
                    /* C's allocation-failure cleanup double-frees some
                       unsupported FFT sizes. Test these rejects in Rust
                       separately, without treating a C crash as a golden. */
                    if(factor!=1)continue;
                }
                mode=opus_custom_mode_create(rates[r],size,&error);
                printf("%d %d %d\n",rates[r],size,mode!=NULL);
                if(mode) opus_custom_mode_destroy(mode);
            }
        }
        return 0;
    }
    if (argc < 4) return 2;
    rate = atoi(argv[2]); size = atoi(argv[3]);
    mode = opus_custom_mode_create(rate, size, &error);
    if (!mode) { printf("ERR %d\n", error); return 0; }
    if (!strcmp(argv[1], "mode")) {
        printf("M %d %d %d %d %d %d\n", mode->overlap, mode->nbEBands, mode->effEBands, mode->maxLM, mode->nbShortMdcts, mode->shortMdctSize);
        printf("B"); for (i=0;i<=mode->nbEBands;i++) printf(" %d",mode->eBands[i]); putchar('\n');
        printf("L"); for (i=0;i<mode->nbEBands;i++) printf(" %d",mode->logN[i]); putchar('\n');
        #ifdef FIXED_POINT
        printf("W"); for (i=0;i<mode->overlap;i++) printf(" %d",mode->window[i]); putchar('\n');
        printf("P"); for (i=0;i<4;i++) printf(" %d",mode->preemph[i]); putchar('\n');
#else
        printf("W"); for (i=0;i<mode->overlap;i++) printf(" %08x",bits(mode->window[i])); putchar('\n');
        printf("P"); for (i=0;i<4;i++) printf(" %08x",bits(mode->preemph[i])); putchar('\n');
#endif
        printf("A"); for (i=0;i<mode->nbEBands*mode->nbAllocVectors;i++) printf(" %d",mode->allocVectors[i]); putchar('\n');
#ifdef ENABLE_QEXT
        if(mode->qext_cache.index) {
            CELTMode ext; compute_qext_mode(&ext,mode);
            printf("QB"); for(i=0;i<=ext.nbEBands;i++)printf(" %d",ext.eBands[i]);putchar('\n');
            printf("QL"); for(i=0;i<ext.nbEBands;i++)printf(" %d",ext.logN[i]);putchar('\n');
            printf("QI"); for(i=0;i<ext.nbEBands*(ext.maxLM+2);i++)printf(" %d",ext.cache.index[i]);putchar('\n');
            printf("QT"); for(i=0;i<ext.cache.size;i++)printf(" %d",ext.cache.bits[i]);putchar('\n');
            printf("QC"); for(i=0;i<ext.nbEBands*(ext.maxLM+1)*2;i++)printf(" %d",ext.cache.caps[i]);putchar('\n');
        }
#endif
        opus_custom_mode_destroy(mode);
        return 0;
    }
    if ((strcmp(argv[1], "codec") && strcmp(argv[1], "codec24") && strcmp(argv[1], "codecext")) || argc != 7) return 2;
    channels=atoi(argv[4]);frames=atoi(argv[5]);bytes=atoi(argv[6]);
    {
        OpusCustomEncoder *encoder=opus_custom_encoder_create(mode,channels,&error);
        OpusCustomDecoder *decoder=opus_custom_decoder_create(mode,channels,&error);
        OpusCustomDecoder *decoder_float=opus_custom_decoder_create(mode,channels,&error);
        OpusCustomDecoder *decoder_24=opus_custom_decoder_create(mode,channels,&error);
        opus_int16 input[4096], output[4096];
        float output_float[4096];
        opus_int32 output_24[4096], input_24[4096];
        unsigned char packet[1276];
        if(!encoder || !decoder || bytes>1276 || size*channels>4096) return 3;
        opus_custom_encoder_ctl(encoder,OPUS_SET_COMPLEXITY(10));
#ifdef ENABLE_QEXT
        if(!strcmp(argv[1],"codecext")) opus_custom_encoder_ctl(encoder,OPUS_SET_QEXT(1));
#endif
        /* Dynamic QEXT modes can exceed the unscaled PLC history. The C
           periodic PLC then reads before its allocation; do not record UB. */
        for(frame=0;frame<frames+(size>1024 && !(rate==96000 && (mode->shortMdctSize==240 || mode->shortMdctSize==180)) ? 0 : 2);frame++) {
            opus_uint32 enc_range,dec_range;
            int count,length;
            for(i=0;i<size*channels;i++) input[i]=(opus_int16)(((i+frame*size*channels)*127+811)%24001-12000);
            for(i=0;i<size*channels;i++) input_24[i]=(opus_int32)input[i]*256+((i+frame*size*channels)*23%255)-127;
            length=frame>=frames ? 0 : !strcmp(argv[1],"codec24") ? opus_custom_encode24(encoder,input_24,size,packet,bytes) : opus_custom_encode(encoder,input,size,packet,bytes);
            if(length<0){printf("ENCERR %d\n",length);return 4;}
            count=opus_custom_decode(decoder,length?packet:NULL,length,output,size);
            if(count<0){printf("DECERR %d\n",count);return 5;}
            opus_custom_encoder_ctl(encoder,OPUS_GET_FINAL_RANGE(&enc_range));
            opus_custom_decoder_ctl(decoder,OPUS_GET_FINAL_RANGE(&dec_range));
            printf("E %d %d %u %u ",frame,length,enc_range,dec_range);
            if(!length) putchar('-');
            for(i=0;i<length;i++)printf("%02x",packet[i]);
            putchar(' ');
            for(i=0;i<count*channels;i++)printf("%02x%02x",(unsigned short)output[i]&255,((unsigned short)output[i]>>8)&255);
            if(opus_custom_decode_float(decoder_float,length?packet:NULL,length,output_float,size)!=count) return 6;
            if(opus_custom_decode24(decoder_24,length?packet:NULL,length,output_24,size)!=count) return 7;
            putchar(' ');
            for(i=0;i<count*channels;i++)printf("%08x",bits(output_float[i]));
            putchar(' ');
            for(i=0;i<count*channels;i++)printf("%08x",(unsigned)output_24[i]);
            putchar('\n');
        }
        opus_custom_decoder_destroy(decoder_float);opus_custom_decoder_destroy(decoder_24);
        opus_custom_encoder_destroy(encoder);opus_custom_decoder_destroy(decoder);
    }
    opus_custom_mode_destroy(mode);
    return 0;
}
