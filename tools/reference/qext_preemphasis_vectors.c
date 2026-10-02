/* Independent fixed QEXT preemphasis vectors, including custom coefficients. */
#include <stdint.h>
#include <stdio.h>
#include "celt.h"
#include "modes.h"
#include "opus_custom.h"
static void word(FILE *file, uint32_t value) {
    int i; for(i=0;i<4;i++) fputc((value>>(8*i))&255,file);
}
int main(int argc,char **argv) {
    const int rates[]={8000,12000,16000,24000,32000,48000,96000};
    FILE *file;
    int rate,up,cc,i,error;
    if(argc!=2 || !(file=fopen(argv[1],"wb")))return 2;
    for(rate=0;rate<7;rate++) {
        OpusCustomMode *mode=opus_custom_mode_create(rates[rate],rates[rate]/100,&error);
        if(!mode || error)return 3;
        for(up=1;up<=3;up++) for(cc=1;cc<=2;cc++) {
            opus_res input[384];
            celt_sig output[192],mem=-123456;
            for(i=0;i<384;i++)input[i]=INT16TORES((opus_int16)((i*631+811)%60001-30000));
            celt_preemphasis(input,output,192,cc,up,mode->preemph,&mem,0);
            word(file,rates[rate]);word(file,up);word(file,cc);
            for(i=0;i<192;i++)word(file,(uint32_t)output[i]);
            word(file,(uint32_t)mem);
        }
        opus_custom_mode_destroy(mode);
    }
    fclose(file);return 0;
}
