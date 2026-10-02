/* Q31 comb-filter vectors from the pinned scalar C implementation. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "celt.h"
#include "modes.h"
#include "opus_custom.h"

static void word(FILE *file, uint32_t value) {
    int i;
    for (i=0;i<4;i++) fputc((value >> (8*i)) & 255, file);
}
int main(int argc, char **argv) {
    FILE *file;
    opus_val32 x[2300], y[240];
    int large = argc == 3;
    int history = large ? 2048 : 40, count = large ? 240 : 120;
    int error;
    const OpusCustomMode *mode=opus_custom_mode_create(96000,1920,&error);
#ifdef FIXED_POINT
    const celt_coef window[5] = {107374182,536870912,1073741824,1610612736,1932735232};
    const opus_val16 gains[4][2] = {{0,0},{21299,-11468},{-26214,27853},{16384,0}};
#else
    const celt_coef window[5] = {.05f,.25f,.5f,.75f,.9f};
    const opus_val16 gains[4][2] = {{0,0},{21299/32768.f,-11468/32768.f},{-26214/32768.f,27853/32768.f},{16384/32768.f,0}};
#endif
    int taps0,taps1,gain,inplace,overlap,i;
    if ((argc != 2 && argc != 3) || !(file=fopen(argv[1],"wb"))) return 2;
    for(taps0=0;taps0<3;taps0++) for(taps1=0;taps1<3;taps1++)
    for(gain=0;gain<4;gain++) for(inplace=0;inplace<2;inplace++)
    for(overlap=large?240:0;overlap<=(large?240:5);overlap+=5) {
        for(i=0;i<history+count;i++) x[i]=((i%11)-5)*900+(i%2?200:-200);
        comb_filter(inplace?x+history:y,x+history,18,26,count,gains[gain][0],gains[gain][1],taps0,taps1,large?mode->window:window,overlap,0);
        word(file,taps0);word(file,taps1);word(file,gain);word(file,inplace);word(file,overlap);
        for(i=0;i<count;i++) {
            opus_val32 value = inplace?x[history+i]:y[i];
#ifdef FIXED_POINT
            word(file,(uint32_t)value);
#else
            uint32_t bits; memcpy(&bits,&value,4); word(file,bits);
#endif
        }
    }
    fclose(file);
    return 0;
}
