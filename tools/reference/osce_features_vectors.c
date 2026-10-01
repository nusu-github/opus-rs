/* Independent scalar C feature-extraction fixtures, using the pinned reference. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "osce_features.h"
#include "structs.h"
static void dump(const char *name, int frame, const float *x, int length) {
    int i;printf("%s %d",name,frame);
    for(i=0;i<length;i++){uint32_t bits;memcpy(&bits,x+i,4);printf(" %08x",bits);} puts("");
}
int main(void) {
    static silk_decoder_state decoder;
    silk_decoder_control control={0};OSCEBWEFeatureState bwe={0};
    float features[372],bwe_features[228],bits[2],fade[160],original[160];
    opus_int16 pcm[320],fade_i16[480],original_i16[480];
    int periods[4],frame,i,k;
    for(i=0;i<41;i++)bwe.last_spec[2*i]=1e-9;
    for(frame=0;frame<8;frame++) {
        decoder.nb_subfr=frame%3==1?2:4;decoder.LPC_order=frame%2?10:16;decoder.indices.signalType=frame%3;
        for(i=0;i<320;i++)pcm[i]=(opus_int16)((i*359+frame*311)%60001-30000);
        for(k=0;k<2;k++)for(i=0;i<16;i++)control.PredCoef_Q12[k][i]=(i*73+k*97+frame*31)%701-350;
        for(k=0;k<4;k++){control.pitchL[k]=40+(k*31+frame*7)%160;control.Gains_Q16[k]=17371+k*13791+frame*578;}
        for(i=0;i<20;i++)control.LTPCoef_Q14[i]=(i*719+frame*991)%16001-8000;
        memset(features,0,sizeof(features));memset(periods,0,sizeof(periods));
        osce_calculate_features(&decoder,&control,features,bits,periods,pcm,90+frame*57);
        dump("FEATURES",frame,features,372);dump("BITS",frame,bits,2);
        printf("PERIODS %d",frame);for(k=0;k<4;k++)printf(" %d",periods[k]);puts("");
        memset(bwe_features,0,sizeof(bwe_features));
        osce_bwe_calculate_features(&bwe,bwe_features,pcm,decoder.nb_subfr*80);dump("BWE",frame,bwe_features,228);
    }
    for(i=0;i<160;i++){fade[i]=(i*13%251-125)/128.f;original[i]=(i*17%127-63)/64.f;}
    osce_cross_fade_10ms(fade,original,160);dump("FADE",0,fade,160);
    for(i=0;i<480;i++){fade_i16[i]=i*137%60001-30000;original_i16[i]=i*177%55001-27500;}
    osce_bwe_cross_fade_10ms(fade_i16,original_i16,480);
    printf("FADEI16 0");for(i=0;i<480;i++)printf(" %d",fade_i16[i]);puts("");
    return 0;
}
