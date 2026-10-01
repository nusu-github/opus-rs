/* Independent current-model neural PLC and analysis fixtures. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "lpcnet_private.h"
static void dump(const char *name,int frame,const float *x,int count) {
    int i;printf("%s %d",name,frame);
    for(i=0;i<count;i++){uint32_t bits;memcpy(&bits,x+i,4);printf(" %08x",bits);}puts("");
}
int main(void) {
    static LPCNetPLCState plc;LPCNetEncState enc;
    opus_int16 pcm[160];float features[NB_TOTAL_FEATURES];int frame,k;
    if(lpcnet_plc_init(&plc)||lpcnet_encoder_init(&enc))return 2;
    for(frame=0;frame<64;frame++) {
        for(k=0;k<160;k++)pcm[k]=((k+frame*160)*137)%30001-15000;
        lpcnet_compute_single_frame_features(&enc,pcm,features,0);
        dump("ANALYSIS",frame,features,NB_TOTAL_FEATURES);
        if((frame>=8&&frame<16)||(frame>=20&&frame<23)||(frame>=32&&frame<56))lpcnet_plc_conceal(&plc,pcm);
        else lpcnet_plc_update(&plc,pcm);
        printf("PCM %d",frame);for(k=0;k<160;k++)printf(" %d",pcm[k]);puts("");
        dump("PREDICTION",frame,plc.features,NB_TOTAL_FEATURES);
    }
    return 0;
}
