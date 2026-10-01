/* Independent DRED entropy, activity, offset, and quantizer fixtures. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dred_encoder.h"
#include "dred_decoder.h"
static void dump(const char *name,int frame,const float *x,int count) {
    int i;printf("%s %d",name,frame);
    for(i=0;i<count;i++){uint32_t bits;memcpy(&bits,x+i,4);printf(" %08x",bits);}puts("");
}
int main(void) {
    static DREDEnc enc;OpusDRED dec;unsigned char packet[250],activity[DRED_MAX_FRAMES*4];
    int test,i,n,ret;
    for(test=0;test<48;test++) {
        memset(&enc,0,sizeof(enc));memset(&dec,0,sizeof(dec));
        enc.latents_buffer_fill=5+test%20;enc.latent_offset=test%3;enc.dred_offset=test%13;
        enc.last_extra_dred_offset=test%4==0?2:0;
        for(i=0;i<DRED_MAX_FRAMES*DRED_LATENT_DIM;i++)enc.latents_buffer[i]=((i*19+test*23)%127-63)*(1.f/16);
        for(i=0;i<DRED_MAX_FRAMES*DRED_STATE_DIM;i++)enc.state_buffer[i]=((i*17+test*29)%127-63)*(1.f/16);
        for(i=0;i<DRED_MAX_FRAMES*4;i++)activity[i]=test%6==0?0:((i+test*3)%47<16);
        n=dred_encode_silk_frame(&enc,packet,1+test%12,8+test%6*40,test%16,test%8,15,activity,0);
        printf("PACKET %d %d %d",test,n,enc.last_extra_dred_offset);for(i=0;i<n;i++)printf(" %02x",packet[i]);puts("");
        if(n) {
            ret=dred_ec_decode(&dec,packet,n,40,test%7);
            printf("DECODE %d %d %d %d",test,ret,dec.dred_offset,dec.process_stage);puts("");
            dump("STATE",test,dec.state,DRED_STATE_DIM);
            dump("LATENTS",test,dec.latents,dec.nb_latents*(DRED_LATENT_DIM+1));
        }
    }
    return 0;
}
