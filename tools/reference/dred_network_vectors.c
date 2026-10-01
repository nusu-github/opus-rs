/* Independent RDOVAE network fixtures from the pinned scalar C implementation. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dred_rdovae_enc.h"
#include "dred_rdovae_dec.h"
extern const WeightArray rdovaeenc_arrays[];
extern const WeightArray rdovaedec_arrays[];
static void dump(const char *name,int frame,const float *x,int count) {
    int i;printf("%s %d",name,frame);
    for(i=0;i<count;i++){uint32_t bits;memcpy(&bits,x+i,4);printf(" %08x",bits);}puts("");
}
int main(void) {
    RDOVAEEnc enc;RDOVAEDec dec;RDOVAEEncState es={0};RDOVAEDecState ds={0};
    float input[40],latent[DRED_LATENT_DIM+1],state[DRED_STATE_DIM],features[80];int frame,k;
    if(init_rdovaeenc(&enc,rdovaeenc_arrays)||init_rdovaedec(&dec,rdovaedec_arrays))return 2;
    for(frame=0;frame<16;frame++) {
        for(k=0;k<40;k++)input[k]=((k*19+frame*23)%127-63)*(1.f/64);
        dred_rdovae_encode_dframe(&es,&enc,latent,state,input,0);
        dump("LATENT",frame,latent,DRED_LATENT_DIM);dump("STATE",frame,state,DRED_STATE_DIM);
        if(frame==0||frame==8)dred_rdovae_dec_init_states(&ds,&dec,state,0);
        latent[DRED_LATENT_DIM]=(frame%16)*.125f-1.f;
        dred_rdovae_decode_qframe(&ds,&dec,features,latent,0);dump("FEATURES",frame,features,80);
    }
    return 0;
}
