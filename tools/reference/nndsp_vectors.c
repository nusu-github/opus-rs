/* Independent C adaptive-DSP fixtures; compile against the pinned source tree. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nndsp.h"

static LinearLayer layer(float *weights, float *bias, int inputs, int outputs) {
    LinearLayer result = {0};
    int i;
    for (i=0;i<inputs*outputs;i++) weights[i] = (float)((i*17+3)%61-30)/256.f;
    for (i=0;i<outputs;i++) bias[i] = (float)(i%7-3)/16.f;
    result.float_weights=weights; result.bias=bias;
    result.nb_inputs=inputs; result.nb_outputs=outputs;
    return result;
}
static void dump(const float *values, int count) {
    int i;
    for(i=0;i<count;i++) {
        uint32_t bits; memcpy(&bits,values+i,4);
        printf("%08x%c",bits,i+1==count?'\n':' ');
    }
}
int main(void) {
    float kw[240], kb[30], gw[24], gb[3];
    float ckw[40], ckb[5], cgw[8], cgb[1], cggw[8], cggb[1];
    float a1fw[320],a1fb[20],a1tw[440],a1tb[20],a2w[800],a2b[20];
    LinearLayer k=layer(kw,kb,8,30), g=layer(gw,gb,8,3);
    LinearLayer ck=layer(ckw,ckb,8,5), cg=layer(cgw,cgb,8,1), cgg=layer(cggw,cggb,8,1);
    LinearLayer a1f=layer(a1fw,a1fb,16,20),a1t=layer(a1tw,a1tb,22,20),a2=layer(a2w,a2b,40,20);
    AdaConvState conv; AdaCombState comb; AdaShapeState shape;
    float window[10], input[80], features[8], output[120];
    int frame,i;
    init_adaconv_state(&conv);init_adacomb_state(&comb);init_adashape_state(&shape);
    compute_overlap_window(window,10);dump(window,10);
    for(frame=0;frame<4;frame++) {
        for(i=0;i<80;i++) input[i]=(float)(((i+frame*40)*13)%251-125)/128.f;
        for(i=0;i<8;i++) features[i]=(float)(((i+frame)*7)%31-15)/16.f;
        adaconv_process_frame(&conv,output,input,features,&k,&g,8,40,10,2,3,5,4,0.5f,-0.2f,1.f,window,0);dump(output,120);
        adacomb_process_frame(&comb,output,input,features,&ck,&cg,&cgg,32+frame*3,8,40,10,5,2,0.5f,-0.2f,0.1f,window,0);dump(output,40);
        adashape_process_frame(&shape,output,input,features,&a1f,&a1t,&a2,8,40,4,2,0);dump(output,40);
    }
    return 0;
}
