/* Scalar neural arithmetic fixtures, independent of any model weights. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "nnet.h"
#include "vec.h"
static void dump(const float *x,int n){int k;for(k=0;k<n;k++){uint32_t b;memcpy(&b,x+k,4);printf("%08x%c",b,k+1==n?'\n':' ');}}
int main(void){
 float w[96],input[12],out[32],scale[16];opus_int8 qw[192];int idx[5]={2,0,8,1,4};int k;
 for(k=0;k<96;k++)w[k]=(k%31-15)*.01234567f;
 for(k=0;k<12;k++)input[k]=(k-6)*.1357911f;
 for(k=0;k<16;k++)scale[k]=(k+1)*.00001234567f;
 for(k=0;k<192;k++)qw[k]=(k*17%255)-127;
 sparse_sgemv8x4(out,w,idx,16,input);dump(out,16);
 sparse_cgemv8x4(out,qw,idx,scale,16,12,input);dump(out,16);
 cgemv8x4(out,qw,scale,16,12,input);dump(out,16);
 for(k=0;k<12;k++)input[k]=(k-6)*2.34567f;
 compute_activation_c(out,input,12,ACTIVATION_EXP);dump(out,12);
 compute_activation_c(out,input,12,ACTIVATION_SOFTMAX);dump(out,12);
 return 0;
}
