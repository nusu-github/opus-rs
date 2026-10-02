/* Independent scalar OSCE neural-network oracle; tests only. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "osce.c"
static void dump(const char *name, int frame, const float *x, int length) {
  int i; printf("%s %d",name,frame);
  for(i=0;i<length;i++){uint32_t bits;memcpy(&bits,x+i,4);printf(" %08x",bits);} puts("");
}
int main(void) {
  OSCEModel model;
  LACEState lace_state;
  NoLACEState nolace_state;
  BBWENetState bwe_state;
  float input[320],features[372],bwe_features[228],bits[2],output[960];
  int periods[4],frame,k;
  if(osce_load_models(&model,NULL,0)) return 2;
  reset_lace_state(&lace_state);reset_nolace_state(&nolace_state);reset_bbwenet_state(&bwe_state);
  for(frame=0;frame<5;frame++) {
    for(k=0;k<320;k++)input[k]=((k*37+frame*101)%1024-512)*(1.f/2048);
    for(k=0;k<372;k++)features[k]=((k*13+frame*17)%127-63)*(1.f/64);
    for(k=0;k<228;k++)bwe_features[k]=((k*19+frame*23)%127-63)*(1.f/64);
    for(k=0;k<4;k++)periods[k]=40+(k*31+frame*7)%160;
    bits[0]=20+frame*200;bits[1]=7+frame*31;
    lace_process_20ms_frame(&model.lace,&lace_state,output,input,features,bits,periods,0);dump("LACE",frame,output,320);
    nolace_process_20ms_frame(&model.nolace,&nolace_state,output,input,features,bits,periods,0);dump("NOLACE",frame,output,320);
    bbwenet_process_frames(&model.bbwenet,&bwe_state,output,input,bwe_features,2,0);dump("BWE",frame,output,960);
  }
  return 0;
}
