/* Independent scalar Burg cepstral feature fixture. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "freq.h"
int main(void) {
 float pcm[160],features[36];int frame,k;
 for(frame=0;frame<32;frame++) {
  for(k=0;k<160;k++)pcm[k]=((k+frame*160)*137)%30001-15000;
  burg_cepstral_analysis(features,pcm);
  printf("BURG %d",frame);
  for(k=0;k<36;k++){uint32_t v;memcpy(&v,features+k,4);printf(" %08x",v);}puts("");
 }
 return 0;
}
