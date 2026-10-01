/* Independent QEXT allocation and fine-energy vectors from the pinned C source. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "modes.h"
#include "rate.h"
#include "quant_bands.h"
#include "entenc.h"
#include "entdec.h"
static void energy(const celt_glog *v,int n){for(int i=0;i<n;i++){
#ifdef FIXED_POINT
 printf(" %d",v[i]);
#else
 uint32_t bits;memcpy(&bits,&v[i],4);printf(" %08x",bits);
#endif
}putchar('\n');}
int main(void){
 for(int rate=48000;rate<=96000;rate+=48000){int error;CELTMode *m=opus_custom_mode_create(rate,rate/50,&error);if(!m)return 2;CELTMode q;compute_qext_mode(&q,m);
 for(int ch=1;ch<=2;ch++)for(int lm=0;lm<4;lm++)for(int extension=0;extension<2;extension++)for(int budget=0;budget<4;budget++)for(int pattern=0;pattern<3;pattern++){
 int start=pattern==2?5:0,end=m->nbEBands,qe=extension?14:0,total=(int[]){0,64,512,4000}[budget]<<BITRES;
 int storage=(int[]){4,4,16,256}[budget];unsigned char data[256]={0};ec_enc enc;ec_enc_init(&enc,data,storage);int pulses[40]={0},quant[40]={0};celt_glog bands[64],qb[28];
 for(int i=0;i<2*m->nbEBands;i++)bands[i]=GCONST(1.f)*((i*13+pattern*7)%97-55)/4;
 for(int i=0;i<28;i++)qb[i]=GCONST(1.f)*((i*11+pattern*19)%79-45)/4;
 opus_val16 freq=pattern==1?QCONST16(1.7f,13):QCONST16(.4f,13);opus_val32 tone=pattern==1?QCONST32(.99f,29):QCONST32(.25f,29);
 clt_compute_extra_allocation(m,extension?&q:NULL,start,end,qe,bands,qb,total,pulses,quant,ch,lm,&enc,1,freq,tone);
 printf("A %d %d %d %d %d %d %d %u %u\n",rate,ch,lm,qe,start,total,pattern,enc.rng,ec_tell_frac(&enc));
 for(int i=0;i<end+qe;i++)printf("%s%d",i?" ":"",pulses[i]);putchar('\n');for(int i=0;i<end+qe;i++)printf("%s%d",i?" ":"",quant[i]);putchar('\n');ec_enc_done(&enc);for(int i=0;i<storage;i++)printf("%02x",data[i]);putchar('\n');
 }
 if(rate==48000)for(int ch=1;ch<=2;ch++)for(int p=0;p<4;p++)for(int e=0;e<3;e++)for(int size=8;size<=64;size+=56){
 int prev[25],extra[25],base=(int[]){0,3,8,14}[p],bits=(int[]){1,8,14}[e];celt_glog old[50],err[50];unsigned char data[64]={0};ec_enc enc;ec_enc_init(&enc,data,size);
 for(int i=0;i<m->nbEBands;i++){prev[i]=base;extra[i]=bits;}for(int i=0;i<ch*m->nbEBands;i++){old[i]=GCONST(1.f)*(i%17-8)/4;err[i]=GCONST(1.f)*(i%13-6)/16;}
 quant_fine_energy(m,0,m->nbEBands,old,err,prev,extra,&enc,ch);printf("F %d %d %d %d %u %u\n",ch,base,bits,size,enc.rng,ec_tell_frac(&enc));energy(old,ch*m->nbEBands);energy(err,ch*m->nbEBands);ec_enc_done(&enc);for(int i=0;i<size;i++)printf("%02x",data[i]);putchar('\n');
 }
 /* Both modes are immutable static modes in this scalar profile. */
 }
}
