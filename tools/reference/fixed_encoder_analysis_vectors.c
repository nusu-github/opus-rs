/* Fixed CELT decision stages, evaluated directly by the pinned C source. */
#include <stdio.h>
#include "celt_encoder.c"

static unsigned random_step(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
int main(void) {
    int lm, channels, pattern, allow;
    const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
    for (lm=0;lm<4;lm++) for (channels=1;channels<=2;channels++) for (pattern=0;pattern<5;pattern++) for (allow=0;allow<2;allow++) {
        opus_int32 samples[2160];
        int count=(120<<lm)+120, i, selected=0, weak=0, result;
        opus_val16 estimate=0;
        unsigned seed=12345u+37*pattern+channels;
        for (i=0;i<count*channels;i++) {
            int value=((int)(random_step(&seed)>>9)-4194304);
            if (pattern==0) value=0;
            if (pattern==1 && i%count<count/2) value>>=12;
            if (pattern==2) value=(i%count==count/2) ? 100000000 : 0;
            if (pattern==3) value>>=8;
            samples[i]=value;
        }
        result=transient_analysis(samples,count,channels,&estimate,&selected,allow,&weak,pattern==4?100:500,pattern==4?530000000:0);
        printf("T %d %d %d %d %d %d %d %d\n",lm,channels,pattern,allow,result,estimate,selected,weak);
    }
    for (lm=0;lm<4;lm++) for (channels=1;channels<=2;channels++) for (pattern=0;pattern<8;pattern++) {
        celt_norm x[1920];
        celt_glog energy[42], alternate[42], old[42], surround[21];
        int offsets[21]={0},importance[21]={0},spread[21]={0},tf[21]={0};
        int n=120<<lm,i,trim,select,stereo=-1,patch,vbr;
        opus_int32 boost=0,depth;
        opus_val16 saving=64;
        AnalysisInfo analysis;
        unsigned seed=4321u+57*pattern+channels;
        OPUS_CLEAR(&analysis,1);
        for (i=0;i<n*channels;i++) x[i]=(int)(random_step(&seed)>>9)-4194304;
        for (i=0;i<21*channels;i++) {
            energy[i]=(int)(random_step(&seed)%300000001u)-100000000;
            alternate[i]=energy[i]+(int)(random_step(&seed)%10000001u)-5000000;
            old[i]=energy[i]+(int)(random_step(&seed)%10000001u)-5000000;
        }
        for (i=0;i<21;i++) surround[i]=pattern==7?GCONST(.5f)*(i%4):0;
        depth=dynalloc_analysis(energy,alternate,old,21,0,21,channels,offsets,16,mode->logN,pattern&1,pattern&2,pattern&4,mode->eBands,lm,pattern<2?24:240,&boost,0,surround,&analysis,importance,spread,pattern==6?8192:0,pattern==6?530000000:0);
        select=tf_analysis(mode,21,pattern&1,tf,120,x,n,lm,8192,channels-1,importance);
        trim=alloc_trim_analysis(mode,x,energy,21,lm,channels,n,&analysis,&saving,8192,17,0,64000+pattern*8000,0);
        if (channels==2) stereo=stereo_analysis(mode,x,lm,n);
        patch=patch_transient_decision(energy,old,21,0,21,channels);
        vbr=compute_vbr(mode,&analysis,1024+128*pattern,lm,64000+8000*pattern,21,channels,17,pattern&4,saving,boost,pattern<4?512:8192,0,depth,0,pattern&2,-GCONST(.5f),GCONST(.5f));
        printf("A %d %d %d %d %d %d %d %d %d %d %d",lm,channels,pattern,depth,boost,select,trim,saving,stereo,patch,vbr);
        for (i=0;i<21;i++) printf(" %d %d %d %d",offsets[i],importance[i],spread[i],tf[i]);
        printf("\n");
    }
    return 0;
}
