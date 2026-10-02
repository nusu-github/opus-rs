/* Tonality analysis fixtures from the pinned scalar C implementation. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "analysis.c"

static uint32_t bits(float value) {
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}
static uint32_t sample_bits(opus_val32 value) {
#ifdef FIXED_POINT
    return (uint32_t)value;
#else
    return bits(value);
#endif
}
static uint32_t hash_word(uint32_t hash, uint32_t value) {
    int i;
    for (i=0;i<4;i++) { hash ^= (value>>(8*i))&255; hash *= 16777619u; }
    return hash;
}
int main(void) {
    const int rates[3]={16000,24000,48000};
    const CELTMode *mode=opus_custom_mode_create(48000,960,NULL);
    int rate,channels,format,pattern,frame,i;
    for (rate=0;rate<3;rate++) for(channels=1;channels<=2;channels++) for(format=0;format<3;format++) for(pattern=0;pattern<3;pattern++) {
        TonalityAnalysisState state;
        uint32_t seed=12345+channels*17+pattern*91;
        tonality_analysis_init(&state,rates[rate]);
        for(frame=0;frame<12;frame++) {
            opus_int16 pcm16[1920]; opus_int32 pcm24[1920]; float pcmf[1920];
            AnalysisInfo info;
            uint32_t hash=2166136261u;
            int n=rates[rate]/50;
            for(i=0;i<n*channels;i++) {
                int value;
                seed=seed*1664525u+1013904223u;
                value=(int)(seed>>16)-32768;
                if(pattern==1) value=((i+frame*n*channels)*137%65536)-32768;
                if(pattern==2 && frame>=5) value=0;
                pcm16[i]=value; pcm24[i]=value*256+(int)((seed>>8)&255); pcmf[i]=pcm24[i]*(1.f/8388608);
            }
            OPUS_CLEAR(&info,1);
            run_analysis(&state,mode,format==0?(const void*)pcm16:format==1?(const void*)pcm24:(const void*)pcmf,n,n,0,channels==2?1:-1,channels,rates[rate],24,format==0?downmix_int:format==1?downmix_int24:downmix_float,&info);
            for(i=0;i<720;i++) hash=hash_word(hash,sample_bits(state.inmem[i]));
            for(i=0;i<3;i++) hash=hash_word(hash,sample_bits(state.downmix_state[i]));
            printf("A %d %d %d %d %d %d %d",rates[rate],channels,format,pattern,frame,info.valid,info.bandwidth);
            printf(" %u %u %u %u %u %u %u %u %u %u",bits(info.tonality),bits(info.tonality_slope),bits(info.noisiness),bits(info.activity),bits(info.music_prob),bits(info.music_prob_min),bits(info.music_prob_max),bits(info.activity_probability),bits(info.max_pitch_ratio),bits(state.hp_ener_accum));
            for(i=0;i<19;i++) printf(" %u",(unsigned)info.leak_boost[i]);
            printf(" %u\n",hash);
        }
    }
#ifndef FIXED_POINT
    {
        float input[12]={NAN,INFINITY,-INFINITY,3.f,-3.f,2.f,-2.f,1.f,-1.f,.5f,-.5f,-0.f};
        float output[12];
        for(channels=1;channels<=3;channels++) {
            downmix_float(input,output,12/channels,0,0,-2,channels);
            printf("D %d",channels);
            for(i=0;i<12/channels;i++) printf(" %u",bits(output[i]));
            printf("\n");
        }
    }
#endif
    return 0;
}
