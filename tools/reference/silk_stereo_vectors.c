/* Exact SILK stereo width and predictor transitions from the pinned C source. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "main.h"

int main(void) {
    const int rates[] = {24000,22400,22000,18000,10000,32000,8000,16000,21000,40000,12000,22500};
    for (int fs = 8; fs <= 16; fs += 4) for (int duration = 10; duration <= 20; duration += 10) {
        stereo_enc_state state;
        memset(&state, 0, sizeof(state));
        state.mid_side_amp_Q0[1] = state.mid_side_amp_Q0[3] = 1;
        state.smth_width_Q14 = 16384;
        uint32_t seed = UINT32_C(0x12345678);
        for (int frame = 0; frame < 12; frame++) {
            int n = fs*duration, allocated[2];
            opus_int16 left[322] = {0}, right[322] = {0};
            opus_int8 indices[2][3], mid_only;
            for (int i = 0; i < n; i++) {
                seed = UINT32_C(1664525)*seed + UINT32_C(1013904223); left[i+2] = (opus_int16)(seed>>16);
                seed = UINT32_C(1664525)*seed + UINT32_C(1013904223); right[i+2] = (opus_int16)(seed>>16);
            }
            silk_stereo_LR_to_MS(&state, left+2, right+2, indices, &mid_only, allocated, rates[frame], frame ? 255 : 0, frame==10, fs, n);
            printf("%d\t%d\t%d\t%d\t%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\t", fs,duration,frame,rates[frame],allocated[0],allocated[1],mid_only,state.smth_width_Q14,state.width_prev_Q14,state.pred_prev_Q13[0],state.pred_prev_Q13[1],state.mid_side_amp_Q0[0],state.mid_side_amp_Q0[1],state.mid_side_amp_Q0[2],state.mid_side_amp_Q0[3],indices[0][0],indices[0][1],indices[0][2],indices[1][0],indices[1][1],indices[1][2]);
            for(int i=0;i<n;i++)printf("%s%d",i?",":"",left[i+2]);
            printf("\t");for(int i=0;i<n;i++)printf("%s%d",i?",":"",right[i+1]);printf("\n");
        }
    }
    return 0;
}
