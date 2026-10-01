#!/usr/bin/env python3
"""Extract canonical scalar LPCNet FFT, window and DCT tables from pinned C."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[2]
source=(root/'dnn/lpcnet_tables.c').read_text()
def array(name):
    match=re.search(r'\b'+name+r'\[[^\]]*\]\s*=\s*\{(.*?)\};',source,re.S)
    if not match: raise ValueError(name)
    return match.group(1)
output=['// Generated from dnn/lpcnet_tables.c; regenerate with tools/reference/extract_dnn_tables.py.', '#![allow(clippy::excessive_precision)]','use crate::celt::{KissFftCpx, KissFftState};']
for c,rust,size,kind in [('fft_bitrev','BITREV',320,'usize'),('half_window','HALF_WINDOW',160,'f32'),('dct_table','DCT_TABLE',324,'f32')]:
    values=[v.strip().rstrip('f') for v in array(c).split(',') if v.strip()]
    assert len(values)==size
    output+=[f'pub(crate) const {rust}: [{kind};{size}] = [']
    output += ['    '+', '.join(values[k:k+8])+',' for k in range(0,size,8)]
    output+= ['];']
complexes=re.findall(r'\{\s*([^,{}]+),\s*([^,{}]+)\}',array('fft_twiddles'))
assert len(complexes)==320
output += ['const TWIDDLES: [KissFftCpx;320] = [']
for real,imag in complexes: output += [f'    KissFftCpx {{ r: {real.strip().rstrip("f")}, i: {imag.strip().rstrip("f")} }},']
output += ['];','pub(crate) fn fft() -> KissFftState { KissFftState::from_static(320, 0.0031250000, None, &[5,64,4,16,4,4,4,1], &BITREV, &TWIDDLES) }']
(root/'rust/src/dnn_tables.rs').write_text('\n'.join(output)+'\n')
