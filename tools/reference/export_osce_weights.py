#!/usr/bin/env python3
"""Export generated Opus neural arrays to a portable little-endian weight blob."""
import argparse
import pathlib
import re
import struct
p=argparse.ArgumentParser(__doc__)
p.add_argument('model_directory',type=pathlib.Path)
p.add_argument('output',type=pathlib.Path)
p.add_argument('--debug-float', action='store_true', help='Retain diagnostic float copies of quantized layers')
p.add_argument('--all-models', action='store_true', help='Include Deep PLC and DRED models alongside OSCE')
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('wb') as out:
    models=('lace','nolace','bbwenet')
    if a.all_models:
        models+=('plc','pitchdnn','fargan','dred_rdovae_enc','dred_rdovae_dec')
    for model in models:
        source=(a.model_directory/f'{model}_data.c').read_text()
        if not a.debug_float:
            source=re.sub(r'(?ms)^#ifndef DISABLE_DEBUG_FLOAT\s*$.*?^#endif\s*/\*\s*DISABLE_DEBUG_FLOAT\s*\*/\s*$', '', source)
        count=0
        for kind,name,size,data in re.findall(r'static const (float|opus_int8|int|qweight) (\w+)\[(\d+)\] = \{(.*?)\};',source,re.S):
            values=[v.strip() for v in data.split(',') if v.strip()]
            assert len(values)==int(size),(name,len(values),size)
            if kind=='float': code,type_id,convert='f',0,float
            elif kind=='int': code,type_id,convert='i',1,int
            else: code,type_id,convert='b',3,int
            payload=struct.pack('<'+code*len(values),*(convert(v.rstrip('f')) for v in values))
            block_size=(len(payload)+63)//64*64
            header=struct.pack('<4siiii44s',b'DNNw',0,type_id,len(payload),block_size,name.encode())
            out.write(header);out.write(payload);out.write(bytes(block_size-len(payload)))
            count+=1
        if not count: raise ValueError(f'No weights in {model}')
print(a.output,a.output.stat().st_size)
