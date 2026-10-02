#!/usr/bin/env python3
"""Compare loaded LACE, NoLACE and BBWENet decoder PCM exactly against scalar C."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from compare_codec import pcm_input
p=argparse.ArgumentParser(__doc__)
p.add_argument('--candidate',default='target/osce-check/debug/opus-rs-oracle')
p.add_argument('--reference',default='target/reference-osce-quantized/opus-reference')
p.add_argument('--weights',default='target/reference/osce-weights-quantized.bin')
p.add_argument('--reference-weights', help='Load this runtime model into a Rust reference oracle; omit for bundled C models')
a=p.parse_args()
failures=0;total=0
with tempfile.TemporaryDirectory(prefix='opus-osce-') as temporary:
    root=Path(temporary)
    for rate in (16000,48000):
      for channels in (1,2):
        for mode in ([1000] if rate==16000 else [1000,1001]):
          frame=rate//50
          source=root/'pcm';source.write_bytes(pcm_input(frame*8,channels,"mixed"))
          cenv=os.environ.copy();cenv.pop('OPUS_ORACLE_DNN_BLOB',None)
          if a.reference_weights: cenv['OPUS_ORACLE_DNN_BLOB']=str(Path(a.reference_weights).resolve())
          rows=subprocess.check_output([a.reference,'codec',str(rate),str(channels),str(frame),'8',str(mode),str(24000*channels),str(source)],env=cenv,text=True).splitlines()
          normal=[f'{frame} 0 {r.split()[7]}' for r in rows]
          for complexity in (0,4,6,7):
            for bwe in ([0] if rate==16000 else [0,1]):
              for loss in (False,True):
                packets=normal[:]
                if loss: packets[3]=f'{frame} 0 -'
                path=root/'packets';path.write_text('\n'.join(packets)+'\n')
                cenv['OPUS_ORACLE_DECODER_COMPLEXITY']=str(complexity)
                cenv['OPUS_ORACLE_OSCE_BWE']=str(bwe)
                renv=cenv.copy();renv['OPUS_ORACLE_DNN_BLOB']=str(Path(a.weights).resolve())
                ref=subprocess.run([a.reference,'decode',str(rate),str(channels),str(path)],env=cenv,capture_output=True,text=True)
                rust=subprocess.run([a.candidate,'decode',str(rate),str(channels),str(path)],env=renv,capture_output=True,text=True)
                name=f'{rate}/{channels}/mode{mode}/complexity{complexity}/bwe{bwe}/loss{int(loss)}'
                good=ref.returncode==rust.returncode==0 and ref.stdout==rust.stdout
                total+=1;failures+=not good
                print(('PASS ' if good else 'FAIL ')+name,flush=True)
                if not good:
                  print('Reference stderr',ref.stderr[:200],'Candidate stderr',rust.stderr[:200],flush=True)
                  (Path('target/reference')/'osce-failed.packets').write_text(path.read_text())
                  (Path('target/reference')/'osce-expected.tsv').write_text(ref.stdout)
                  (Path('target/reference')/'osce-actual.tsv').write_text(rust.stdout)
                  if ref.stdout and rust.stdout:
                    for index,(x,y) in enumerate(zip(ref.stdout.splitlines(),rust.stdout.splitlines())):
                      if x!=y: print('first different frame',index,flush=True);break
    print(f'{total} OSCE decode cases; {failures} failures')
raise SystemExit(bool(failures))
