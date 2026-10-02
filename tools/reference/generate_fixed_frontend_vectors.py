#!/usr/bin/env python3
"""Generate integer stereo-width and fade vectors from pinned scalar C."""
import os
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
for res24, qext in [(False, False), (True, False), (False, True), (True, True)]:
    suffix = ('-res24' if res24 else '') + ('-qext' if qext else '')
    reference = ROOT / f'target/reference-fixed{suffix}'
    source = reference / 'source'
    binary = reference / 'fixed-frontend-vectors'
    flags = (['-DENABLE_RES24'] if res24 else []) + (['-DENABLE_QEXT'] if qext else [])
    subprocess.run([
        os.environ.get('CC', 'cc'), '-O2', '-std=c99', '-DFIXED_POINT', '-DOPUS_BUILD', '-DVAR_ARRAYS', *flags,
        *[flag for directory in ['include', 'src', 'celt', 'silk'] for flag in ['-I', str(source / directory)]],
        str(ROOT / 'tools/reference/fixed_frontend_vectors.c'), str(source / 'libopus.a'), '-lm', '-o', str(binary),
    ], check=True)
    (ROOT / f'rust/tests/fixtures/reference/fixed-frontend{suffix}.txt').write_bytes(subprocess.check_output([str(binary)]))
