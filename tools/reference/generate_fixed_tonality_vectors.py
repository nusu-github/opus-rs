#!/usr/bin/env python3
"""Generate tonality analysis vectors from the pinned scalar C implementation."""
import argparse
import os
from pathlib import Path
import subprocess
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--float', action='store_true', dest='floating')
parser.add_argument('--pfa', action='store_true')
args = parser.parse_args()
profile = ('pfa' if args.floating else 'fixed-pfa') if args.pfa else ('float' if args.floating else 'fixed')
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / ('target/reference' if profile == 'float' else f'target/reference-{profile}')
SOURCE = REFERENCE / 'source'
BINARY = REFERENCE / 'fixed-tonality-vectors'
subprocess.run([
    os.environ.get('CC', 'cc'), '-O2', '-ffp-contract=off', '-std=c99',
    *([] if args.floating else ['-DFIXED_POINT']), '-DOPUS_BUILD', '-DVAR_ARRAYS', *(['-DENABLE_PFA'] if args.pfa else []),
    *[flag for directory in ['include', 'src', 'celt', 'silk'] for flag in ['-I', str(SOURCE / directory)]],
    str(ROOT / 'tools/reference/fixed_tonality_vectors.c'), str(SOURCE / 'libopus.a'), '-lm', '-o', str(BINARY),
], check=True)
(ROOT / 'rust/tests/fixtures/reference' / f'{profile}-tonality.txt').write_bytes(subprocess.check_output([str(BINARY)]))
