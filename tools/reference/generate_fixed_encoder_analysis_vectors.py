#!/usr/bin/env python3
"""Generate fixed CELT decision vectors from the pinned scalar C implementation."""
import os
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'target/reference-fixed'
SOURCE = REFERENCE / 'source'
BINARY = REFERENCE / 'fixed-encoder-analysis-vectors'
subprocess.run([
    os.environ.get('CC', 'cc'), '-O2', '-std=c99', '-DFIXED_POINT', '-DOPUS_BUILD', '-DVAR_ARRAYS',
    *[flag for directory in ['include', 'src', 'celt', 'silk'] for flag in ['-I', str(SOURCE / directory)]],
    str(ROOT / 'tools/reference/fixed_encoder_analysis_vectors.c'), str(SOURCE / 'libopus.a'), '-lm', '-o', str(BINARY),
], check=True)
(ROOT / 'rust/tests/fixtures/reference/fixed-encoder-analysis.txt').write_bytes(subprocess.check_output([str(BINARY)]))
