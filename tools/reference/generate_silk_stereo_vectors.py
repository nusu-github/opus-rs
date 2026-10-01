#!/usr/bin/env python3
"""Generate stateful SILK stereo controls and samples with pinned scalar C."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "target/reference/source"
exe = ROOT / "target/reference/silk-stereo-vectors"
subprocess.run(["cc", "-O2", "-std=c99", "-DOPUS_BUILD", "-DVAR_ARRAYS",
               *["-I" + str(source / p) for p in ("include", "silk", "celt")],
               str(ROOT / "tools/reference/silk_stereo_vectors.c"),
               str(source / "libopus.a"), "-lm", "-o", str(exe)], check=True)
output = subprocess.check_output([str(exe)], text=True)
path = ROOT / "rust/tests/fixtures/reference/silk-stereo.tsv"
path.write_text(output)
print(f"Wrote {len(output.splitlines())} stateful vectors to {path}")
