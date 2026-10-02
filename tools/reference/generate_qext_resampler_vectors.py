#!/usr/bin/env python3
"""Generate exact stateful QEXT SILK resampler vectors from pinned scalar C."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "target/reference-qext/source"
exe = ROOT / "target/reference-qext/qext-resampler-vectors"
subprocess.run([
    "cc", "-O2", "-std=c99", "-DOPUS_BUILD", "-DENABLE_QEXT", "-DVAR_ARRAYS",
    "-ffp-contract=off", "-fno-fast-math",
    *["-I" + str(source / p) for p in ("include", "silk", "celt")],
    str(ROOT / "tools/reference/qext_resampler_vectors.c"),
    str(source / "libopus.a"), "-lm", "-o", str(exe),
], check=True)
output = subprocess.check_output([str(exe)], text=True)
path = ROOT / "rust/tests/fixtures/reference/qext-resampler.tsv"
path.write_text(output)
print(f"Wrote {len(output.splitlines())} stateful vectors to {path}")
