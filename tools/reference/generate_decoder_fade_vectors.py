#!/usr/bin/env python3
"""Record the pinned decoder's native integer crossfade and float conversion."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "503d81b138d76621aae4b12786e90de48aa8db3a"

for profile in ("fixed", "fixed-res24", "fixed-qext", "fixed-res24-qext"):
    source = ROOT / "target" / f"reference-{profile}" / "source"
    assert (source / ".reference-revision").read_text().strip() == REVISION
    flags = ["-DFIXED_POINT"]
    if "res24" in profile:
        flags.append("-DENABLE_RES24")
    if "qext" in profile:
        flags.append("-DENABLE_QEXT")
    binary = source.parent / "decoder-fade-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
        "-fno-fast-math", "-DOPUS_BUILD", "-DHAVE_LRINTF", "-DVAR_ARRAYS", *flags,
        *[flag for directory in ("include", "celt", "silk", "src")
          for flag in ("-I", str(source / directory))],
        str(ROOT / "tools/reference/decoder_fade_vectors.c"),
        str(source / "libopus.a"), "-lm", "-o", str(binary),
    ], check=True)
    fixture = ROOT / "rust/tests/fixtures/reference" / f"decoder-fade-{profile}.txt"
    fixture.write_bytes(subprocess.check_output([str(binary)]))
