#!/usr/bin/env python3
"""Generate independent C QEXT allocation and fine-energy fixtures."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
for profile in ("qext", "fixed-qext"):
    source = ROOT / f"target/reference-{profile}/source"
    executable = ROOT / f"target/reference-{profile}/qext-allocation-vectors"
    flags = ["-O2", "-DOPUS_BUILD", "-DENABLE_QEXT", "-DVAR_ARRAYS", "-ffp-contract=off", "-fno-fast-math"]
    if profile == "fixed-qext":
        flags += ["-DFIXED_POINT"]
    subprocess.run(["cc", *flags, *["-I"+str(source / p) for p in ("include", "silk", "celt", "src")],
                    str(ROOT / "tools/reference/qext_allocation_vectors.c"), str(source / "libopus.a"), "-lm", "-o", str(executable)], check=True)
    output = subprocess.check_output([str(executable)])
    destination = ROOT / f"rust/tests/fixtures/reference/{profile}-allocation.tsv"
    destination.write_bytes(output)
    print(destination)
