#!/usr/bin/env python3
"""Generate exact FFT and MDCT fixtures with the pinned fixed-point C build."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference-fixed"


def main() -> None:
    source = REFERENCE / "source"
    binary = REFERENCE / "fixed-transform-vectors"
    output = ROOT / "rust/tests/fixtures/reference"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
        "-fno-fast-math", "-DFIXED_POINT", "-DOPUS_BUILD",
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/fixed_transform_vectors.c"),
        str(source / "libopus.a"), "-lm", "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary), str(output / "fixed-mdct.bin"), str(output / "fixed-fft.bin")], check=True)


if __name__ == "__main__":
    main()
