#!/usr/bin/env python3
"""Regenerate current scalar C fixed math golden values."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference-fixed"


def main() -> None:
    source = REFERENCE / "source"
    binary = REFERENCE / "fixed-math-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DFIXED_POINT", "-DOPUS_BUILD",
        "-I", str(source / "include"), "-I", str(source / "celt"),
        str(ROOT / "tools/reference/fixed_math_vectors.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    (ROOT / "rust/tests/fixtures/reference/fixed-math.txt").write_bytes(subprocess.check_output([str(binary)]))


if __name__ == "__main__":
    main()
