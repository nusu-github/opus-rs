#!/usr/bin/env python3
"""Generate hybrid encoder fixtures using only the pinned scalar C codec."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference"


def main() -> None:
    source = REFERENCE / "source"
    binary = REFERENCE / "hybrid-encode-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
        "-fno-fast-math", "-DOPUS_BUILD", "-I", str(source / "include"),
        "-I", str(source / "src"), "-I", str(source / "celt"),
        "-I", str(source / "silk"), str(ROOT / "tools/reference/hybrid_encode_vectors.c"),
        str(source / "libopus.a"), "-lm", "-o", str(binary),
    ], check=True)
    output = subprocess.check_output([str(binary)], text=True)
    # Keep the descriptive names used by the existing regression tests.
    for frame in (0, 1):
        output = output.replace(f"HP_DELAY{frame}_PACKET", f"HP_DELAY_PACKET{frame}")
        output = output.replace(f"HP_DELAY{frame}_RANGE", f"HP_DELAY_RANGE{frame}")
    (ROOT / "rust/tests/fixtures/hybrid_encode_vectors.rs").write_text(output)


if __name__ == "__main__":
    main()
