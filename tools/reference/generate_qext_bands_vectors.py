#!/usr/bin/env python3
"""Generate complete QEXT band quantization state from pinned scalar C."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--res24", action="store_true")
    args = parser.parse_args()
    profile = "fixed-res24-qext" if args.res24 else "fixed-qext" if args.fixed else "qext"
    reference = ROOT / f"target/reference-{profile}"
    source = reference / "source"
    flags = ["-DFIXED_POINT"] if args.fixed or args.res24 else []
    if args.res24:
        flags.append("-DENABLE_RES24")
    binary = reference / "qext-bands-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DENABLE_QEXT", "-DOPUS_BUILD",
        "-DVAR_ARRAYS", "-DHAVE_LRINTF", "-ffp-contract=off", "-fno-fast-math", *flags,
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/qext_bands_vectors.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    destination = ROOT / f"rust/tests/fixtures/reference/{profile}-bands.bin"
    destination.write_bytes(subprocess.check_output([str(binary)]))
    print(f"288 cases: {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
