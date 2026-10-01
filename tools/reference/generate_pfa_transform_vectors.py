#!/usr/bin/env python3
"""Generate independent exact PFA FFT/MDCT fixtures for float and fixed profiles."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    profile = "fixed-pfa" if args.fixed else "pfa"
    reference = ROOT / f"target/reference-{profile}"
    source = reference / "source"
    if not (source / "libopus.a").is_file():
        subprocess.run(["bash", str(ROOT / "tools/reference/build.sh"), profile], check=True)
    binary = reference / "transform-vectors"
    output = ROOT / "rust/tests/fixtures/reference"
    name = "fixed_transform_vectors.c" if args.fixed else "pfa_transform_vectors.c"
    flags = ["-DFIXED_POINT"] if args.fixed else []
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
        "-fno-fast-math", "-DENABLE_PFA", "-DOPUS_BUILD", *flags,
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference" / name), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary), str(output / f"{profile}-mdct.bin"),
                    str(output / f"{profile}-fft.bin")], check=True)


if __name__ == "__main__":
    main()
