#!/usr/bin/env python3
"""Generate QEXT comb and preemphasis goldens from pinned scalar C builds."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "503d81b138d76621aae4b12786e90de48aa8db3a"
OUTPUT = ROOT / "rust/tests/fixtures/reference"


def compile_driver(build_name, driver_name, flags):
    source = ROOT / "target" / build_name / "source"
    assert (source / ".reference-revision").read_text().strip() == REVISION
    executable = source.parent / driver_name.replace("_", "-")
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-ffp-contract=off",
        "-fno-fast-math", "-DOPUS_BUILD", "-DENABLE_QEXT", *flags,
        "-I", str(source / "include"), "-I", str(source / "celt"),
        str(ROOT / "tools/reference" / f"{driver_name}.c"),
        str(source / "libopus.a"), "-lm", "-o", str(executable),
    ], check=True)
    return executable


def main():
    fixed = compile_driver("reference-fixed-qext", "qext_comb_vectors", ["-DFIXED_POINT"])
    subprocess.run([str(fixed), str(OUTPUT / "fixed-qext-comb.bin")], check=True)
    subprocess.run([str(fixed), str(OUTPUT / "fixed-qext-comb96.bin"), "96"], check=True)
    floating = compile_driver("reference-qext", "qext_comb_vectors", [])
    subprocess.run([str(floating), str(OUTPUT / "qext-comb96.bin"), "96"], check=True)
    preemphasis = compile_driver("reference-custom-fixed-qext", "qext_preemphasis_vectors",
                                ["-DFIXED_POINT", "-DCUSTOM_MODES"])
    subprocess.run([str(preemphasis), str(OUTPUT / "fixed-qext-preemphasis.bin")], check=True)


if __name__ == "__main__":
    main()
