#!/usr/bin/env python3
"""Regenerate all inherited fixed VQ scenarios against the current Q24 C profile."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference-fixed"
OUTPUT = ROOT / "rust/tests/fixtures/reference"
ONE = 2147483647


def main() -> None:
    cases = []
    def case(op, name, values, a=0, b=0, c=0, d=0, e=0):
        cases.append(f"{op} {name} {len(values)} {a} {b} {c} {d} {e} " + " ".join(map(str, values)))
    case("normalise", "normalise_case1", [1,-1,1,0,0,0,0,0], 3, ONE)
    case("normrot", "band0_decode_case", [-1,1,0,-1,0,1,0,-1], 5, ONE, -1, 8, 5)
    mixed = [12000,-8000,6000,-4000,2000,0,-1000,500]
    case("rotate", "rotation_forward", mixed, 1, 2, 3, 2)
    case("rotate", "rotation_inverse", [9459,-9458,9459,0,0,0,0,0], -1, 2, 3, 2)
    # Even when the butterfly has no pairs, C rounds the norm through Q14.
    case("rotate1", "rotation_singleton", [630784 + 181], 1, 30000, 12000)
    case("rotate1", "rotation_full_stride", [-4249600 + 181, 496640 + 513], 2, 30000, 12000)
    for name, values, gain in [
        ("zero", [0]*8, ONE),
        ("mixed", [1000,-2000,3000,-4000,500,-600,700,-800], ONE),
        ("large", [30000,-30000,20000,-10000], ONE),
        ("half_gain", [30000,-30000,20000,-10000], 1073741824),
    ]: case("renorm", name, values, gain)
    long = [5000,-4000,3000,-2000,1000,-500,250,-125,6000,-5000,4000,-3000,2000,-1000,500,-250]
    for name, values, k, spread, blocks, gain, resynth in [
        ("alg1", mixed, 3,2,2,ONE,1),
        ("alg1_no_resynth", mixed, 3,2,2,ONE,0),
        ("alg2", [0,16000,-16000,8000,-8000,4000,0,2000,-2000,0], 3,0,5,ONE,1),
        ("alg3", [8000,-4000,2000,-1000], 2,2,1,ONE,1),
        ("alg4", [12000,-8000], 1,3,1,ONE,1),
        ("alg5", long, 3,1,2,ONE,1),
        ("alg_b8", long, 3,2,8,ONE,1),
        ("alg6", [0,10000,-9000,8000,-7000,6000,-5000,4000,-3000,2000,-1000,500,-250,125,-60,30], 3,3,2,1073741824,1),
        ("alg7", [16000,8000,4000,2000,1000,500,250,125], 6,0,2,ONE,1),
        ("alg8", [6000,-6000]*4, 8,0,4,ONE,1),
        ("alg9", [0,0,0,0,0,0,12000,-8000], 3,0,4,ONE,1),
        ("alg_q24", [v*1024 for v in mixed], 9,2,2,ONE,1),
    ]: case("alg", name, values, k, spread, blocks, gain, resynth)
    for name, values, k in [
        ("search_silence", [0]*8, 6),
        ("search_bigk", [0]*2, 10),
        ("search_small", [100,-50,25,-12,6,-3,2,-1], 6),
        ("search_mixed", [11429,-8383,6429,-4217,1993,-209,-975,534], 3),
    ]: case("search", name, values, k)
    script = OUTPUT / "fixed-vq.script"
    script.write_text("\n".join(cases) + "\n")
    source = REFERENCE / "source"
    binary = REFERENCE / "fixed-vq-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DFIXED_POINT", "-DOPUS_BUILD", "-DVAR_ARRAYS",
        "-I", str(source / "include"), "-I", str(source / "src"), "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/fixed_vq_vectors.c"), str(source / "libopus.a"), "-lm", "-o", str(binary),
    ], check=True)
    script.with_suffix(".txt").write_bytes(subprocess.check_output([str(binary), str(script)]))
    bands_binary = REFERENCE / "fixed-bands-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DFIXED_POINT", "-DOPUS_BUILD", "-DVAR_ARRAYS",
        "-I", str(source / "include"), "-I", str(source / "src"), "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/fixed_bands_vectors.c"), str(source / "libopus.a"), "-lm", "-o", str(bands_binary),
    ], check=True)
    (OUTPUT / "fixed-bands.txt").write_bytes(subprocess.check_output([str(bands_binary)]))


if __name__ == "__main__": main()
