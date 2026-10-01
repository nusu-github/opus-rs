#!/usr/bin/env python3
"""Record fixed CELT tone decisions directly from the pinned C implementation."""
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference-fixed"
OUTPUT = ROOT / "rust/tests/fixtures/reference"


def main() -> None:
    cases = []
    def case(op, name, values, a=0, b=0):
        cases.append(f"{op} {name} {len(values)} {a} {b} " + " ".join(map(str, values)))
    for n in (120, 240, 480, 960):
        for amplitude in (0, 1, 127, 1023, 16383, 32767):
            values = [amplitude if i % 3 else -amplitude for i in range(n)]
            case("normalize", f"normalize_{n}_{amplitude}", values)
        for delay in (1, 2, 4, 8, 16, 32):
            for pattern in (0, 1, 2, 3):
                values = [
                    0 if pattern == 0 else 1000 if pattern == 1 else
                    int(1000 * math.sin(i * .131)) if pattern == 2 else
                    ((i * 1664525 + 1013904223) & 2047) - 1024
                    for i in range(n)
                ]
                case("lpc", f"lpc_{n}_{delay}_{pattern}", values, delay)
        for channels in (1, 2):
            for frequency in (0, 100, 440, 1000, 6000, 14000, 24000):
                values = [int((1 << 26) * math.cos(2 * math.pi * frequency * i / 48000 + c * .25))
                          for c in range(channels) for i in range(n)]
                case("detect", f"detect_{n}_{channels}_{frequency}", values, channels, 48000)
            for pattern in (0, 1, 2):
                values = [0 if pattern == 0 else
                          ((i * 1664525 + 1013904223) & 134217727) - 67108864 if pattern == 1 else
                          (1 << 27) if i % n == 3 else 0
                          for i in range(n * channels)]
                case("detect", f"detect_pattern_{n}_{channels}_{pattern}", values, channels, 48000)
    one = 1 << 29
    values = [-one + i * (one // 64) for i in range(129)]
    values += [-one + ((i * 1664525 + 1013904223) & ((1 << 30) - 1)) for i in range(1024)]
    values += [-one + 1, -one + 32767, -1, 0, 1, one - 32767, one - 1]
    case("acos", "acos_domain", values)
    script = OUTPUT / "fixed-tone.script"
    script.write_text("\n".join(cases) + "\n")
    source = REFERENCE / "source"
    binary = REFERENCE / "fixed-tone-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DFIXED_POINT", "-DOPUS_BUILD", "-DVAR_ARRAYS",
        "-ffp-contract=off", "-fno-fast-math", "-fno-tree-vectorize",
        "-I", str(source / "include"), "-I", str(source / "src"), "-I", str(source / "celt"),
        "-I", str(source / "silk"), "-I", str(source / "silk/fixed"),
        str(ROOT / "tools/reference/fixed_tone_vectors.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    script.with_suffix(".txt").write_bytes(subprocess.check_output([str(binary), str(script)]))


if __name__ == "__main__":
    main()
