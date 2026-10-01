#!/usr/bin/env python3
"""Generate exact QEXT VQ/cubic fixtures from independently built scalar C."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "rust/tests/fixtures/reference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--res24", action="store_true")
    args = parser.parse_args()
    profile = "fixed-res24-qext" if args.res24 else "fixed-qext" if args.fixed else "qext"
    reference = ROOT / f"target/reference-{profile}"
    source = reference / "source"
    cases = []
    for n in (2, 3, 8, 24):
        for extra in range(2, 15):
            for pattern in range(4):
                values = [((((index * 7919 + pattern * 104729) % 65536) - 32768) * 128)
                          for index in range(n)]
                if pattern == 1:
                    values = [0] * n
                elif pattern == 2:
                    values = [0] * n
                    values[n // 2] = -8388608
                blocks = (1, n if n < 8 else 4, 1, 2 if n % 2 == 0 else 3)[pattern]
                gain = 32768 if pattern % 2 == 0 else 16384
                resynth = int(pattern != 3)
                for pulses in ((1, 3, 5, 32, 128) if n <= 3 else (1, 3, 5)):
                    for tight in (False, True):
                        capacity = max(1, ((n - 1) * (extra + 1) + 2 + 7) // 8) if tight else 128
                        name = f"alg_n{n}_e{extra}_p{pattern}_k{pulses}_tight{int(tight)}"
                        cases.append(f"alg {name} {n} {pulses} {pattern} {blocks} {gain} {resynth} {extra} 128 {capacity} " + " ".join(map(str, values)))
        for resolution in range(1, 15):
            for pattern in range(4):
                values = [((((index * 7919 + pattern * 104729) % 65536) - 32768) * 128)
                          for index in range(n)]
                if pattern == 1:
                    values = [0] * n
                blocks = 1 if pattern % 2 == 0 else (2 if n % 2 == 0 else 3)
                gain = 32768 if pattern < 2 else 16384
                name = f"cubic_n{n}_r{resolution}_p{pattern}"
                cases.append(f"cubic {name} {n} {resolution} 0 {blocks} {gain} {int(pattern != 3)} 0 128 1 " + " ".join(map(str, values)))
    script = OUTPUT / "qext-vq.script"
    contents = "\n".join(cases) + "\n"
    if not script.exists() or script.read_text() != contents:
        script.write_text(contents)
    flags = ["-DFIXED_POINT"] if args.fixed or args.res24 else []
    if args.res24:
        flags.append("-DENABLE_RES24")
    binary = reference / "qext-vq-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DENABLE_QEXT", "-DOPUS_BUILD",
        "-DVAR_ARRAYS", "-DHAVE_LRINTF", "-ffp-contract=off", "-fno-fast-math", *flags,
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/qext_vq_vectors.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    destination = OUTPUT / f"{profile}-vq.txt"
    destination.write_bytes(subprocess.check_output([str(binary), str(script)]))
    print(f"{len(cases)} cases: {destination}")


if __name__ == "__main__":
    main()
