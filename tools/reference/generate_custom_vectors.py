#!/usr/bin/env python3
"""Generate dynamic-mode tables, packets and decoded PCM from pinned scalar C."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = [(48000, 512), (44100, 480), (32000, 320), (16000, 160),
         (8000, 80), (48000, 960), (48000, 480), (48000, 60), (48000, 100),
         (96000, 960), (22050, 240), (12000, 120), (48000, 768), (48000, 64), (48000, 90)]


def main():
    profile = sys.argv[1] if len(sys.argv) > 1 else "float"
    base_profile = profile.removesuffix("-pfa").removesuffix("-qext")
    if base_profile not in ("float", "fixed", "fixed24"):
        raise SystemExit("Usage: generate_custom_vectors.py float|fixed|fixed24[-qext][-pfa]")
    suffix = "-" + profile if profile != "float" else ""
    reference = ROOT / f"target/reference-custom{suffix}"
    source = reference / "source"
    binary = reference / "opus-custom-reference"
    output = ROOT / f"rust/tests/fixtures/reference/custom{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    flags = ["-DFIXED_POINT"] if base_profile != "float" else []
    if base_profile == "fixed24":
        flags.append("-DENABLE_RES24")
    if "-qext" in profile:
        flags.append("-DENABLE_QEXT")
    if profile.endswith("-pfa"):
        flags.append("-DENABLE_PFA")
    subprocess.run([os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DOPUS_BUILD",
                    "-DCUSTOM_MODES", "-DENABLE_OPUS_CUSTOM_API", "-ffp-contract=off",
                    *flags, "-I", str(source / "include"), "-I", str(source / "celt"),
                    str(ROOT / "tools/reference/custom_oracle.c"), str(source / "libopus.a"),
                    "-lm", "-o", str(binary)], check=True)
    if profile in ("float", "float-qext"):
        (output / "constructors.txt").write_bytes(subprocess.check_output([str(binary), "sweep"]))
    cases = CASES + ([(96000, 1920), (96000, 2048), (96000, 240), (96000, 480), (96000, 1440), (48000, 720)] if "-qext" in profile else [])
    for rate, size in cases:
        command = [str(binary), "mode", str(rate), str(size)]
        mode = subprocess.check_output(command)
        if mode.startswith(b"ERR"):
            raise RuntimeError(f"Unexpected rejected mode {rate}/{size}: {mode!r}")
        (output / f"{rate}-{size}.mode").write_bytes(mode)
        if rate == 96000 and size == 2048:
            continue  # C decoder postfilter reads before its history allocation.
        for channels in (1, 2):
            data = subprocess.check_output([str(binary), "codec", str(rate), str(size),
                                            str(channels), "3", "80"])
            (output / f"{rate}-{size}-{channels}.tsv").write_bytes(data)
    for rate, size, channels in [(48000,90,1),(32000,320,2),(96000,960,1)]:
        data = subprocess.check_output([str(binary), "codec24", str(rate), str(size),str(channels),"3","80"])
        (output / f"{rate}-{size}-{channels}-24.tsv").write_bytes(data)
    if "-qext" in profile:
        for rate, size in [(48000,960),(96000,1920),(48000,720),(96000,1440)]:
            for channels in (1,2):
                data = subprocess.check_output([str(binary), "codecext", str(rate), str(size), str(channels), "3", "800"])
                (output / f"{rate}-{size}-{channels}-ext.tsv").write_bytes(data)
    (output / "mode-cases.txt").write_text("".join(f"{rate} {size}\n" for rate, size in cases))
    (output / "cases.txt").write_text("".join(f"{rate} {size}\n" for rate, size in cases if (rate, size) != (96000, 2048)))


if __name__ == "__main__":
    main()
