#!/usr/bin/env python3
"""Generate periodic PLC state hashes and full packet-loss decode vectors in C."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import pcm_input

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "rust/tests/fixtures/reference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfa", action="store_true")
    parser.add_argument("--res24", action="store_true")
    parser.add_argument("--qext", action="store_true")
    args = parser.parse_args()
    profile = "fixed" + ("-res24" if args.res24 else "") + ("-qext" if args.qext else "") + ("-pfa" if args.pfa else "")
    reference = ROOT / "target" / f"reference-{profile}"
    stem = profile + "-plc"
    source = reference / "source"
    assert (source / ".reference-revision").read_text().strip() == "503d81b138d76621aae4b12786e90de48aa8db3a"
    binary = reference / "fixed-plc-vectors"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DFIXED_POINT", "-DOPUS_BUILD",
        "-DVAR_ARRAYS", "-DENABLE_OPUS_CUSTOM_API", "-ffp-contract=off", "-fno-fast-math",
        *(["-DENABLE_PFA"] if args.pfa else []),
        *(["-DENABLE_RES24"] if args.res24 else []),
        *(["-DENABLE_QEXT"] if args.qext else []),
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "tools/reference/fixed_plc_vectors.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    (OUTPUT / f"{stem}-state.txt").write_bytes(subprocess.check_output([str(binary), str(OUTPUT / "fixed-plc-prime.script")]))
    oracle = reference / "opus-reference"
    with tempfile.TemporaryDirectory() as temporary:
        pcm = Path(temporary) / "input.pcm"
        for rate, channels, duration, pattern in [(8000,1,5000,"mixed"), (24000,1,5000,"mixed"),
                                                  (24000,2,10000,"mixed"), (48000,1,20000,"noise")]:
            size = rate * duration // 1000000
            pcm.write_bytes(pcm_input(size * 5, channels, pattern))
            encoded = subprocess.check_output([str(oracle), "codec", str(rate), str(channels), str(size),
                                               "5", "1002", str(48000 * channels), str(pcm)], text=True)
            packets = [line.split("\t")[7] for line in encoded.splitlines()]
            script = OUTPUT / f"{stem}-{rate}-{channels}-{size}.packets"
            script.write_text("".join(f"{size} 0 {packet if i != 1 else '-'}\n" for i, packet in enumerate(packets)))
            script.with_suffix(".tsv").write_bytes(subprocess.check_output([str(oracle), "decode", str(rate), str(channels), str(script)]))


if __name__ == "__main__":
    main()
