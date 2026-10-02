#!/usr/bin/env python3
"""Regenerate exact stereo-to-mono decoder fixtures from the pinned C oracle."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import pcm_input

ROOT = Path(__file__).resolve().parents[2]
PROFILES = [f"fixed{res}{qext}{pfa}" for res in ("", "-res24")
            for qext in ("", "-qext") for pfa in ("", "-pfa")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    args = parser.parse_args()
    fixtures = ROOT / "rust/tests/fixtures/reference"
    reference = ROOT / f"target/reference-{args.profile}/opus-reference"
    if not reference.is_file():
        subprocess.run(["bash", str(ROOT / "tools/reference/build.sh"), args.profile], check=True)
    rows = []
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("OPUS_ORACLE_")}
    with tempfile.TemporaryDirectory(prefix="opus-downmix-") as directory:
        script = Path(directory) / "input.packets"
        cases = []
        for line in (fixtures / "rfc8251-downmix-packets.tsv").read_text().splitlines():
            name, packet = line.split("\t")
            for rate in (8000, 48000):
                cases.append((name, packet, rate, False))
        if "qext" in args.profile:
            for rate in (48000, 96000):
                frame = rate // 50
                pcm_path = Path(directory) / "input.pcm"
                pcm_path.write_bytes(pcm_input(frame, 2, "mixed"))
                encoded = subprocess.check_output(
                    [str(reference), "codec", str(rate), "2", str(frame), "1",
                     "1002", "768000", str(pcm_path)], text=True,
                    env={**environment, "OPUS_ORACLE_QEXT": "1", "OPUS_ORACLE_VBR": "0"})
                packet = encoded.strip().split("\t")[7]
                assert len(bytes.fromhex(packet)) > 1275
                cases.append((f"generated-qext-{rate}", packet, rate, True))
        for name, packet, rate, inline_packet in cases:
            script.write_text(f"{rate * 120 // 1000} 0 {packet}\n")
            output = subprocess.check_output(
                [str(reference), "decode", str(rate), "1", str(script)],
                text=True, env=environment)
            fields = output.strip().split("\t")
            assert len(fields) == 7 and fields[0] == "D", output
            pcm = bytes.fromhex(fields[5])
            bits = b"".join(int(fields[6][index:index + 8], 16).to_bytes(4, "little")
                            for index in range(0, len(fields[6]), 8))
            row = [name, str(rate), *fields[2:5], hashlib.sha256(pcm).hexdigest(),
                   hashlib.sha256(bits).hexdigest()]
            if inline_packet:
                row.append(packet)
            rows.append("\t".join(row) + "\n")
    destination = fixtures / f"{args.profile}-downmix.tsv"
    destination.write_text("".join(rows))
    print(destination)


if __name__ == "__main__":
    main()
