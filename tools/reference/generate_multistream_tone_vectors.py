#!/usr/bin/env python3
"""Record the low-frequency tone/transient regression through C multistream."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "target/reference"


def main() -> None:
    source = REFERENCE / "source"
    binary = REFERENCE / "multistream-reference"
    subprocess.run([
        os.environ.get("CC", "cc"), "-O2", "-std=c99", "-DOPUS_BUILD",
        "-I", str(source / "include"), "-I", str(source / "src"),
        "-I", str(source / "celt"), "-I", str(source / "silk"),
        str(ROOT / "rust/tests/multistream_reference.c"), str(source / "libopus.a"),
        "-lm", "-o", str(binary),
    ], check=True)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "input.pcm"
        samples = [(((time * (127 + channel*73) + channel*199) % 65536) - 32768) * 211 / 8388608
                   for time in range(240 * 3) for channel in range(2)]
        path.write_bytes(b"".join(struct.pack("<f", sample) for sample in samples))
        lines = []
        for vbr in (0, 1):
            output = subprocess.check_output([str(binary), "streams", "48000", "2", "240", "3",
                                              "f32", "2051", str(vbr), "128000", str(path)], text=True)
            lines.extend(f"{vbr} {line}" for line in output.splitlines())
        (ROOT / "rust/tests/fixtures/reference/multistream-tone.tsv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
