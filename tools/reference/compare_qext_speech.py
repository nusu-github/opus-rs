#!/usr/bin/env python3
"""Compare 96 kHz SILK and hybrid packets, ranges, and PCM against scalar C."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from compare_codec import ROOT, difference, pcm_input, run


def configurations():
    for mode in (1000, 1001):
        bitrate = 24000 if mode == 1000 else 48000
        for channels in (1, 2):
            for vbr in (0, 1):
                for duration in (10, 20, 40, 60, 80, 100, 120):
                    yield mode, channels, vbr, duration, "mixed", bitrate
                for pattern in ("silence", "impulse", "alternating", "noise"):
                    yield mode, channels, vbr, 20, pattern, bitrate
                if mode == 1001:
                    yield mode, channels, vbr, 20, "mixed", 192000


def encoded_fields(output: str) -> str:
    return "\n".join(
        "\t".join(fields[:5] + [fields[4], fields[4], fields[7], "", ""])
        for fields in (line.split("\t") for line in output.splitlines())
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference-qext/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="opus-qext-speech-") as directory:
        pcm = Path(directory) / "input.pcm"
        for mode, channels, vbr, duration, pattern, bitrate in configurations():
            size = 96 * duration
            pcm.write_bytes(pcm_input(size * 3, channels, pattern))
            os.environ.update(OPUS_ORACLE_QEXT="1", OPUS_ORACLE_VBR=str(vbr))
            arguments = ["96000", str(channels), str(size), "3", str(mode), str(bitrate * channels), str(pcm)]
            label = f"{mode}/96000/{channels}/{duration}ms/{pattern}/vbr{vbr}/{bitrate}"
            try:
                expected = run(args.reference.resolve(), arguments)
                actual = run(args.candidate.resolve(), arguments)
                codec_error = difference(expected, actual)
                encode_error = difference(encoded_fields(expected), encoded_fields(actual))
            except RuntimeError as failure:
                codec_error = encode_error = str(failure)
            records.append({"case": label, "encode_error": encode_error, "codec_error": codec_error})
            if codec_error:
                print(f"FAIL {label}: {codec_error}", flush=True)
    encode_failures = sum(record["encode_error"] is not None for record in records)
    codec_failures = sum(record["codec_error"] is not None for record in records)
    print(f"{len(records)} cases; {encode_failures} encode failures; {codec_failures} codec failures")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(records, indent=2) + "\n")
    return int(codec_failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
