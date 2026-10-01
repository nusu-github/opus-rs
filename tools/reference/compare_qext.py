#!/usr/bin/env python3
"""Compare QEXT encoding and decoding against the pinned scalar C oracle."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import ROOT, difference, pcm_input


def configurations(quick: bool):
    durations = (2500, 5000, 10000, 20000) if quick else (2500, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 120000)
    for rate in (48000, 96000):
        for channels in (1, 2):
            for duration in durations:
                for bitrate in (96000, 192000, 384000, 750000):
                    for vbr in (0, 1):
                        yield rate, channels, duration, bitrate, vbr, "mixed"
    if not quick:
        for rate in (48000, 96000):
            for channels in (1, 2):
                for duration in (2500, 20000, 120000):
                    for pattern in ("silence", "impulse", "alternating", "noise"):
                        for vbr in (0, 1):
                            yield rate, channels, duration, 384000, vbr, pattern


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference-qext/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--record-directory", type=Path)
    args = parser.parse_args()
    failures = count = 0
    with tempfile.TemporaryDirectory(prefix="opus-qext-") as directory:
        pcm = Path(directory) / "input.pcm"
        for rate, channels, duration, bitrate, vbr, pattern in configurations(args.quick):
            count += 1
            frame = rate * duration // 1000000
            pcm.write_bytes(pcm_input(frame * args.frames, channels, pattern))
            command = ["codec", str(rate), str(channels), str(frame), str(args.frames), "1002", str(bitrate * channels), str(pcm)]
            environment = {**os.environ, "OPUS_ORACLE_QEXT": "1", "OPUS_ORACLE_VBR": str(vbr)}
            label = f"qext/{rate}/{channels}/{duration}us/{bitrate}/vbr{vbr}/{pattern}"
            try:
                rows = []
                for executable in (args.reference, args.candidate):
                    result = subprocess.run([str(executable.resolve()), *command], text=True, capture_output=True, env=environment, timeout=120)
                    if result.returncode:
                        raise RuntimeError(f"{executable}: {result.stderr.strip()}")
                    rows.append(result.stdout)
                if args.record_directory:
                    args.record_directory.mkdir(parents=True, exist_ok=True)
                    name = label.replace("/", "-")
                    for suffix, row in zip(("c", "rust"), rows):
                        (args.record_directory / f"{name}.{suffix}.tsv").write_text(row)
                if args.encode_only:
                    for index, row in enumerate(rows):
                        rows[index] = "\n".join("\t".join(fields[:5] + [fields[4], fields[4], fields[7], "", ""]) for fields in (line.split("\t") for line in row.splitlines()))
                error = difference(*rows)
            except (RuntimeError, subprocess.TimeoutExpired) as failure:
                error = str(failure)
            if error:
                failures += 1
                print(f"FAIL {label}: {error}", flush=True)
                if args.fail_fast:
                    break
            else:
                print(f"PASS {label}", flush=True)
    print(f"{count} QEXT cases; {failures} failures")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
