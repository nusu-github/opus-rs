#!/usr/bin/env python3
"""Exercise DTX activity transitions, application/signal hints, and VBR policies."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from compare_codec import MODES, ROOT, difference, pcm_input, run


def edge_configurations():
    configurations = []
    for mode in MODES:
        for channels in (1, 2):
            for pattern in ("silence", "activity"):
                configurations.append((mode, channels, 2049, -1000, 1, 1, pattern, 50))
            for application in (2048, 2049):
                for signal in (-1000, 3001, 3002):
                    for constrained in (0, 1):
                        configurations.append((mode, channels, application, signal, constrained, 0, "mixed", 5))
    return configurations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    configurations = edge_configurations()
    results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="opus-encoder-edges-") as directory:
        pcm = Path(directory) / "input.pcm"
        for mode, channels, application, signal, constrained, dtx, pattern, frames in configurations:
            rate, size = 48000, 960
            os.environ.update(OPUS_ORACLE_COMPLEXITY="10", OPUS_ORACLE_VBR="1", OPUS_ORACLE_FEC="0", OPUS_ORACLE_LOSS="0",
                              OPUS_ORACLE_APPLICATION=str(application), OPUS_ORACLE_SIGNAL=str(signal), OPUS_ORACLE_CVBR=str(constrained), OPUS_ORACLE_DTX=str(dtx))
            data = bytearray(pcm_input(size * frames, channels, "mixed" if pattern == "activity" else pattern))
            if pattern == "activity":
                begin, end = 5 * size * channels * 2, 45 * size * channels * 2
                data[begin:end] = bytes(end - begin)
            pcm.write_bytes(data)
            case = f"{mode}/{channels}/app{application}/signal{signal}/cvbr{constrained}/dtx{dtx}/{pattern}"
            arguments = [str(rate), str(channels), str(size), str(frames), str(MODES[mode]), str((24000 if mode == "silk" else 48000) * channels), str(pcm)]
            try:
                expected = run(args.reference.resolve(), arguments)
                actual = run(args.candidate.resolve(), arguments)
                error = difference(expected, actual)
            except (RuntimeError, subprocess.TimeoutExpired) as failure:
                error = str(failure)
            results.append(case)
            if error:
                failures.append({"case": case, "difference": error})
                print(f"FAIL {case}: {error}", flush=True)
                if args.fail_fast:
                    break
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps({"cases": results, "failures": failures}, indent=2) + "\n")
    print(f"{len(results)} encoder edge cases; {len(failures)} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
