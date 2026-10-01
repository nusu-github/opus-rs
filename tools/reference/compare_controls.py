#!/usr/bin/env python3
"""Compare C and Rust encoder complexity and rate-control branches exactly."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from compare_codec import ROOT, difference, pcm_input, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    cases, failures = [], []
    with tempfile.TemporaryDirectory(prefix="opus-controls-") as directory:
        pcm = Path(directory) / "input.pcm"
        for complexity in range(11):
            for vbr in (0, 1):
                os.environ["OPUS_ORACLE_COMPLEXITY"] = str(complexity)
                os.environ["OPUS_ORACLE_VBR"] = str(vbr)
                for rate in (8000, 12000, 16000):
                    for channels in (1, 2):
                        size = rate // 50
                        pcm.write_bytes(pcm_input(size * 5, channels, "mixed"))
                        case = f"silk/{rate}/{channels}/complexity{complexity}/vbr{vbr}"
                        arguments = [str(rate), str(channels), str(size), "5", "1000", str(24000 * channels), str(pcm)]
                        try:
                            expected = run(args.reference.resolve(), arguments)
                            actual = run(args.candidate.resolve(), arguments)
                            error = difference(expected, actual)
                        except RuntimeError as failure:
                            error = str(failure)
                        cases.append(case)
                        if error:
                            failures.append({"case": case, "difference": error})
                            print(f"FAIL {case}: {error}", flush=True)
                            if args.fail_fast:
                                break
                    if failures and args.fail_fast:
                        break
                if failures and args.fail_fast:
                    break
            if failures and args.fail_fast:
                break
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps({"cases": cases, "failures": failures}, indent=2) + "\n")
    print(f"{len(cases)} encoder-control cases; {len(failures)} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
