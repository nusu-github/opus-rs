#!/usr/bin/env python3
"""Exercise persistent mode transitions, gain/reset, real FEC, and long loss bursts."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from compare_codec import ROOT, difference, pcm_input, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decode-only", action="store_true")
    args = parser.parse_args()
    failures, count = [], 0
    with tempfile.TemporaryDirectory(prefix="opus-sequences-") as directory:
        output = args.output_dir or Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        pcm_path = Path(directory) / "input.pcm"
        for rate in (24000, 48000):
            for channels in (1, 2):
                size = rate // 50
                pcm_path.write_bytes(pcm_input(size * 20, channels, "mixed"))
                encode_args = [str(rate), str(channels), str(size), "20", "-1000", str(48000 * channels), str(pcm_path)]
                encoded = run(args.reference.resolve(), encode_args, command="codec_transition")
                if not args.decode_only:
                    count += 1
                    try:
                        error = difference(encoded, run(args.candidate.resolve(), encode_args, command="codec_transition"))
                    except RuntimeError as failure:
                        error = str(failure)
                    if error:
                        failures.append(f"encode/{rate}/{channels}: {error}")
                        print(f"FAIL {failures[-1]}", flush=True)
                packets = [line.split("\t")[7] for line in encoded.splitlines()]
                normal = [f"{size} 0 {packet}" for packet in packets]
                variants = {"normal": normal, "burst-loss": normal[:3] + [f"{size} 0 -"] * 20 + normal[3:],
                            "fec": normal[:1] + [f"{size} 1 {packets[2]}"] + normal[2:],
                            "reset": normal[:7] + ["reset"] + normal[7:],
                            "gain": ["gain 1536"] + normal[:3] + [f"{size} 0 -", f"{size} 0 -", "gain -768"] + normal[3:11] + ["reset", "gain 0"] + normal[11:]}
                for label, lines in variants.items():
                    name = f"transitions-{rate}-{channels}-{label}"
                    script = output / f"{name}.packets"
                    script.write_text("\n".join(lines) + "\n")
                    expected = run(args.reference.resolve(), [str(rate), str(channels), str(script)], command="decode")
                    count += 1
                    if args.output_dir:
                        script.with_suffix(".expected.tsv").write_text(expected)
                    try:
                        actual = run(args.candidate.resolve(), [str(rate), str(channels), str(script)], command="decode")
                        error = difference(expected, actual)
                        if error and args.output_dir:
                            script.with_suffix(".actual.tsv").write_text(actual)
                    except RuntimeError as failure:
                        error = str(failure)
                    if error:
                        failures.append(f"{name}: {error}")
                        print(f"FAIL {failures[-1]}", flush=True)
                # Force CELT for a burst crossing periodic PLC into noise PLC.
                celt_args = [str(rate), str(channels), str(size), "6", "1002", str(48000 * channels), str(pcm_path)]
                pcm_path.write_bytes(pcm_input(size * 6, channels, "mixed"))
                celt = run(args.reference.resolve(), celt_args)
                celt_packets = [line.split("\t")[7] for line in celt.splitlines()]
                name = f"celt-long-loss-{rate}-{channels}"
                script = output / f"{name}.packets"
                script.write_text("\n".join([f"{size} 0 {p}" for p in celt_packets[:3]] + [f"{size} 0 -"] * 20 + [f"{size} 0 {p}" for p in celt_packets[3:]]) + "\n")
                expected = run(args.reference.resolve(), [str(rate), str(channels), str(script)], command="decode")
                count += 1
                actual = run(args.candidate.resolve(), [str(rate), str(channels), str(script)], command="decode")
                error = difference(expected, actual)
                if args.output_dir:
                    script.with_suffix(".expected.tsv").write_text(expected)
                    script.with_suffix(".actual.tsv").write_text(actual)
                if error:
                    failures.append(f"{name}: {error}")
                    print(f"FAIL {failures[-1]}", flush=True)
    print(f"{count} sequence cases; {len(failures)} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
