#!/usr/bin/env python3
"""Cross-decode pinned C packets and compare integer PCM, float bits and ranges."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from compare_codec import MODES, ROOT, cases, difference, pcm_input, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", choices=tuple(MODES), action="append", help="Restrict to requested codec modes")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, help="Retain packet scripts and reference output for debugging")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--loss", action="store_true", help="Include packet loss concealment and FEC recovery sequences")
    args = parser.parse_args()
    if not args.candidate and not args.output_dir:
        parser.error("provide --candidate or --output-dir")
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    count, failures = 0, 0
    with tempfile.TemporaryDirectory(prefix="opus-decode-reference-") as directory:
        output = args.output_dir or Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        pcm_path = Path(directory) / "input.pcm"
        for rate, channels, duration, mode, pattern in cases(args.quick):
            if args.mode and mode not in args.mode:
                continue
            size = rate * duration // 1000000
            bitrate = (24000 if mode == "silk" else 48000) * channels
            name = f"{mode}-{rate}-{channels}-{duration}us-{pattern}"
            pcm_path.write_bytes(pcm_input(size * args.frames, channels, pattern))
            encoder_args = [str(rate), str(channels), str(size), str(args.frames), str(MODES[mode]), str(bitrate), str(pcm_path)]
            try:
                codec_output = run(args.reference.resolve(), encoder_args)
                packets = [line.split("\t")[7] for line in codec_output.splitlines()]
                streams = {"normal": [f"{size} 0 {packet}" for packet in packets]}
                if args.loss:
                    streams["loss"] = [f"{size} 0 {packet if index != 1 else '-'}" for index, packet in enumerate(packets)]
                    streams["fec"] = [f"{size} 0 {packets[0]}", f"{size} 1 {packets[2]}"] + [f"{size} 0 {packet}" for packet in packets[2:]]
                for scenario, lines in streams.items():
                    script = output / f"{name}-{scenario}.packets"
                    script.write_text("\n".join(lines) + "\n")
                    arguments = [str(rate), str(channels), str(script)]
                    expected = run(args.reference.resolve(), arguments, command="decode")
                    count += 1
                    if args.output_dir:
                        script.with_suffix(".expected.tsv").write_text(expected)
                    if args.candidate:
                        actual = run(args.candidate.resolve(), arguments, command="decode")
                        error = difference(expected, actual)
                        if error:
                            failures += 1
                            print(f"FAIL {name}/{scenario}: {error}", flush=True)
                            if args.output_dir:
                                script.with_suffix(".actual.tsv").write_text(actual)
                            if args.fail_fast:
                                print(f"{count} decoder cases; {failures} failures")
                                return 1
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures += 1
                print(f"ERROR {name}: {error}", flush=True)
                if args.fail_fast:
                    break
    print(f"{count} decoder cases; {failures} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
