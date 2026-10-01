#!/usr/bin/env python3
"""Compare complete codec results against the pinned scalar C process oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATES = (8000, 12000, 16000, 24000, 48000)
MODES = {"silk": 1000, "hybrid": 1001, "celt": 1002, "auto": -1000}
FIELDS = ("tag", "frame", "packet_length", "sample_count", "encoder_range", "decoder_range", "float_decoder_range", "packet", "pcm_i16", "pcm_f32")


def cases(quick: bool):
    if quick:
        yield (8000, 1, 20000, "silk", "mixed")
        yield (16000, 2, 20000, "silk", "mixed")
        yield (24000, 1, 20000, "hybrid", "mixed")
        yield (48000, 2, 20000, "hybrid", "mixed")
        yield (48000, 2, 2500, "celt", "mixed")
        yield (48000, 1, 20000, "celt", "mixed")
        yield (12000, 1, 60000, "auto", "mixed")
        return
    for mode in MODES:
        for rate in RATES:
            if mode == "hybrid" and rate < 24000:
                continue
            for channels in (1, 2):
                durations = (10000, 20000, 40000, 60000, 80000, 100000, 120000)
                if mode in ("celt", "auto"):
                    durations = (2500, 5000) + durations
                for duration in durations:
                    yield (rate, channels, duration, mode, "mixed")
    for mode, rate in (("silk", 16000), ("hybrid", 48000), ("celt", 48000)):
        for channels in (1, 2):
            for pattern in ("silence", "impulse", "alternating", "noise"):
                yield (rate, channels, 20000, mode, pattern)


def pcm_input(samples: int, channels: int, pattern: str) -> bytes:
    """Integer-only input generator, independent of platform transcendental math."""
    output = bytearray()
    state = 0x12345678
    for index in range(samples):
        for channel in range(channels):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            noise = (state >> 16) - 32768
            triangle = ((index * (197 + channel * 37)) % 32768) - 16384
            if pattern == "silence":
                sample = 0
            elif pattern == "impulse":
                sample = 32767 if index % 257 == 0 else 0
            elif pattern == "alternating":
                sample = -32768 if (index + channel) % 2 else 32767
            elif pattern == "noise":
                sample = noise
            else:
                sample = max(-32768, min(32767, triangle + noise // 8))
            output.extend(struct.pack("<h", sample))
    return bytes(output)


def run(executable: Path, arguments: list[str], command: str = "codec") -> str:
    result = subprocess.run([str(executable), command, *arguments], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{executable}: exit {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def difference(expected: str, actual: str) -> str | None:
    expected_lines, actual_lines = expected.splitlines(), actual.splitlines()
    if len(expected_lines) != len(actual_lines):
        return f"frame count: expected {len(expected_lines)}, got {len(actual_lines)}"
    for index, (left_line, right_line) in enumerate(zip(expected_lines, actual_lines)):
        left, right = left_line.split("\t"), right_line.split("\t")
        names = FIELDS if left[0] == "C" else ("tag", "frame", "sample_count", "decoder_range", "float_decoder_range", "pcm_i16", "pcm_f32")
        if left[0] == "R":
            names = ("tag", "frame", "dred_samples", "dred_end")
        elif left[0] == "Q":
            names = ("tag", "frame", "sample_count", "decoder_range", "pcm_i24")
        if len(left) != len(names) or len(right) != len(names):
            return f"frame {index}: invalid TSV field count"
        for field, expected_value, actual_value in zip(names, left, right):
            if expected_value == actual_value:
                continue
            if field in ("packet", "pcm_i16", "pcm_f32", "pcm_i24"):
                width = 8 if field in ("pcm_f32", "pcm_i24") else 4 if field == "pcm_i16" else 2
                offset = next((n for n, (a, b) in enumerate(zip(expected_value, actual_value)) if a != b), min(len(expected_value), len(actual_value))) // width
                start = offset * width
                return f"frame {index} {field}[{offset}]: expected {expected_value[start:start + width]}, got {actual_value[start:start + width]} (hex lengths {len(expected_value)}/{len(actual_value)})"
            return f"frame {index} {field}: expected {expected_value}, got {actual_value}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path, help="Rust codec oracle executable; required unless generating reference records")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", choices=tuple(MODES), action="append", help="Restrict to requested codec modes")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--record", type=Path, help="Write a JSON manifest of exact reference SHA-256 digests")
    parser.add_argument("--profile", default="scalar-float-no-fma-no-approx-lrintf", help="Reference build profile recorded in the manifest")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not args.candidate and not args.record:
        parser.error("provide --candidate or --record")
    records, failures = [], []
    with tempfile.TemporaryDirectory(prefix="opus-reference-") as directory:
        pcm_path = Path(directory) / "input.pcm"
        for rate, channels, duration, mode, pattern in cases(args.quick):
            if args.mode and mode not in args.mode:
                continue
            frame_size = rate * duration // 1000000
            bitrate = (24000 if mode == "silk" else 48000) * channels
            case = f"{mode}/{rate}/{channels}/{duration}us/{pattern}"
            pcm_path.write_bytes(pcm_input(frame_size * args.frames, channels, pattern))
            arguments = [str(rate), str(channels), str(frame_size), str(args.frames), str(MODES[mode]), str(bitrate), str(pcm_path)]
            try:
                expected = run(args.reference.resolve(), arguments)
                for line in expected.splitlines():
                    fields = line.split("\t")
                    if fields[4] != fields[5] or fields[4] != fields[6]:
                        raise RuntimeError("reference encoder and decoder final ranges differ")
                records.append({"case": case, "frames": args.frames, "bitrate": bitrate, "input_sha256": hashlib.sha256(pcm_path.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(expected.encode()).hexdigest()})
                if args.candidate:
                    actual = run(args.candidate.resolve(), arguments)
                    error = difference(expected, actual)
                    if error:
                        failures.append({"case": case, "difference": error})
                        print(f"FAIL {case}: {error}", flush=True)
                        if args.fail_fast:
                            break
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append({"case": case, "difference": str(error)})
                print(f"ERROR {case}: {error}", flush=True)
                if args.fail_fast:
                    break
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps({"revision": "503d81b138d76621aae4b12786e90de48aa8db3a", "profile": args.profile, "cases": records, "failures": failures}, indent=2) + "\n")
    print(f"{len(records)} reference cases; {len(failures)} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
