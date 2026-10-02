#!/usr/bin/env python3
"""Compare in-process codec costs with exact, independently checked outputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/reference"))
from compare_codec import pcm_input


def configurations(suite):
    cases = [
        ("silk-mono", 48000, 1, 960, 1000, 16000, 10),
        ("silk-stereo", 48000, 2, 960, 1000, 32000, 10),
        ("hybrid-mono", 48000, 1, 960, 1001, 32000, 10),
        ("hybrid-stereo", 48000, 2, 960, 1001, 64000, 10),
        ("celt-mono", 48000, 1, 960, 1002, 64000, 10),
        ("celt-stereo", 48000, 2, 960, 1002, 128000, 10),
    ]
    if suite == "expanded":
        cases += [
            ("silk-narrowband", 8000, 1, 160, 1000, 12000, 10),
            ("silk-16k", 16000, 1, 320, 1000, 16000, 10),
            ("silk-low-complexity", 48000, 1, 960, 1000, 16000, 0),
            ("celt-2.5ms", 48000, 2, 120, 1002, 128000, 10),
            ("celt-10ms", 48000, 2, 480, 1002, 128000, 10),
            ("celt-60ms", 48000, 2, 2880, 1002, 128000, 10),
            ("celt-low-complexity", 48000, 2, 960, 1002, 128000, 0),
        ]
    if suite == "qext":
        cases = [(f"qext-{rate}-{channels}", rate, channels, rate // 50,
                  1002, 384000 * channels, 10)
                 for rate in (48000, 96000) for channels in (1, 2)]
    return cases


def run(executable, operation, dimensions, iterations, pcm, environment):
    command = [str(executable), operation, *map(str, dimensions), str(iterations), str(pcm)]
    output = subprocess.check_output(command, env=environment)
    return output if operation == "verify" else json.loads(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("core", "expanded", "qext"), default="core")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--loss-every", type=int, default=0)
    parser.add_argument("--time-source", choices=("cpu", "wall"), default="cpu")
    parser.add_argument("--candidate-revision", help="Commit containing the candidate codec, if it is a preserved baseline")
    args = parser.parse_args()
    if args.repetitions < 1 or args.duration_ms < 1 or args.loss_every < 0:
        parser.error("repetitions and duration must be positive; loss interval must be nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cpu = min(os.sched_getaffinity(0)) if args.cpu is None else args.cpu
    os.sched_setaffinity(0, {cpu})
    environment = dict(os.environ, OPUS_BENCH_LOSS_EVERY=str(args.loss_every))
    source_hash = hashlib.sha256()
    source_files = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", *sorted((ROOT / "rust").rglob("*.rs"))]
    for source in source_files:
        source_hash.update(source.relative_to(ROOT).as_posix().encode() + b"\0")
        source_hash.update(source.read_bytes())
    record = {
        "schema_version": 1,
        "reference_revision": "503d81b138d76621aae4b12786e90de48aa8db3a",
        "measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rust_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_revision": args.candidate_revision or "working-tree",
        "working_tree_source_sha256": source_hash.hexdigest() if not args.candidate_revision else None,
        "working_tree_changes": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "profile": args.profile,
        "suite": args.suite,
        "loss_every": args.loss_every,
        "cpu_affinity": cpu,
        "platform": platform.platform(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "gcc": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        "reference_binary_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_binary_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "timing": "In-process thread CPU time from /proc/thread-self/schedstat; monotonic wall time also retained; setup, corpus generation, and file I/O excluded; 64 warmup frames; 128-frame mixed-input corpus; alternating order; median per-frame cost.",
        "threshold": "Rust elapsed time / C elapsed time >=1.10",
        "repetitions": args.repetitions,
        "minimum_c_duration_ms": args.duration_ms,
        "time_source": args.time_source,
        "cases": [],
    }
    for name, *dimensions in configurations(args.suite):
        if args.case and name not in args.case:
            continue
        rate, channels, frame, mode, bitrate, complexity = dimensions
        pcm = args.output.parent / f"input-{rate}-{channels}-{frame}.pcm"
        pcm.write_bytes(pcm_input(frame * 128, channels, "mixed"))
        expected = run(args.reference, "verify", dimensions, 128, pcm, environment)
        actual = run(args.candidate, "verify", dimensions, 128, pcm, environment)
        if actual != expected:
            raise RuntimeError(f"{name}: independent packet/PCM/range verification failed")
        for operation in ("encode", "decode"):
            calibration_iterations = 256
            while True:
                calibration = run(args.reference, operation, dimensions, calibration_iterations, pcm, environment)
                calibration_ns = calibration["cpu_ns"] if args.time_source == "cpu" else calibration["elapsed_ns"]
                if calibration_ns >= 20_000_000:
                    break
                calibration_iterations *= 4
                if calibration_iterations > 1_048_576:
                    raise RuntimeError("timing source did not advance during calibration")
            iterations = max(256, min(200000, int(calibration_iterations * args.duration_ms * 1000000 / calibration_ns)))
            measurements = {"c": [], "rust": []}
            wall_measurements = {"c": [], "rust": []}
            for repetition in range(args.repetitions):
                order = [("c", args.reference), ("rust", args.candidate)]
                if repetition % 2:
                    order.reverse()
                pair = {}
                for label, executable in order:
                    result = run(executable, operation, dimensions, iterations, pcm, environment)
                    pair[label] = result
                    elapsed = result["cpu_ns"] if args.time_source == "cpu" else result["elapsed_ns"]
                    measurements[label].append(elapsed / iterations)
                    wall_measurements[label].append(result["elapsed_ns"] / iterations)
                if (pair["c"]["checksum"], pair["c"]["final_range"]) != (pair["rust"]["checksum"], pair["rust"]["final_range"]):
                    raise RuntimeError(f"{name}/{operation}: timed state outputs differ")
            c_ns = statistics.median(measurements["c"])
            rust_ns = statistics.median(measurements["rust"])
            case = {
                "name": name, "operation": operation, "sample_rate": rate,
                "channels": channels, "frame_samples": frame, "mode": mode,
                "bitrate": bitrate, "complexity": complexity, "iterations": iterations,
                "input_sha256": hashlib.sha256(pcm.read_bytes()).hexdigest(),
                "verified_output_sha256": hashlib.sha256(expected).hexdigest(),
                "c_ns_per_frame": c_ns, "rust_ns_per_frame": rust_ns,
                "rust_over_c": rust_ns / c_ns,
                "slowdown_percent": (rust_ns / c_ns - 1) * 100,
                "samples_ns_per_frame": measurements,
                "wall_samples_ns_per_frame": wall_measurements,
            }
            record["cases"].append(case)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(f"{args.profile}/{name}/{operation}: C {c_ns / 1000:.1f}us, Rust {rust_ns / 1000:.1f}us, {rust_ns / c_ns:.2f}x ({case['slowdown_percent']:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
