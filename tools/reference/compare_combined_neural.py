#!/usr/bin/env python3
"""Compare combined Deep PLC, OSCE, and DRED at 48 and 96 kHz with scalar C."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import ROOT, difference, pcm_input


def run(executable: Path, command: str, arguments: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run([str(executable.resolve()), command, *arguments],
                            text=True, capture_output=True, env=environment, timeout=180)
    if result.returncode:
        raise RuntimeError(f"{executable}: {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference-neural-combined-quantized-qext/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Combined OSCE, Deep PLC, and DRED model blob")
    parser.add_argument("--reference-weights", type=Path, help="Model blob for a Rust reference; omit for bundled C models")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    records = []
    environment = {key: value for key, value in os.environ.items() if not key.startswith("OPUS_ORACLE_")}
    environment.update(OPUS_ORACLE_QEXT="1", OPUS_ORACLE_FEC="1", OPUS_ORACLE_LOSS="15",
                       OPUS_ORACLE_VBR="1", OPUS_ORACLE_DRED_DURATION="40",
                       OPUS_ORACLE_DECODER_COMPLEXITY="10", OPUS_ORACLE_OSCE_BWE="1")
    reference_environment = environment.copy()
    if args.reference_weights:
        reference_environment["OPUS_ORACLE_DNN_BLOB"] = str(args.reference_weights.resolve())
    candidate_environment = {**environment, "OPUS_ORACLE_DNN_BLOB": str(args.weights.resolve())}

    def compare(name: str, command: str, arguments: list[str], expected: str | None = None) -> str:
        error = None
        actual = ""
        try:
            if expected is None:
                expected = run(args.reference, command, arguments, reference_environment)
            actual = run(args.candidate, command, arguments, candidate_environment)
            error = difference(expected, actual)
        except (RuntimeError, subprocess.TimeoutExpired) as failure:
            error = str(failure)
        records.append({"case": name, "error": error})
        print(f"{'FAIL' if error else 'PASS'} {name}" + (f": {error}" if error else ""), flush=True)
        if error and args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = args.output_dir / name.replace("/", "-")
            prefix.with_suffix(".expected.tsv").write_text(expected or "")
            prefix.with_suffix(".actual.tsv").write_text(actual)
        if error and args.fail_fast:
            raise RuntimeError(error)
        return expected or ""

    with tempfile.TemporaryDirectory(prefix="opus-neural-combined-") as directory:
        temporary = Path(directory)
        output = args.output_dir or temporary
        output.mkdir(parents=True, exist_ok=True)
        configurations = [(48000, 1, 1000), (48000, 2, 1001), (48000, 1, 1002),
                          (96000, 1, 1000), (96000, 2, 1001), (96000, 2, 1002)]
        try:
            for rate, channels, mode in configurations:
                frame = rate // 50
                stem = f"combined/{rate}/{channels}/mode{mode}"
                pcm = temporary / "input.pcm"
                pcm.write_bytes(pcm_input(frame * 16, channels, "mixed"))
                bitrate = (192000 if mode == 1002 else 48000) * channels
                reference_environment.update(OPUS_ORACLE_DECODER_COMPLEXITY="10", OPUS_ORACLE_OSCE_BWE="1")
                candidate_environment.update(OPUS_ORACLE_DECODER_COMPLEXITY="10", OPUS_ORACLE_OSCE_BWE="1")
                encoded = compare(stem + "/codec", "codec", [str(rate), str(channels), str(frame), "16", str(mode), str(bitrate), str(pcm)])
                packets = [line.split("\t")[7] for line in encoded.splitlines()]
                if len(packets) != 16:
                    continue
                normal = [f"{frame} 0 {packet}" for packet in packets]
                variants = {
                    "normal": normal,
                    "long-plc": normal[:5] + [f"{frame} 0 -"] * 20 + normal[5:],
                    "fec": normal[:5] + [f"{frame} 1 {packets[6]}"] + normal[6:],
                    "reset-gain": normal[:4] + ["gain 1536", f"{frame} 0 -"] + normal[4:9] + ["reset", "gain -768"] + normal[9:],
                }
                for complexity in (6, 10):
                    for bwe in ((0,) if mode == 1002 else (0, 1)):
                        controls = {"OPUS_ORACLE_DECODER_COMPLEXITY": str(complexity), "OPUS_ORACLE_OSCE_BWE": str(bwe)}
                        reference_environment.update(controls)
                        candidate_environment.update(controls)
                        for variant, lines in variants.items():
                            name = f"{stem}/complexity{complexity}/bwe{bwe}/{variant}"
                            script = output / (name.replace("/", "-") + ".packets")
                            script.write_text("\n".join(lines) + "\n")
                            compare(name, "decode", [str(rate), str(channels), str(script)])
                reference_environment.update(OPUS_ORACLE_DECODER_COMPLEXITY="10", OPUS_ORACLE_OSCE_BWE="1")
                candidate_environment.update(OPUS_ORACLE_DECODER_COMPLEXITY="10", OPUS_ORACLE_OSCE_BWE="1")
                dred_normal = [f"{frame} -1 {packet}" for packet in packets]
                for burst in (1, 4):
                    future = 5 + burst
                    recovered = [f"{frame} {(future - missing) * frame} {packets[future]}" for missing in range(5, future)]
                    name = f"{stem}/dred-loss{burst}"
                    script = output / (name.replace("/", "-") + ".packets")
                    script.write_text("\n".join(dred_normal[:5] + recovered + dred_normal[future:]) + "\n")
                    for deferred in (0, 1):
                        reference_environment["OPUS_ORACLE_DRED_DEFER"] = str(deferred)
                        candidate_environment["OPUS_ORACLE_DRED_DEFER"] = str(deferred)
                        compare(f"{name}/deferred{deferred}", "dred_decode", [str(rate), str(channels), str(script)])
        except RuntimeError:
            pass  # The failing case is already recorded by compare().
    failures = sum(record["error"] is not None for record in records)
    if not failures and len(records) != 110:
        raise RuntimeError(f"Expected 110 combined cases, received {len(records)}")
    print(f"{len(records)} combined neural cases; {failures} failures")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(records, indent=2) + "\n")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
