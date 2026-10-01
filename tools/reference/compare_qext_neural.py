#!/usr/bin/env python3
"""Compare supported 96 kHz QEXT combinations with DRED, Deep PLC and OSCE."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import ROOT, difference, pcm_input


def run(binary: Path, command: str, arguments: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        [str(binary.resolve()), command, *arguments], env=environment,
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode:
        raise RuntimeError(f"{binary}: {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="QEXT + bundled Deep PLC candidate")
    parser.add_argument("--osce-candidate", type=Path, required=True, help="QEXT + OSCE candidate without Deep PLC")
    parser.add_argument("--dred-reference", type=Path,
                        default=ROOT / "target/reference-dred-quantized-qext/opus-reference")
    parser.add_argument("--osce-reference", type=Path,
                        default=ROOT / "target/reference-osce-quantized-qext/opus-reference")
    parser.add_argument("--osce-weights", type=Path,
                        default=ROOT / "target/reference/osce-weights-quantized.bin")
    parser.add_argument("--osce-reference-weights", type=Path,
                        help="Load this model into a Rust OSCE reference; omit for bundled C models")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    count = failures = 0
    rate, frame = 96000, 1920
    environment = dict(os.environ)
    environment.pop("OPUS_ORACLE_DNN_BLOB", None)
    environment.update(OPUS_ORACLE_QEXT="1", OPUS_ORACLE_DRED_DURATION="40",
                       OPUS_ORACLE_LOSS="15", OPUS_ORACLE_DECODER_COMPLEXITY="10")

    def compare(name: str, reference: Path, command: str, arguments: list[str],
                reference_env: dict[str, str], candidate_env: dict[str, str],
                candidate: Path | None = None) -> str | None:
        nonlocal count, failures
        count += 1
        expected = actual = ""
        try:
            expected = run(reference, command, arguments, reference_env)
            actual = run(candidate or args.candidate, command, arguments, candidate_env)
            error = difference(expected, actual)
        except (RuntimeError, subprocess.TimeoutExpired) as failure:
            error = str(failure)
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            label = name.replace("/", "-")
            (args.output_dir / f"{label}.expected.tsv").write_text(expected)
            (args.output_dir / f"{label}.actual.tsv").write_text(actual)
        if error:
            failures += 1
            print(f"FAIL {name}: {error}", flush=True)
        else:
            print(f"PASS {name}", flush=True)
        return expected if expected else None

    with tempfile.TemporaryDirectory(prefix="opus-qext-neural-") as directory:
        temporary = Path(directory)
        pcm = temporary / "input.pcm"
        script = temporary / "input.packets"
        for channels in (1, 2):
            pcm.write_bytes(pcm_input(frame * 16, channels, "mixed"))
            for mode in (1000, 1001, 1002, -1000):
                arguments = [str(rate), str(channels), str(frame), "16", str(mode),
                             str(96000 * channels), str(pcm)]
                for vbr in (0, 1):
                    environment["OPUS_ORACLE_VBR"] = str(vbr)
                    encoded = compare(f"dred/96k/{channels}/mode{mode}/vbr{vbr}",
                                      args.dred_reference, "codec", arguments,
                                      environment, environment)
                if not encoded:
                    continue
                packets = [line.split("\t")[7] for line in encoded.splitlines()]
                normal = [f"{frame} 0 {packet}" for packet in packets]
                # C deliberately retains conventional PLC for 96 kHz CELT,
                # including builds that have a loaded neural model.
                script.write_text("\n".join(normal[:3] + [f"{frame} 0 -"] * 12 + normal[3:]) + "\n")
                compare(f"plc/96k/{channels}/mode{mode}", args.dred_reference, "decode",
                        [str(rate), str(channels), str(script)], environment, environment)
                future = 8
                recovered = [f"{frame} {(future - lost) * frame} {packets[future]}"
                             for lost in range(5, future)]
                normal_dred = [f"{frame} -1 {packet}" for packet in packets]
                script.write_text("\n".join(normal_dred[:5] + recovered + normal_dred[future:]) + "\n")
                for deferred in (0, 1):
                    environment["OPUS_ORACLE_DRED_DEFER"] = str(deferred)
                    compare(f"recovery/96k/{channels}/mode{mode}/defer{deferred}",
                            args.dred_reference, "dred_decode",
                            [str(rate), str(channels), str(script)], environment, environment)

        environment.pop("OPUS_ORACLE_DRED_DURATION", None)
        environment.pop("OPUS_ORACLE_DRED_DEFER", None)
        environment.update(OPUS_ORACLE_LOSS="0", OPUS_ORACLE_VBR="1")
        if args.osce_reference_weights:
            environment["OPUS_ORACLE_DNN_BLOB"] = str(args.osce_reference_weights.resolve())
        for channels in (1, 2):
            pcm.write_bytes(pcm_input(frame * 8, channels, "mixed"))
            for mode in (1000, 1001):
                encoded = run(args.osce_reference, "codec",
                              [str(rate), str(channels), str(frame), "8", str(mode),
                               str(24000 * channels), str(pcm)], environment)
                normal = [f"{frame} 0 {line.split(chr(9))[7]}" for line in encoded.splitlines()]
                for complexity in (0, 4, 6, 7):
                    for bwe in (0, 1):
                        for loss in (False, True):
                            lines = normal[:]
                            if loss:
                                lines[3] = f"{frame} 0 -"
                            script.write_text("\n".join(lines) + "\n")
                            environment.update(OPUS_ORACLE_DECODER_COMPLEXITY=str(complexity),
                                               OPUS_ORACLE_OSCE_BWE=str(bwe))
                            candidate_env = {**environment,
                                             "OPUS_ORACLE_DNN_BLOB": str(args.osce_weights.resolve())}
                            compare(f"osce/96k/{channels}/mode{mode}/complexity{complexity}/bwe{bwe}/loss{int(loss)}",
                                    args.osce_reference, "decode",
                                    [str(rate), str(channels), str(script)], environment, candidate_env,
                                    args.osce_candidate)
    print(f"{count} QEXT neural cases; {failures} failures")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
