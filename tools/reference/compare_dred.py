#!/usr/bin/env python3
"""Compare real DRED packet parsing and neural audio recovery with scalar C."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from compare_codec import MODES, RATES, ROOT, difference, pcm_input


def run(binary: Path, command: str, args: list[str], env: dict[str, str]) -> str:
    result = subprocess.run([str(binary.resolve()), command, *args], env=env,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"{binary}: {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference-dred-quantized/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    configurations = [(mode, rate, channels) for mode in MODES for rate in RATES
                      if mode != "hybrid" or rate >= 24000 for channels in (1, 2)]
    if args.quick:
        configurations = [("silk", 16000, 1), ("silk", 48000, 2),
                          ("hybrid", 48000, 2), ("celt", 48000, 1),
                          ("celt", 24000, 2), ("auto", 12000, 1)]
    env = os.environ.copy()
    env.update(OPUS_ORACLE_DRED_DURATION="40", OPUS_ORACLE_LOSS="15",
               OPUS_ORACLE_VBR="1", OPUS_ORACLE_DECODER_COMPLEXITY="0")
    count = failures = 0
    with tempfile.TemporaryDirectory(prefix="opus-dred-recovery-") as directory:
        temporary = Path(directory)
        output = args.output_dir or temporary
        output.mkdir(parents=True, exist_ok=True)
        for mode, rate, channels in configurations:
            frame = rate // 50
            pcm = temporary / "input.pcm"
            pcm.write_bytes(pcm_input(frame * 16, channels, "mixed"))
            encoded = run(args.reference, "codec", [str(rate), str(channels), str(frame),
                          "16", str(MODES[mode]), str(48000 * channels), str(pcm)], env)
            packets = [row.split("\t")[7] for row in encoded.splitlines()]
            normal = [f"{frame} -1 {packet}" for packet in packets]
            for burst in ((1, 8) if args.quick else (1, 3, 8)):
                future = 5 + burst
                recovered = [f"{frame} {(future - missing) * frame} {packets[future]}"
                             for missing in range(5, future)]
                script = output / f"{mode}-{rate}-{channels}-loss{burst}.packets"
                script.write_text("\n".join(normal[:5] + recovered + normal[future:]) + "\n")
                for deferred in (0, 1):
                    env["OPUS_ORACLE_DRED_DEFER"] = str(deferred)
                    name = f"{mode}/{rate}/{channels}/loss{burst}/deferred{deferred}"
                    count += 1
                    try:
                        arguments = [str(rate), str(channels), str(script)]
                        expected = run(args.reference, "dred_decode", arguments, env)
                        actual = run(args.candidate, "dred_decode", arguments, env)
                        error = difference(expected, actual)
                        if args.output_dir:
                            script.with_suffix(f".defer{deferred}.expected.tsv").write_text(expected)
                            script.with_suffix(f".defer{deferred}.actual.tsv").write_text(actual)
                    except (RuntimeError, subprocess.TimeoutExpired) as failure:
                        error = str(failure)
                    if error:
                        failures += 1
                        print(f"FAIL {name}: {error}", flush=True)
                        if args.fail_fast:
                            print(f"{count} DRED recovery cases; {failures} failures")
                            return 1
    print(f"{count} DRED recovery cases; {failures} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
