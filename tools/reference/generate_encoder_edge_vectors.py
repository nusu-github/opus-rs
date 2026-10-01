#!/usr/bin/env python3
"""Generate exact packet/range and selected PCM goldens for encoder edge controls."""
import os
import tempfile
from pathlib import Path
from compare_codec import ROOT, MODES, pcm_input, run
from compare_encoder_edges import edge_configurations


def main():
    records = ["# Generated from scalar C revision 503d81b138d76621aae4b12786e90de48aa8db3a."]
    with tempfile.TemporaryDirectory(prefix="opus-edge-vectors-") as directory:
        path = Path(directory) / "input.pcm"
        for mode, channels, application, signal, cvbr, dtx, pattern, frames in edge_configurations():
            os.environ.update(OPUS_ORACLE_COMPLEXITY="10", OPUS_ORACLE_VBR="1", OPUS_ORACLE_FEC="0", OPUS_ORACLE_LOSS="0",
                              OPUS_ORACLE_APPLICATION=str(application), OPUS_ORACLE_SIGNAL=str(signal), OPUS_ORACLE_CVBR=str(cvbr), OPUS_ORACLE_DTX=str(dtx))
            pcm = bytearray(pcm_input(960 * frames, channels, "mixed" if pattern == "activity" else pattern))
            if pattern == "activity":
                begin, end = 5 * 960 * channels * 2, 45 * 960 * channels * 2
                pcm[begin:end] = bytes(end - begin)
            path.write_bytes(pcm)
            output = run(ROOT / "target/reference/opus-reference", ["48000", str(channels), "960", str(frames), str(MODES[mode]), str((24000 if mode == "silk" else 48000) * channels), str(path)])
            records.append(f"E {MODES[mode]} {channels} {application} {signal} {cvbr} {dtx} {pattern} {frames}")
            for frame, line in enumerate(output.splitlines()):
                fields = line.split("\t")
                pcm_fields = f"{fields[8]} {fields[9]}" if dtx and frame in (0, 5, 15, 16, 35, 45, 49) else "- -"
                records.append(f"P {fields[4]} {fields[5]} {fields[6]} {fields[7]} {pcm_fields}")
    destination = ROOT / "rust/tests/fixtures/reference/encoder-edges.tsv"
    destination.write_text("\n".join(records) + "\n")
    print(f"Generated {len(edge_configurations())} edge configurations: {destination}")


if __name__ == "__main__":
    main()
