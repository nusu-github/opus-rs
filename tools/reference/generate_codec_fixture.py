#!/usr/bin/env python3
"""Regenerate the complete C-produced CELT codec fixture without Rust code."""
from pathlib import Path
import tempfile
from compare_codec import ROOT, pcm_input, run


def main() -> None:
    destination = ROOT / "rust/tests/fixtures/reference"
    destination.mkdir(parents=True, exist_ok=True)
    pcm = destination / "celt-stereo-20ms.pcm"
    pcm.write_bytes(pcm_input(960 * 3, 2, "mixed"))
    output = run(ROOT / "target/reference/opus-reference", ["48000", "2", "960", "3", "1002", "96000", str(pcm)])
    (destination / "celt-stereo-20ms.tsv").write_text(output)
    with tempfile.TemporaryDirectory(prefix="opus-silk-fixture-") as directory:
        input_path = Path(directory) / "silk.pcm"
        for rate, channels in ((8000, 1), (16000, 2)):
            size = rate // 50
            input_path.write_bytes(pcm_input(size * 5, channels, "mixed"))
            output = run(ROOT / "target/reference/opus-reference", [str(rate), str(channels), str(size), "5", "1000", str(24000 * channels), str(input_path)])
            packets = [line.split("\t")[7] for line in output.splitlines()]
            for loss in (False, True):
                name = f"silk-{rate}-{channels}-{'loss' if loss else 'normal'}"
                path = destination / f"{name}.packets"
                path.write_text("\n".join(f"{size} 0 {packet if not loss or index != 1 else '-'}" for index, packet in enumerate(packets)) + "\n")
                decoded = run(ROOT / "target/reference/opus-reference", [str(rate), str(channels), str(path)], command="decode")
                (destination / f"{name}.tsv").write_text(decoded)


        mono_stereo = []
        for channels in (1, 2):
            input_path.write_bytes(pcm_input(960 * 3, channels, "mixed"))
            encoded = run(ROOT / "target/reference/opus-reference", ["48000", str(channels), "960", "3", "1000", str(24000 * channels), str(input_path)])
            mono_stereo.append([line.split("\t")[7] for line in encoded.splitlines()])
        mixed_packets = mono_stereo[0][:2] + mono_stereo[1][:2] + mono_stereo[0][2:] + mono_stereo[1][2:]
        script = destination / "silk-channel-transitions.packets"
        script.write_text("".join(f"960 0 {packet}\n" for packet in mixed_packets))
        for channels in (1, 2):
            decoded = run(ROOT / "target/reference/opus-reference", ["48000", str(channels), str(script)], command="decode")
            (destination / f"silk-channel-transitions-{channels}.tsv").write_text(decoded)


if __name__ == "__main__":
    main()
