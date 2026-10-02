#!/usr/bin/env python3
"""Wrap checked C-reference packets in a deterministic Ogg Opus test stream."""

from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rust/tests/fixtures/reference/celt-stereo-20ms.tsv"


def page(payload: bytes, sequence: int, flags: int, granule: int) -> bytes:
    lacing = bytes([255] * (len(payload) // 255) + [len(payload) % 255])
    header = bytearray(b"OggS" + bytes([0, flags]))
    header += struct.pack("<QII", granule, 1, sequence)
    header += bytes(4) + bytes([len(lacing)]) + lacing
    result = header + payload
    checksum = 0
    for byte in result:
        checksum ^= byte << 24
        for _ in range(8):
            checksum = ((checksum << 1) ^ (0x04C11DB7 if checksum & 0x80000000 else 0)) & 0xFFFFFFFF
    result[22:26] = struct.pack("<I", checksum)
    return bytes(result)


def main() -> None:
    rows = [line.split("\t") for line in SOURCE.read_text().splitlines() if line.startswith("C\t")]
    identification = b"OpusHead" + bytes([1, 2]) + struct.pack("<HIhB", 0, 48000, 0, 0)
    comments = b"OpusTags" + struct.pack("<I", 7) + b"opus-rs" + struct.pack("<I", 0)
    output = page(identification, 0, 2, 0) + page(comments, 1, 0, 0)
    granule = 0
    for index, row in enumerate(rows):
        granule += int(row[3])
        output += page(bytes.fromhex(row[7]), index + 2, 4 if index == len(rows) - 1 else 0, granule)
    SOURCE.with_suffix(".ogg").write_bytes(output)


if __name__ == "__main__":
    main()
