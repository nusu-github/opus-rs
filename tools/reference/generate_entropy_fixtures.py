#!/usr/bin/env python3
"""Regenerate entropy traces with the separately compiled upstream C oracle."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "rust/tests/fixtures/reference/entropy"
ORACLE = ROOT / "target/reference/opus-reference"


def generate(name: str, operations: list[str], size: int = 4096) -> None:
    source = DEST / f"{name}.script"
    source.write_text(f"size {size}\n" + "\n".join(operations) + "\n")
    result = subprocess.run(
        [str(ORACLE), "entropy", str(source)], check=True, capture_output=True
    )
    source.with_suffix(".tsv").write_bytes(result.stdout)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    state = 0x71A60F35

    def draw() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state

    operations = []
    for i in range(768):
        value = draw()
        kind = i % 7
        if kind == 0:
            total = 2 + draw() % 0xFFFFFFFE
            operations.append(f"uint {value % total} {total}")
        elif kind == 1:
            bits = 1 + draw() % 25
            operations.append(f"bits {value & ((1 << bits) - 1)} {bits}")
        elif kind == 2:
            operations.append(f"bit {value & 1} {1 + draw() % 15}")
        elif kind == 3:
            operations.append(f"icdf {value % 4}")
        elif kind == 4:
            operations.append(f"icdf16 {value % 4}")
        elif kind == 5:
            total = 2 + draw() % 65534
            low = value % total
            high = low + 1 + draw() % (total - low)
            operations.append(f"encode {low} {high} {total}")
        else:
            bits = 1 + draw() % 16
            low = value % (1 << bits)
            high = low + 1 + draw() % ((1 << bits) - low)
            operations.append(f"bin {low} {high} {bits}")
    generate("mixed", operations + ["shrink 2048"])

    integers = []
    for total in [2, 3, 7, 255, 256, 257, 65535, 65536, 65537, 0x7FFFFFFF, 0xFFFFFFFF]:
        for value in [0, 1, total // 2, total - 1]:
            integers.append(f"uint {value} {total}")
    generate("uint_boundaries", integers, 128)
    generate("raw_boundaries", [f"bits {(1 << bits) - 1} {bits}" for bits in range(1, 26)] + ["shrink 41"], 256)
    generate("empty", [], 1)
    generate("zero_storage", ["shrink 0"], 1)
    generate("overflow", ["uint 255 256"] * 16 + ["bits 33554431 25"] * 4, 1)
    generate("corrupted_uint", ["uint 256 257"] * 20, 1)
    generate("overlap", ["bit 1 2", "bits 127 7"], 1)
    generate("patch_error", ["patch 1 1", "uint 5 13"], 8)
    for extra in [0, 5, 16, 64]:
        generate(f"patch_{extra}", ["bin 5 6 3"] + ["bit 0 1"] * extra + ["patch 5 3"], 16)


if __name__ == "__main__":
    main()
