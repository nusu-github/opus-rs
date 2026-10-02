#!/usr/bin/env python3
"""Extract the canonical Q15 MDCT table from the pinned C source revision."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REVISION = "503d81b138d76621aae4b12786e90de48aa8db3a"


def main() -> None:
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:celt/static_modes_fixed.h"], cwd=ROOT, text=True
    )
    body = source.split("static const celt_coef mdct_twiddles960[1800] = {", 1)[1]
    body = body.split("};", 1)[0].split("#else", 1)[1].split("#endif", 1)[0]
    values = [int(value) for value in re.findall(r"-?\d+", body)]
    assert len(values) == 1800
    lines = [
        f"// Extracted from {REVISION}:celt/static_modes_fixed.h.",
        "// Interleaved negative-sine and cosine pairs, in Q15.",
        "// Regenerate with tools/reference/extract_fixed_mdct_table.py.",
        "pub(super) const MDCT_TWIDDLES: [i16; 1800] = [",
    ]
    for i in range(0, len(values), 12):
        lines.append("    " + ", ".join(map(str, values[i:i + 12])) + ",")
    lines.append("];")
    (ROOT / "rust/src/celt/mdct_twiddles_fixed_48000_960.rs").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
