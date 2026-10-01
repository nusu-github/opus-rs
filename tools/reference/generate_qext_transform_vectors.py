#!/usr/bin/env python3
"""Generate the scalar/PFA transform oracle fixtures from pinned C."""
from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parents[2]
profile=sys.argv[1] if len(sys.argv)>1 else "fixed-qext"
assert profile in ("float","qext","qext-pfa","fixed-qext","fixed-qext-pfa")
source=ROOT/("target/reference/source" if profile == "float" else f"target/reference-{profile}/source")
assert (source / ".reference-revision").read_text().strip() == "503d81b138d76621aae4b12786e90de48aa8db3a"
binary=source.parent/"qext-transform-vectors"
flags=["-DENABLE_PFA"] if profile.endswith("pfa") else []
fixed = profile.startswith("fixed-")
if fixed: flags.append("-DFIXED_POINT")
if "qext" in profile: flags.append("-DENABLE_QEXT")
subprocess.run([os.environ.get("CC","cc"),"-O2","-std=c99","-ffp-contract=off","-fno-fast-math","-DOPUS_BUILD",*flags,"-I",str(source/"include"),"-I",str(source/"celt"),str(ROOT/("tools/reference/fixed_transform_vectors.c" if fixed else "tools/reference/pfa_transform_vectors.c")),str(source/"libopus.a"),"-lm","-o",str(binary)],check=True)
output=ROOT/"rust/tests/fixtures/reference"
for rate,size in ([(48000,960)] if profile == "float" else [(48000,960),(96000,1920)]):
 subprocess.run([str(binary),str(output/f"{profile}-{rate}-mdct.bin"),str(output/f"{profile}-{rate}-fft.bin"),str(rate),str(size)],check=True)
