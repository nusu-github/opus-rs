#!/usr/bin/env python3
"""Extract QEXT coefficient and 96 kHz static tables from the pinned C source."""
from pathlib import Path
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = "503d81b138d76621aae4b12786e90de48aa8db3a"
SOURCE = ROOT / "target/reference/source"


def preprocess(header, fixed):
    flags = ["-DFIXED_POINT"] if fixed else []
    return subprocess.check_output([os.environ.get("CC", "cc"), "-E", "-P", "-x", "c",
        "-DOPUS_BUILD", "-DVAR_ARRAYS", "-DENABLE_QEXT", *flags, "-I", str(SOURCE / "include"),
        "-I", str(SOURCE / "celt"), "-"], input=f'#include "{header}"\n', text=True)


def values(source, name):
    body = re.search(r"\b"+name+r"\[\d+\]\s*=\s*\{(.*?)\};", source, re.S).group(1)
    return re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", body)


def array(name, kind, data):
    lines = [f"pub(crate) const {name}: [{kind}; {len(data)}] = ["]
    for i in range(0, len(data), 8):
        lines.append("    "+", ".join(data[i:i+8])+",")
    return "\n".join(lines+[" ];\n"])


def main():
    assert (SOURCE / ".reference-revision").read_text().strip() == REVISION
    fixed = preprocess("static_modes_fixed.h", True)
    floating = preprocess("static_modes_float.h", False)
    out = [f"// Extracted from Opus {REVISION}; BSD-3-Clause.\n",
           "// Regenerate with tools/reference/extract_qext_tables.py.\n"]
    for name in ["window120", "window240", "mdct_twiddles960", "mdct_twiddles1920"]:
        out.append(array(name.upper(), "i32", values(fixed, name)))
    for name in ["fft_twiddles48000_960", "fft_twiddles96000_1920"]:
        data = values(fixed, name)
        pairs = [f"({data[i]}, {data[i+1]})" for i in range(0,len(data),2)]
        out.append(array(name.upper(), "(i32, i32)", pairs))
    for name,kind in [("qext_cache_index50","i16"),("qext_cache_bits50","u8"),("qext_cache_caps50","u8")]:
        out.append(array(name.upper(),kind,values(fixed,name)))
    (ROOT/"rust/src/celt/qext_fixed_tables.rs").write_text("\n".join(out))

    out = [f"// Extracted from Opus {REVISION}; BSD-3-Clause.\n",
        "// Regenerate with tools/reference/extract_qext_tables.py.\n",
        "use super::{KissFftCpx, KissFftState, MdctLookup};\n"]
    out.append(array("WINDOW_240", "f32", [v+"f32" for v in values(floating,"window240")]))
    data = values(floating,"fft_twiddles96000_1920")
    out.append(array("FFT_TWIDDLES", "KissFftCpx", [f"KissFftCpx::new({data[i]}f32, {data[i+1]}f32)" for i in range(0,len(data),2)]))
    data = values(floating,"mdct_twiddles1920")
    planar = []; offset = 0
    for size in (1920,960,480,240):
        part=data[offset:offset+size];planar.extend(part[1::2]+part[::2]);offset+=size
    out.append(array("MDCT_TWIDDLES", "f32", [v+"f32" for v in planar]))
    for n in (960,480,240,120):
        out.append(array(f"BITREV_{n}","usize",values(floating,f"fft_bitrev{n}")))
    factors={960:[5,192,3,64,4,16,4,4,4,1],480:[5,96,3,32,4,8,2,4,4,1],240:[5,48,3,16,4,4,4,1],120:[5,24,3,8,2,4,4,1]}
    out.append("static FFT_STATES: [KissFftState; 4] = [")
    for shift,n in enumerate((960,480,240,120)):
        shift_text="None" if shift==0 else f"Some({shift})"
        out.append(f"KissFftState::from_static({n}, 1.0/{n}.0, {shift_text}, &{factors[n]}, &BITREV_{n}, &FFT_TWIDDLES),")
    out.append("];\npub(crate) static MDCT: MdctLookup = MdctLookup::from_static(3840, 3, &FFT_STATES, &FFT_STATES, &MDCT_TWIDDLES, &[0,1920,2880,3360,3600]);\n")
    (ROOT/"rust/src/celt/static_mode_96000_1920.rs").write_text("\n".join(out))


if __name__ == "__main__":
    main()
