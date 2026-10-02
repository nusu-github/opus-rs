#!/usr/bin/env python3
"""Compare every RFC 8251 packet against the pinned C decoder, without tolerance."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
import subprocess
import tempfile
import threading
from pathlib import Path

from compare_codec import ROOT, difference

ARCHIVE_URL = "https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz"
ARCHIVE_SHA256 = "6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55"


def packets(path: Path):
    with path.open("rb") as file:
        while header := file.read(8):
            if len(header) != 8:
                raise ValueError(f"{path}: truncated packet header")
            length, final_range = struct.unpack(">II", header)
            packet = file.read(length)
            if len(packet) != length:
                raise ValueError(f"{path}: truncated packet payload")
            yield packet, final_range


def compare(reference: Path, candidate: Path, arguments: list[str], timeout: int) -> dict:
    processes = [subprocess.Popen([str(executable.resolve()), "decode", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for executable in (reference, candidate)]
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        for process in processes:
            process.kill()

    timer = threading.Timer(timeout, stop)
    timer.start()
    hashes = [hashlib.sha256(), hashlib.sha256()]
    first = None
    count = 0
    try:
        for count, lines in enumerate(itertools.zip_longest(processes[0].stdout, processes[1].stdout), start=1):
            for index, line in enumerate(lines):
                if line is not None:
                    hashes[index].update(line.encode())
            if first is None and lines[0] != lines[1]:
                first = "output record count differs" if None in lines else difference(*lines)
                first = f"packet {count - 1}: {first}"
        for process in processes:
            process.wait()
        errors = [process.stderr.read().strip() for process in processes]
        if timed_out.is_set():
            first = f"timed out after {timeout} seconds"
        elif any(process.returncode for process in processes):
            first = first or f"process failure: reference={processes[0].returncode}, candidate={processes[1].returncode}"
        return {"packets": count, "reference_sha256": hashes[0].hexdigest(), "candidate_sha256": hashes[1].hexdigest(), "difference": first, "stderr": errors}
    finally:
        timer.cancel()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=ROOT / "target/reference/rfc8251/opus_newvectors")
    parser.add_argument("--reference", type=Path, default=ROOT / "target/reference/opus-reference")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", type=int, default=[48000])
    parser.add_argument("--channels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, help="Limit each file to its first N packets for debugging")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--profile", default="scalar-float-no-fma-no-approx-lrintf", help="Reference build profile recorded in the manifest")
    args = parser.parse_args()
    files = sorted(args.vectors.glob("testvector*.bit"))
    if len(files) != 12:
        parser.error(f"expected 12 official .bit files in {args.vectors}, found {len(files)}")
    results = []
    with tempfile.TemporaryDirectory(prefix="opus-rfc8251-") as directory:
        script = Path(directory) / "input.packets"
        for rate in args.rates:
            for channels in args.channels:
                for vector in files:
                    selected = list(itertools.islice(packets(vector), args.limit))
                    script.write_text("".join(f"{rate * 120 // 1000} 0 {packet.hex()}\n" for packet, _ in selected))
                    result = compare(args.reference, args.candidate, [str(rate), str(channels), str(script)], args.timeout)
                    result.update({"vector": vector.name, "rate": rate, "channels": channels, "bitstream_sha256": hashlib.sha256(vector.read_bytes()).hexdigest()})
                    results.append(result)
                    status = "FAIL" if result["difference"] else "PASS"
                    print(f"{status} {vector.name}/{rate}/{channels}: {result['packets']} packets" + (f"; {result['difference']}" if result["difference"] else ""), flush=True)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps({"archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256, "profile": args.profile, "limit": args.limit, "cases": results}, indent=2) + "\n")
    failures = sum(result["difference"] is not None for result in results)
    print(f"{len(results)} official vector cases; {failures} failures")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
