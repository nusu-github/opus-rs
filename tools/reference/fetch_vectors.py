#!/usr/bin/env python3
"""Download and checksum-pin the official RFC 8251 Opus packet corpus."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

from compare_codec import ROOT
from compare_vectors import ARCHIVE_SHA256, ARCHIVE_URL


def main() -> None:
    directory = ROOT / "target/reference"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "opus_testvectors-rfc8251.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        try:
            with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != ARCHIVE_SHA256:
                raise RuntimeError("RFC 8251 archive checksum mismatch")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError(f"{archive}: RFC 8251 archive checksum mismatch")
    output = directory / "rfc8251/opus_newvectors"
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for index in range(1, 13):
            name = f"testvector{index:02d}.bit"
            member = source.getmember(f"opus_newvectors/{name}")
            if not member.isfile():
                raise RuntimeError(f"Unexpected archive entry: {name}")
            with source.extractfile(member) as input_file, (output / name).open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    print(output)


if __name__ == "__main__":
    main()
