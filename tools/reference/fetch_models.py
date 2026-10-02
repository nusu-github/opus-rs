#!/usr/bin/env python3
"""Fetch and verify the exact model archive required by the pinned C revision."""
import hashlib
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SHA256 = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e"
NAME = f"opus_data-{SHA256}.tar.gz"
DIRECTORY = ROOT / "target/reference/models"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive = DIRECTORY / NAME
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(f"https://media.xiph.org/opus/models/{NAME}", temporary)
        if digest(temporary) != SHA256:
            raise SystemExit("Downloaded model archive failed SHA-256 verification.")
        temporary.replace(archive)
    if digest(archive) != SHA256:
        raise SystemExit("Cached model archive failed SHA-256 verification.")
    with tarfile.open(archive, "r:gz") as source:
        source.extractall(DIRECTORY, filter="data")
    print(DIRECTORY / "dnn")


if __name__ == "__main__":
    main()
