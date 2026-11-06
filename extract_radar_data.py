"""Read and verify the one CSV required from the manually downloaded RADAR TAR."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile

DATASET_DOI = "10.22000/8bnbhqccf4aq7kea"
EXPECTED_SHA256 = "c8827a378879ce9d8d1ed7539e451b7b741eec30c5cc070487e3f467a4a42a6b"
ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_verified_csv(archive: Path, destination: Path) -> Path:
    """Stream a regular CSV member to a fixed path; never extract archive paths."""
    archive = archive.expanduser().resolve(strict=True)
    destination = destination.resolve()
    destination.relative_to(ROOT)
    if destination.exists():
        if digest(destination) != EXPECTED_SHA256:
            raise ValueError(f"Refusing to overwrite a different CSV: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Remove or inspect the existing partial file first: {partial}")
    try:
        with tarfile.open(archive, mode="r:*") as tar:
            matches = [m for m in tar.getmembers() if m.isfile() and
                       (m.name == "master_dataset.csv" or m.name.endswith("/master_dataset.csv"))]
            if len(matches) != 1:
                raise ValueError(f"Expected one regular master_dataset.csv member; found {len(matches)}")
            member = tar.extractfile(matches[0])
            if member is None:
                raise ValueError("The CSV member could not be read")
            h = hashlib.sha256()
            with member, partial.open("xb") as out:
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    h.update(block)
                    out.write(block)
            if h.hexdigest() != EXPECTED_SHA256:
                raise ValueError("The CSV does not match RADAR Version 1.0.0; no data were accepted")
        partial.replace(destination)
        return destination
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Official RADAR Version 1.0.0 TAR, downloaded manually")
    args = parser.parse_args()
    result = extract_verified_csv(args.archive, ROOT / "data" / "master_dataset.csv")
    print(f"Verified {DATASET_DOI}: {result}")


if __name__ == "__main__":
    main()
