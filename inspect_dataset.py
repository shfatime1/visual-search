"""Inspect ZIP safely; extract only absent files without overwriting originals."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def extract(archive: Path):
    report = {"archive": str(archive), "entries": 0, "extracted": 0, "identical": 0}
    with ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/")]
        csvs = [m for m in members if PurePosixPath(m.filename).name == "products.csv"]
        if len(csvs) != 1:
            raise ValueError("Archive must contain exactly one products.csv")
        prefix = str(PurePosixPath(csvs[0].filename).parent)
        prefix = "" if prefix == "." else prefix + "/"
        pending = []
        for m in members:
            if not m.filename.startswith(prefix):
                continue
            relative = PurePosixPath(m.filename[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts or any(":" in p or "\\" in p for p in relative.parts):
                raise ValueError("Unsafe archive path")
            if m.file_size > 20 * 1024 * 1024:
                raise ValueError("Unexpectedly large archive entry")
            target = (ROOT / "data" / str(relative)).resolve()
            if not target.is_relative_to((ROOT / "data").resolve()):
                raise ValueError("Unsafe archive destination")
            content = zf.read(m)
            report["entries"] += 1
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(content).digest():
                    raise ValueError(f"Existing original differs: {relative}; refusing overwrite")
                report["identical"] += 1
            else:
                pending.append((target, content))
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            report["extracted"] += 1
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive), indent=2))
