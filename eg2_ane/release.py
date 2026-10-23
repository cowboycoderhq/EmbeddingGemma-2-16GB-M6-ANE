"""Write release.json for a model folder: every file with its size and sha256.
  python -m eg_ane.release MODEL_DIR"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = Path(sys.argv[1])
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name != "release.json" and ".cache" not in p.parts)
    rel = {"name": d.name, "files": [{"path": str(p.relative_to(d)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                     for p in files]}
    (d / "release.json").write_text(json.dumps(rel, indent=1) + "\n")
    print(f"release.json: {len(files)} files, {sum(f['bytes'] for f in rel['files']) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
