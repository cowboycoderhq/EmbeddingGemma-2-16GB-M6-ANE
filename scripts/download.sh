#!/bin/bash
# Download the model folder from Hugging Face and check every file against release.json.
#   scripts/download.sh [DEST]      default DEST: ~/Models/EmbeddingGemma-2-FP16-M6-ANE
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-$HOME/Models/EmbeddingGemma-2-FP16-M6-ANE}
REPO=cowboycoderhq/EmbeddingGemma-2-FP16-M6-ANE
.venv/bin/python - "$REPO" "$DEST" <<'EOF'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], Path(sys.argv[2])
snapshot_download(repo, local_dir=str(dest))
rel = json.loads((dest / "release.json").read_text())
bad = []
for f in rel["files"]:
    p = dest / f["path"]
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if p.stat().st_size != f["bytes"] or h.hexdigest() != f["sha256"]:
        bad.append(f["path"])
print(f"{len(rel['files'])} files checked, {len(bad)} bad" + (f": {bad}" if bad else ""))
sys.exit(1 if bad else 0)
EOF
echo "model folder: $DEST"
