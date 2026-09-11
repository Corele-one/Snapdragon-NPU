#!/usr/bin/env python3
"""Create or verify a deterministic SHA256 inventory without following links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_lineage import sha256


def inventory(root: Path) -> dict[str, object]:
    excluded = {".git", "android_ReleaseG_aarch64", "hexagon_ReleaseG_toolv19_v79", "__pycache__"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or any(part in excluded for part in path.parts):
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"root": str(root.resolve()), "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    value = inventory(args.root.resolve())
    if args.check:
        expected = json.loads(args.check.read_text())
        if expected != value:
            print("tree hash inventory changed")
            return 3
        print("tree hash inventory unchanged")
        return 0
    if not args.output:
        parser.error("--output is required unless --check is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite inventory: {args.output}")
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
