"""Verify the pinned PyTorch reference snapshot byte-for-byte."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = root / "third_party" / "pytorch"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative_path, expected in manifest["files"].items():
        path = snapshot / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            failures.append(f"hash mismatch: {relative_path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(manifest['files'])} pinned files at {manifest['commit']}")


if __name__ == "__main__":
    main()
