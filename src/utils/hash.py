from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def hash_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()
