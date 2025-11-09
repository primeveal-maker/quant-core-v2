from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ConfigBundle:
    path: Path
    data: Dict[str, Any]

    @property
    def hash(self) -> str:
        payload = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


def _parse_lines(lines: List[str], indent: int = 0) -> Any:
    result: Dict[str, Any] = {}
    current_key: str | None = None
    while lines:
        line = lines[0]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.pop(0)
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if stripped.startswith("-"):
            if current_key is None:
                raise ValueError("List item without preceding key")
            value = stripped[1:].strip()
            lines.pop(0)
            if value:
                result.setdefault(current_key, []).append(_coerce(value))
            else:
                nested = _parse_lines(lines, indent=current_indent + 2)
                result.setdefault(current_key, []).append(nested)
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            lines.pop(0)
            current_key = key
            if value:
                result[key] = _coerce(value)
            else:
                result[key] = _parse_lines(lines, indent=current_indent + 2)
        else:
            lines.pop(0)
    return result


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [item.strip() for item in inner.split(",")]
        return [_coerce(part) for part in parts]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: Path) -> ConfigBundle:
    lines = path.read_text(encoding="utf-8").splitlines()
    data = _parse_lines(lines)
    return ConfigBundle(path=path, data=data)
