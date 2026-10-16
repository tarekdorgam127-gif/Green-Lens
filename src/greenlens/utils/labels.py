from __future__ import annotations

import re

# Applied in order on the normalized string
_ALIASES = [
    ("_two_spotted_spider_mite", ""),
    ("tomato_tomato_", "tomato_"),
    ("pepper_bell_", "pepper_"),
    ("yellowleaf", "yellow_leaf"),
]


def canonical_key(class_name: str) -> str:
    """Convert class name from dataset/model into a canonical lookup key."""
    s = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
    for old, new in _ALIASES:
        s = s.replace(old, new)
    return re.sub(r"_+", "_", s).strip("_")


def humanize(class_name: str) -> tuple[str, str]:
    """Best-effort (plant, condition) from a raw folder or class name."""
    if "___" in class_name:
        plant, cond = class_name.split("___", 1)
    else:
        parts = [p for p in re.split(r"_+", class_name) if p]
        plant, cond = parts[0], " ".join(parts[1:])
    clean = lambda s: re.sub(r"[_\s]+", " ", s).strip()
    return clean(plant), clean(cond) or "Unknown"


def is_healthy(class_name: str) -> bool:
    """Check if class indicates healthy plant."""
    return "healthy" in class_name.lower()
