from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from greenlens.utils.labels import canonical_key, humanize, is_healthy


SEVERITY_LABEL = {
    "en": {"none": "None", "moderate": "Moderate", "high": "High", "very_high": "Very high"},
    "ar": {"none": "لا يوجد", "moderate": "متوسطة", "high": "عالية", "very_high": "عالية جدًا"},
}


class DiseaseDB:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        if path is None:
            # Default location relative to project root
            root = Path(__file__).resolve().parents[3]
            path = root / "configs" / "disease_info.json"

        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def disclaimer(self, lang: str = "ar") -> str:
        return self.data.get("_meta", {}).get(lang, "")

    def has(self, class_name: str) -> bool:
        return canonical_key(class_name) in self.data

    def lookup(self, class_name: str, lang: str = "ar") -> dict:
        """Lookup disease information by class name with language-specific details."""
        key = canonical_key(class_name)
        entry = self.data.get(key)

        if entry is None:
            plant, cond = humanize(class_name)
            return {
                "known": False,
                "plant": plant,
                "disease": cond,
                "healthy": is_healthy(class_name),
                "severity": "unknown",
                "severity_label": "-",
                "cause": "-",
                "description": (
                    "لا توجد معلومات تفصيلية لهذه الفئة بعد."
                    if lang == "ar"
                    else "No detailed information is available for this class yet."
                ),
                "symptoms": [],
                "treatment": [],
                "prevention": [],
            }

        get_lang = lambda k: entry[k].get(lang, entry[k].get("en", ""))
        severity = entry.get("severity", "none")
        severity_map = SEVERITY_LABEL.get(lang, SEVERITY_LABEL["en"])

        return {
            "known": True,
            "plant": get_lang("plant"),
            "disease": get_lang("disease"),
            "healthy": entry.get("healthy", False),
            "severity": severity,
            "severity_label": severity_map.get(severity, severity),
            "cause": get_lang("cause"),
            "description": get_lang("description"),
            "symptoms": get_lang("symptoms"),
            "treatment": get_lang("treatment"),
            "prevention": get_lang("prevention"),
        }
