from __future__ import annotations

import re
from typing import Literal

ProductKind = Literal["instrument", "report", "bundle"]

ASSESSMENT_TAGS = frozenset(
    {
        "personality",
        "skills",
        "development",
        "ability",
        "knowledge",
        "simulation",
        "situational_judgment",
        "safety",
        "sales",
        "leadership",
        "report",
    }
)


def infer_product_kind(name: str, description: str) -> ProductKind:
    name_lower = name.lower()
    desc_lower = description.lower()

    if "bundle" in name_lower or "focus 8.0" in name_lower or " - safety &" in name_lower:
        return "bundle"
    if name_lower.endswith(" report") or " report " in f" {name_lower} ":
        return "report"
    if desc_lower.startswith("this report") or "designed to be given" in desc_lower:
        return "report"
    if "who have completed the" in desc_lower and "report" in desc_lower:
        return "report"
    return "instrument"


def infer_assessment_tags(name: str, description: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    tags: set[str] = set()
    name_lower = name.lower()
    desc_lower = description.lower()
    kind = infer_product_kind(name, description)

    if kind == "report":
        tags.add("report")
    if kind == "bundle":
        tags.add("personality")

    key_set = set(keys)
    if "Personality & Behavior" in key_set:
        tags.add("personality")
    if "Knowledge & Skills" in key_set:
        tags.add("knowledge")
    if "Competencies" in key_set:
        tags.add("skills")
    if "Development & 360" in key_set:
        tags.add("development")
    if "Ability & Aptitude" in key_set:
        tags.add("ability")
    if "Simulations" in key_set:
        tags.add("simulation")
    if "Biodata & Situational Judgment" in key_set:
        tags.add("situational_judgment")

    if "opq32r" in name_lower or "occupational personality questionnaire" in name_lower:
        tags.add("personality")
    if "global skills assessment" in name_lower:
        tags.add("skills")
    if "global skills development" in name_lower:
        tags.update({"skills", "development", "report"})
    if "graduate scenarios" in name_lower:
        tags.add("situational_judgment")
    if re.search(r"\bverify\b", name_lower):
        tags.add("ability")
    if "dependability" in name_lower or "dsi" in name_lower or "safety" in name_lower:
        tags.add("safety")
    if "sales" in name_lower:
        tags.add("sales")
    if "leadership" in name_lower or "leadership" in desc_lower:
        tags.add("leadership")
    if "development" in name_lower or "re-skill" in desc_lower or "growth skills" in desc_lower:
        tags.add("development")

    return tuple(sorted(tags & ASSESSMENT_TAGS))
