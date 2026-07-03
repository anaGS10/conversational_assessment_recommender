"""Static grouping of the SHL catalog by domain and assessment type.

The grouping is built once (see scripts/build_groups.py) and stored in
data/catalog_groups.json. At query time Gemini picks the relevant groups, and we
search only within the items that belong to those groups.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.catalog import CatalogItem, get_catalog, get_catalog_by_id
from app.config import settings
from app.llm import LLMError, chat_completion

# Domain groups — assigned once by Gemini from each item's name/keys/description.
DOMAIN_GROUPS: dict[str, str] = {
    "technical": "Technical, engineering, IT, software, data, or QA roles and skills (programming languages, databases, cloud, frameworks, testing).",
    "healthcare": "Healthcare, medical, clinical, pharmaceutical, or nursing roles and knowledge.",
    "finance": "Finance, accounting, banking, or financial-analysis roles.",
    "sales": "Sales, selling, account management, or revenue-generating customer-facing roles.",
    "contact_centre": "Contact centre, call centre, or customer-service agent roles.",
    "leadership": "Leadership, management, executive, or director-level assessment.",
    "safety": "Safety-critical, dependability, or industrial-safety roles.",
    "administrative": "Administrative, clerical, office-support, or data-entry roles.",
    "manufacturing_industrial": "Manufacturing, industrial, plant, or mechanical operator roles.",
    "graduate_entry": "Graduate, early-career, or entry-level hiring where candidates have little work experience.",
}

# Type groups — assigned deterministically from the catalog 'keys' field.
KEY_TO_TYPE_GROUP: dict[str, str] = {
    "Knowledge & Skills": "knowledge_and_skills",
    "Personality & Behavior": "personality_and_behavior",
    "Ability & Aptitude": "ability_and_aptitude",
    "Simulations": "simulations",
    "Biodata & Situational Judgment": "biodata_and_situational_judgment",
    "Competencies": "competencies",
    "Development & 360": "development_and_360",
    "Assessment Exercises": "assessment_exercises",
}

DOMAIN_GROUP_NAMES: tuple[str, ...] = tuple(DOMAIN_GROUPS)
TYPE_GROUP_NAMES: tuple[str, ...] = tuple(dict.fromkeys(KEY_TO_TYPE_GROUP.values()))
ALL_GROUP_NAMES: tuple[str, ...] = DOMAIN_GROUP_NAMES + TYPE_GROUP_NAMES

# Seniority buckets → substrings found in the catalog's job_levels field.
SENIORITY_JOB_LEVELS: dict[str, tuple[str, ...]] = {
    "graduate": ("graduate", "entry-level", "student"),
    "entry": ("entry-level", "graduate", "general population"),
    "mid": ("mid-professional", "professional individual contributor", "supervisor"),
    "senior": ("professional individual contributor", "mid-professional", "front line manager"),
    "manager": ("manager", "front line manager", "supervisor"),
    "director": ("director", "executive"),
    "executive": ("executive", "director"),
}

_GROUPS_FILENAME = "catalog_groups.json"


def type_groups_for_item(item: CatalogItem) -> list[str]:
    """Deterministic assessment-type groups from the catalog 'keys' field."""
    groups: list[str] = []
    for key in item.keys:
        group = KEY_TO_TYPE_GROUP.get(key.strip())
        if group and group not in groups:
            groups.append(group)
    return groups


def _groups_path(data_dir: Path | None = None) -> Path:
    return (data_dir or settings.data_dir) / _GROUPS_FILENAME


@lru_cache(maxsize=1)
def load_groups() -> dict[str, tuple[str, ...]]:
    """Return {group_name: (entity_id, ...)} from the built grouping file."""
    path = _groups_path()
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    groups = raw.get("groups", raw) if isinstance(raw, dict) else {}
    by_name = get_catalog_by_name_lower_ids()
    result: dict[str, tuple[str, ...]] = {}
    for name, entries in groups.items():
        ids: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            entity_id: str | None = None
            if isinstance(entry, dict):
                if entry.get("entity_id") is not None:
                    entity_id = str(entry["entity_id"])
                elif entry.get("name"):
                    entity_id = by_name.get(str(entry["name"]).lower())
            elif isinstance(entry, str):
                entity_id = entry if entry.isdigit() else by_name.get(entry.lower())
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                ids.append(entity_id)
        result[name] = tuple(ids)
    return result


@lru_cache(maxsize=1)
def get_catalog_by_name_lower_ids() -> dict[str, str]:
    return {item.name.lower(): item.entity_id for item in get_catalog()}


def items_in_groups(names: list[str]) -> list[CatalogItem]:
    """Union of catalog items belonging to any of the given groups."""
    groups = load_groups()
    by_id = get_catalog_by_id()
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for name in names:
        for entity_id in groups.get(name, ()):
            if entity_id in seen:
                continue
            item = by_id.get(entity_id)
            if item:
                seen.add(entity_id)
                items.append(item)
    return items


def filter_by_seniority(items: list[CatalogItem], seniority: str | None) -> list[CatalogItem]:
    """Narrow to items whose job_levels match the requested seniority.

    Non-destructive: if nothing matches (or seniority is unknown), the original
    list is returned unchanged.
    """
    if not seniority:
        return items
    seniority_key = seniority.strip().lower()
    wanted = SENIORITY_JOB_LEVELS.get(seniority_key)
    if not wanted:
        return items

    matched = [
        item
        for item in items
        if any(
            any(token in level.lower() for token in wanted)
            for level in item.job_levels
        )
    ]
    if seniority_key in {"executive", "director"}:
        matched = [
            item
            for item in matched
            if not re.search(r"\bentry[- ]level\b", item.name.lower())
        ]
    return matched or items


def detect_seniority(conversation: str) -> str:
    """Infer seniority bucket from conversation text (no LLM, no product names)."""
    text = conversation.lower()
    if re.search(
        r"\b(cxo|ceo|cfo|cto|coo|chief\s+\w+\s+officer|c-suite|director|executive|vp\b|vice president)\b",
        text,
    ):
        return "executive"
    if re.search(r"\b(15\+?\s*years?|senior leadership)\b", text):
        return "executive"
    if re.search(r"\b(manager|management role|front[- ]line manager)\b", text) and not re.search(
        r"\b(director|executive|cxo)\b", text
    ):
        return "manager"
    if re.search(r"\b(entry[- ]level|no work experience|no experience|fresh graduate)\b", text):
        return "entry"
    if re.search(r"\bgraduate\b", text) and not re.search(r"\b(graduate management|graduate scheme)\b", text):
        return "graduate"
    if re.search(r"\b(senior|sr\.)\b", text):
        return "senior"
    return ""


def fallback_groups_from_signals(signals) -> list[str]:
    """Map query signals to catalog groups when LLM routing returns nothing."""
    groups: list[str] = []
    if signals.wants_leadership:
        groups.extend(["leadership", "personality_and_behavior"])
    if signals.wants_finance:
        groups.extend(["finance", "knowledge_and_skills", "ability_and_aptitude"])
    if signals.wants_contact_centre:
        groups.extend(["contact_centre", "simulations", "personality_and_behavior"])
    if signals.wants_sales:
        groups.extend(["sales", "personality_and_behavior"])
    if signals.wants_safety:
        groups.append("safety")
    if signals.is_admin_assistant:
        groups.extend(["administrative", "knowledge_and_skills"])
    if signals.wants_development:
        groups.append("development_and_360")
    if signals.wants_knowledge or signals.wants_simulation:
        groups.append("knowledge_and_skills")
    if signals.wants_ability:
        groups.append("ability_and_aptitude")
    if signals.wants_personality or signals.prefer_instruments:
        groups.append("personality_and_behavior")
    if signals.wants_situational:
        groups.append("biodata_and_situational_judgment")
    return list(dict.fromkeys(groups))


def deterministic_domain_groups(item: CatalogItem) -> list[str]:
    """Assign domain groups from catalog metadata when LLM classification is unavailable."""
    domains: list[str] = []
    tags = set(item.assessment_tags)
    name = item.name.lower()
    desc = item.description.lower()

    if "leadership" in tags or "leadership" in name:
        domains.append("leadership")
    if "sales" in tags or re.search(r"\bsales\b", name):
        domains.append("sales")
    if "safety" in tags or "dependability" in name or re.search(r"\bdsi\b", name):
        domains.append("safety")
    if re.search(r"\b(contact center|contact centre|call cent)", f"{name} {desc}"):
        domains.append("contact_centre")
    if re.search(r"\b(financial|accounting|banking)\b", name):
        domains.append("finance")
    if re.search(r"\b(medical|healthcare|pharma|nursing|clinical|hospital)\b", name):
        domains.append("healthcare")
    if re.search(r"\b(admin|clerical|data entry|office support)\b", name):
        domains.append("administrative")
    if re.search(r"\bentry[- ]level\b", name) and "solution" in name:
        domains.append("graduate_entry")
    if ("knowledge" in tags or "simulation" in tags) and re.search(
        r"\b(java|python|sql|\.net|programming|software|networking|linux|coding|devops|cloud|aws)\b",
        name,
    ):
        domains.append("technical")
    if re.search(
        r"\b(engineering|manufacturing|industrial|plant|mechanical|chemical|aeronautical)\b",
        name,
    ):
        domains.append("manufacturing_industrial")

    return [d for d in domains if d in DOMAIN_GROUPS]


GROUP_ROUTING_PROMPT = """You route a hiring conversation to catalog groups so we can search only within them.

GROUPS (choose the ones relevant to the user's need):
{group_catalog}

Also extract:
- skills: concrete skills/tools/knowledge mentioned (e.g. "java", "excel", "spring boot").
- seniority: one of graduate, entry, mid, senior, manager, director, executive — or "" if unclear.

Rules:
- Use ONLY group names from the list above.
- Pick every group that applies (a personality test for leaders → both "leadership" and "personality_and_behavior").
- If the conversation is too vague to route, return empty groups.

Return JSON only:
{{
  "groups": ["group1", "group2"],
  "skills": ["skill1"],
  "seniority": ""
}}
"""


def _group_catalog_text() -> str:
    lines = ["Domain groups:"]
    for name, desc in DOMAIN_GROUPS.items():
        lines.append(f"- {name}: {desc}")
    lines.append("Assessment-type groups:")
    for name in TYPE_GROUP_NAMES:
        lines.append(f"- {name}")
    return "\n".join(lines)


async def route_query_to_groups(conversation: str) -> dict:
    """Return {'groups': [...], 'skills': [...], 'seniority': str} for a conversation."""
    valid = set(ALL_GROUP_NAMES)
    try:
        result = await chat_completion(
            GROUP_ROUTING_PROMPT.format(group_catalog=_group_catalog_text()),
            f"CONVERSATION:\n{conversation}",
            temperature=0.0,
        )
        groups = [
            str(g).strip()
            for g in result.get("groups", [])
            if isinstance(g, str) and str(g).strip() in valid
        ]
        skills = [str(s).strip() for s in result.get("skills", []) if isinstance(s, str) and str(s).strip()]
        seniority = str(result.get("seniority", "") or "").strip()
        return {"groups": groups, "skills": skills, "seniority": seniority}
    except (LLMError, Exception):
        return {"groups": [], "skills": [], "seniority": ""}


DOMAIN_CLASSIFY_PROMPT = """You label SHL assessments with hiring DOMAINS.

DOMAINS (choose zero or more per assessment, ONLY from this list):
{domain_catalog}

Rules:
- Base labels only on each assessment's name, keys, and description.
- An assessment may belong to multiple domains, or to none (return []).
- Do NOT invent domains outside the list.

Return JSON only:
{{
  "labels": {{
    "<entity_id>": ["domain1", "domain2"]
  }}
}}
"""


def _domain_catalog_text() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in DOMAIN_GROUPS.items())


def _classify_payload(items: list[CatalogItem]) -> str:
    blocks: list[str] = []
    for item in items:
        desc = (item.description or "").strip().replace("\n", " ")
        if len(desc) > 300:
            desc = desc[:300] + "…"
        keys = ", ".join(item.keys) or "N/A"
        blocks.append(
            f"entity_id: {item.entity_id}\nname: {item.name}\nkeys: {keys}\ndescription: {desc}"
        )
    return "\n\n".join(blocks)


async def classify_item_domains(items: list[CatalogItem]) -> dict[str, list[str]]:
    """Ask Gemini to assign domain labels to a batch of catalog items."""
    valid = set(DOMAIN_GROUP_NAMES)
    result = await chat_completion(
        DOMAIN_CLASSIFY_PROMPT.format(domain_catalog=_domain_catalog_text()),
        f"ASSESSMENTS:\n{_classify_payload(items)}",
        temperature=0.0,
    )
    labels = result.get("labels", {}) if isinstance(result, dict) else {}
    cleaned: dict[str, list[str]] = {}
    for entity_id, domains in labels.items():
        if not isinstance(domains, list):
            continue
        picked = [str(d).strip() for d in domains if isinstance(d, str) and str(d).strip() in valid]
        cleaned[str(entity_id)] = sorted(dict.fromkeys(picked))
    return cleaned
