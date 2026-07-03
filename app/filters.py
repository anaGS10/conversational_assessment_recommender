from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.catalog import CatalogItem

DROP_PATTERNS = [
    r"\b(drop|remove|skip|exclude|without)\s+(the\s+)?opq",
    r"\bno\s+opq\b",
    r"\bdrop\s+personality\b",
    r"\b(drop|remove|skip|exclude)\s+(the\s+)?(verify|rest|personality|opq)",
    r"\b(without|excluding|minus)\s+(the\s+)?(verify|opq|personality|rest)",
]

ADD_PATTERNS = [
    r"\b(?:add|adding|added)\s+([a-z0-9\s\+]+?)(?:\s+and\s+|\s*,\s*|\s+&\s+)([a-z0-9\s\+]+)",
    r"\b(?:add|adding|added|include|including)\s+([a-z0-9\s\+]+)",
]


@dataclass
class QuerySignals:
    wants_personality: bool = False
    wants_skills: bool = False
    wants_development: bool = False
    wants_ability: bool = False
    wants_knowledge: bool = False
    wants_simulation: bool = False
    wants_situational: bool = False
    wants_safety: bool = False
    wants_leadership: bool = False
    wants_sales: bool = False
    wants_contact_centre: bool = False
    wants_finance: bool = False
    exclude_personality: bool = False
    prefer_instruments: bool = False
    prefer_reports: bool = False
    is_admin_assistant: bool = False
    boost_tags: set[str] = field(default_factory=set)
    penalize_tags: set[str] = field(default_factory=set)


def detect_query_signals(conversation: str) -> QuerySignals:
    text = conversation.lower()
    signals = QuerySignals()

    if re.search(r"\b(personality|behaviou?r|opq|motivat)", text):
        signals.wants_personality = True
    if re.search(r"\b(skills?|gsa|global skills|competenc)", text):
        signals.wants_skills = True
    if re.search(
        r"\b(development\b|training plan|re-?skill|talent audit|growth|development gap)\b",
        text,
    ):
        signals.wants_development = True
        signals.boost_tags.update({"skills", "development"})
        signals.penalize_tags.add("personality")
    if re.search(r"\b(cognitive|aptitude|reasoning|verify\s*g\+?|ability)", text):
        signals.wants_ability = True
    if re.search(r"\b(knowledge|technical|java|python|rust|sql|spring|hipaa|excel|word)", text):
        signals.wants_knowledge = True
    if re.search(r"\b(simulation|simulated|call simulation|live coding)", text):
        signals.wants_simulation = True
    if re.search(r"\b(situational judgement|situational judgment|scenarios)", text):
        signals.wants_situational = True
    if re.search(r"\b(safety|dependability|dsi|compliance culture)", text):
        signals.wants_safety = True
    if re.search(r"\b(leadership|cxo|executive|director)", text):
        signals.wants_leadership = True
    if re.search(r"\b(sales|reps?|selling)", text):
        signals.wants_sales = True
    if re.search(r"\b(contact centre|contact center|call centre|call center|inbound calls?)", text):
        signals.wants_contact_centre = True
        signals.boost_tags.update({"simulation", "english", "contact_centre"})
    if re.search(r"\b(finance|financial analyst|financial accounting|accounting|banking)", text):
        signals.wants_finance = True
        signals.boost_tags.update({"finance", "knowledge", "cognitive"})
    if re.search(r"\b(admin\s*assistant|administrative|office\s*admin|executive\s*assistant)", text):
        signals.is_admin_assistant = True
    if re.search(r"\b(selection|benchmark|hiring|recruit|candidate)", text):
        signals.prefer_instruments = True
    if re.search(r"\b(report format|leadership report|development report|sales report)", text):
        signals.prefer_reports = True

    for pattern in DROP_PATTERNS:
        if re.search(pattern, text):
            signals.exclude_personality = True
            signals.penalize_tags.add("personality")

    if signals.wants_development and not signals.wants_personality:
        signals.boost_tags.update({"skills", "development"})
    if signals.wants_personality and not signals.wants_development:
        signals.boost_tags.add("personality")

    return signals


def _skill_score(item: CatalogItem, skills: list[str]) -> float:
    score = 0.0
    name_lower = item.name.lower()
    desc_lower = item.description.lower() if item.description else ""
    keys_lower = [k.lower() for k in item.keys]
    for skill in skills:
        sl = skill.lower()
        if sl in name_lower or sl in desc_lower:
            score += 5.0
        elif any(sl in k for k in keys_lower):
            score += 4.0
    return score


def _domain_score(item: CatalogItem, domains: list[str]) -> float:
    score = 0.0
    tags = set(item.assessment_tags)
    name_lower = item.name.lower()
    for domain in domains:
        if domain == "technical_hiring":
            if ("knowledge" in tags or "ability" in tags) and item.product_kind == "instrument":
                score += 3.0
        elif domain == "leadership":
            if "leadership" in tags:
                score += 2.0
            if "entry level" in name_lower:
                score -= 100.0
        elif domain == "personality":
            if "personality" in tags and item.product_kind == "instrument":
                score += 2.0
        elif domain == "development":
            if "development" in tags or "skills" in tags:
                score += 2.5
            if item.product_kind == "report" and "development" in name_lower:
                score += 2.0
        elif domain == "clerical":
            if "knowledge" in tags and item.product_kind == "instrument":
                if any(t in name_lower for t in ["admin", "clerical", "office", "excel", "word"]):
                    score += 3.0
                else:
                    score += 1.5
        elif domain == "sales":
            if "sales" in tags:
                score += 2.0
        elif domain == "safety":
            if "safety" in tags:
                score += 2.5
        elif domain == "language":
            if item.languages:
                score += 1.0
        elif domain == "contact_centre":
            if re.search(r"\b(contact center|contact centre|phone simulation|customer serv)\b", name_lower):
                score += 3.0
        elif domain == "finance":
            if re.search(r"\b(financial accounting|accounting|numerical reasoning|verify)\b", name_lower):
                score += 3.0
    return score


def _tag_score(item: CatalogItem, signals: QuerySignals, skills: list[str] | None = None, domains: list[str] | None = None) -> float:
    score = 0.0
    item_tags = set(item.assessment_tags)

    if skills:
        score += _skill_score(item, skills)
    if domains:
        score += _domain_score(item, domains)

    for tag in signals.boost_tags:
        if tag in item_tags:
            score += 2.0

    for tag in signals.penalize_tags:
        if tag in item_tags and item.product_kind == "instrument":
            score -= 1.5

    if signals.wants_development:
        if "development" in item_tags or "skills" in item_tags:
            score += 2.5
        if item.product_kind == "report" and "development" in item.name.lower():
            score += 2.0
        if "opq32r" in item.name.lower() and not signals.wants_personality:
            score -= 2.0

    if signals.wants_skills and "skills" in item_tags:
        score += 2.0
    if signals.wants_personality and "personality" in item_tags:
        score += 1.5
    if signals.wants_ability and "ability" in item_tags:
        score += 2.0
    if signals.wants_knowledge and "knowledge" in item_tags:
        score += 2.0
    if signals.wants_simulation and "simulation" in item_tags:
        score += 2.0
    if signals.wants_situational and "situational_judgment" in item_tags:
        score += 2.0
    if signals.wants_safety and "safety" in item_tags:
        score += 2.5
    if signals.wants_leadership and "leadership" in item_tags:
        score += 1.5
    if signals.wants_sales and "sales" in item_tags:
        score += 2.0
    if signals.wants_contact_centre:
        name_lower = item.name.lower()
        if re.search(r"\b(cashier|hotel front desk|retail sales and service)\b", name_lower):
            score -= 10.0
        elif re.search(r"\b(contact center|contact centre|phone simulation|phone solution|svar spoken english|customer serv)\b", name_lower):
            score += 3.0
        elif "entry level" in name_lower and "contact" not in name_lower.replace("centre", "center"):
            score -= 4.0

    if signals.wants_finance:
        name_lower = item.name.lower()
        if re.search(
            r"\b(cashier|sales solution|core java|oracle dba|hotel front desk|technical support|customer service \(general\))\b",
            name_lower,
        ):
            score -= 10.0
        elif re.search(
            r"\b(financial accounting|basic statistics|numerical reasoning|graduate scenarios)\b",
            name_lower,
        ):
            score += 4.0
        elif re.search(r"\bverify\b", name_lower) and "numerical" in name_lower:
            score += 4.0

    if signals.prefer_instruments and item.product_kind == "instrument":
        score += 0.5
    if signals.prefer_reports and item.product_kind == "report":
        score += 1.0
    if signals.exclude_personality and "opq32r" in item.name.lower():
        score -= 10.0

    return score


def apply_rule_filter(
    candidates: list[CatalogItem],
    signals: QuerySignals,
    *,
    skills: list[str] | None = None,
    domains: list[str] | None = None,
    min_pool: int = 8,
    max_pool: int = 20,
) -> list[CatalogItem]:
    if not candidates:
        return []

    scored = [
        (idx, _tag_score(item, signals, skills, domains), item) for idx, item in enumerate(candidates)
    ]
    scored.sort(key=lambda row: (-row[1], row[0]))

    if signals.exclude_personality:
        scored = [
            row
            for row in scored
            if "opq32r" not in row[2].name.lower()
            and not (row[2].product_kind == "instrument" and "personality" in row[2].assessment_tags)
        ] or scored

    if domains and "leadership" in domains:
        scored = [
            row for row in scored
            if "entry level" not in row[2].name.lower()
        ] or scored

    if signals.wants_contact_centre:
        scored = [
            row
            for row in scored
            if not re.search(
                r"\b(cashier|hotel front desk|retail sales and service)\b",
                row[2].name.lower(),
            )
        ] or scored

    if signals.wants_finance:
        scored = [
            row
            for row in scored
            if not re.search(
                r"\b(cashier|sales solution|core java|oracle dba|hotel front desk|technical support)\b",
                row[2].name.lower(),
            )
        ] or scored

    filtered = [item for _, score, item in scored if score > -5.0]
    if len(filtered) < min_pool:
        filtered = [item for _, _, item in scored[:max_pool]]
    else:
        filtered = filtered[:max_pool]

    return filtered


def extract_refine_terms(user_text: str) -> tuple[list[str], list[str]]:
    text = user_text.lower()
    drop_terms: list[str] = []
    add_terms: list[str] = []

    for match in re.finditer(r"\b(drop|remove|skip|exclude)\s+(the\s+)?([a-z0-9\s\+\-]+)", text):
        term = match.group(3).strip()
        if term and len(term) >= 2:
            drop_terms.append(term)

    for match in re.finditer(r"\bwithout\s+(the\s+)?([a-z0-9\s\+\-]+)", text):
        term = match.group(2).strip()
        if term and len(term) >= 2:
            drop_terms.append(term)

    for pattern in ADD_PATTERNS:
        for match in re.finditer(pattern, text):
            for g in range(1, match.re.groups + 1):
                grp = match.group(g)
                if not grp:
                    continue
                for part in re.split(r"\s+(?:and|&)\s+|\s*,\s*", grp.strip()):
                    part = part.strip()
                    part = re.sub(r"^(a|an|the|some|any)\s+", "", part)
                    part = re.sub(r"\s+(?:as\s+well|also|too)$", "", part)
                    if part and len(part) >= 2:
                        add_terms.append(part)

    seen: set[str] = set()
    deduped: list[str] = []
    for term in add_terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)

    return drop_terms, deduped
