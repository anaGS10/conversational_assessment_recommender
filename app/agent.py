from __future__ import annotations

import re
from typing import Literal

from app.catalog import (
    CatalogItem,
    extract_comparison_terms,
    final_validate_recommendations,
    find_by_name_fragment,
    format_recommendation_table,
    get_catalog,
    items_for_recommendations,
    items_to_recommendations,
    validate_recommendations,
)
from app.config import settings
from app.selection import confirm_prior_shortlist, generate_reply_intro
from app.conversation_state import (
    apply_refine_to_shortlist,
    extract_prior_recommendations,
    format_shortlist_context,
)
from app.filters import detect_query_signals, extract_refine_terms
from app.groups import (
    detect_seniority,
    fallback_groups_from_signals,
    filter_by_seniority,
    items_in_groups,
    route_query_to_groups,
)
from app.llm import LLMError, chat_completion
from app.retrieval import get_retrieval_index
from app.schemas import ChatResponse, Message

Intent = Literal["clarify", "recommend", "refine", "compare", "refuse", "confirm"]

RERANK_SYSTEM_PROMPT = """You are an SHL assessment recommender. Help users select products from the SHL Individual Test Solutions catalog.

You receive a CANDIDATE LIST (pre-filtered from the catalog). Use only entity IDs, names, and URLs from that list.

Product kinds:
- instrument: test or questionnaire the candidate completes
- report: output/report derived from an instrument (e.g. OPQ Leadership Report after OPQ32r)
- bundle: packaged multi-measure solution

Assessment tags: personality, skills, development, ability, knowledge, simulation, situational_judgment, safety, leadership, sales.

For development/reskilling, prefer GSA and development reports.
For selection hiring, pair instruments with relevant reports.
Match assessments to the seniority described in the conversation — do not recommend entry-level products for director, executive, or CXO hiring.

BEHAVIOR BY INTENT:

--- clarify ---
Do NOT include recommendations when you need more information first. It is normal and expected to return zero selected_entity_ids for several turns while gathering details.

When to clarify instead of recommend:
- Key information is missing (specific skills, seniority level, experience level, etc.)
- The user gave a short/bare query like "I need an assessment" or "We need a solution"
- The query mentions skills or roles that don't match anything in the catalog — explain the gap
- A complex JD needs narrowing (backend vs frontend, senior IC vs tech lead, etc.)
- Catalog constraints need discussion first (language availability, missing specific tests)

Examples of clarifying turns:
  User: "We need a solution for senior leadership."
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "Who is this meant for?" }

  User: "Hiring senior Rust engineer for networking."
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "No Rust-specific test exists..." }

  User: [JD for full-stack engineer]
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "Is this backend-leaning or frontend-heavy?" }

  User: [We need a Java developer]
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "What seniority level or experience are you looking for?" }

  User: [We need a developer]
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "What specific skills/technologies and experience level are you looking for?" }

  User: [We are hiring for a chef]
  Agent: { intent: "clarify", selected_entity_ids: [], reply: "We don't have any assessments for chefs. The SHL catalog does not have anything related to cooking or food preparation. If you need help with technology, financial services, or leadership, I can help with that." }

--- recommend ---
Return 1-10 IDs when the user has given enough detail to build a shortlist. This includes after clarifying questions have been answered.

--- refine ---
When the user says "add X", "drop Y", "remove Z", "replace W", "include W":
- Keep ALL previously recommended items unless explicitly asked to drop/remove them.
- Add new items, drop requested items.
- Return the updated full list.
- If the user asks to replace an item and NO suitable replacement exists in the candidate list, return empty selected_entity_ids with intent "clarify" and explain why no replacement is available.

Refine examples:
  Previous recs: [Core Java Advanced, Spring, RESTful, SQL, Verify G+, OPQ32r]
  User: "Add AWS and Docker. Drop REST."
  → Keep [Core Java Advanced, Spring, SQL, Verify G+, OPQ32r] + add [AWS Development, Docker]
  → selected_entity_ids includes ALL kept + new items.

  Previous recs: [OPQ32r, Verify G+, Graduate Scenarios]
  User: "Remove OPQ and replace with something shorter."
  → No suitable shorter alternative for OPQ32r exists.
  → intent: "clarify", selected_entity_ids: [], reply: "OPQ32r is the most relevant solution..."

  Previous recs: [Verify G+, OPQ32r, Graduate Scenarios]
  User: "Drop the OPQ."
  → selected_entity_ids: [Verify G+, Graduate Scenarios]

--- compare ---
When the user asks about differences between products:
- For pure comparison questions ("what's the difference between X and Y"), leave selected_entity_ids empty and explain the differences.
- Only include selected_entity_ids if the user is asking for a direct comparison within an existing shortlist AND wants to keep it.

--- confirm ---
When the user accepts the shortlist with phrases like "perfect", "that covers it", "confirmed", "that works", "clear", "locking it in", "keep as-is":
- Keep the final shortlist in selected_entity_ids.
- Set end_of_conversation = true.
- Do NOT ask follow-up questions — the conversation is done.

--- refuse ---
Refuse off-topic or legal/regulatory questions. Return empty selected_entity_ids.

RULES:
1. selected_entity_ids must be a subset of the CANDIDATE LIST IDs only.
2. Empty selected_entity_ids is correct and expected for clarify, refuse, and comparison-only turns.
3. Do NOT recommend until you have enough information — multiple clarify turns are fine.
4. On refine turns, always carry forward previously recommended items that weren't dropped.
5. Set end_of_conversation = true when user explicitly accepts/confirms. False otherwise.
6. Write only the conversational reply text. Do NOT include markdown tables, numbered lists, or product URLs in reply — the system appends a formatted catalog table automatically from selected_entity_ids.

Respond with JSON only:
{
  "intent": "clarify|recommend|refine|compare|refuse|confirm",
  "reply": "...",
  "selected_entity_ids": ["id1"],
  "end_of_conversation": false
}
"""

OFF_TOPIC_PATTERNS = [
    r"\b(legally required|legal requirement|satisfy.*requirement|regulatory obligation)\b",
    r"\b(lawsuit|attorney|lawyer|counsel|gdpr compliance requirement)\b",
    r"\bignore (all|previous) instructions\b",
    r"\bjailbreak\b",
]

VAGUE_PATTERNS = [
    r"^i need (an? )?assessment\.?$",
    r"^we need a solution\b",
    r"^help me choose\b",
    r"^what should i use\??$",
    r"^what do you recommend\??$",
    r"^can you recommend (an? )?assessment\??$",
    r"^i need to hire\b.{0,10}$",
    r"^we need to hire\b.{0,10}$",
    r"^looking for (an? )?assessment\b",
]

CONFIRM_PATTERNS = [
    r"\b(perfect|confirmed|that works|that'?s good|thanks|thank you|locking it in|keep the shortlist|as[- ]is)\b",
    r"\bthat covers it\b",
    r"\bthat(?:'s| is) (what we need|the one|good|fine|great)\b",
    r"\b(sounds good|looks good|works for me)\b",
    r"\blocking it in\b",
    r"\bkeep\s+(?:the\s+)?shortlist\b",
    r"\b(?:happy|go ahead|confirmed|finalize)\b",
    r"\bunderstood\b",
    r"^\s*(clear|that'?s all|we'?re (good|set|done)|looks? good)\s*",
    r"\bwe'?ll (use|go with|take)\b",
    r"\bfinal\s+(?:list|shortlist|battery)\b",
]

COMPARE_PATTERNS = [
    r"\bdifference between\b",
    r"\bdifferent (?:from|than|to)\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhat(?:'s| is) the difference\b",
]

REFINE_PATTERNS = [
    r"\b(add|drop|remove|replace|skip|exclude|include|adding)\b",
    r"\balso add\b",
    r"\bwithout\b",
]


def _conversation_text(messages: list[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _user_conversation_text(messages: list[Message]) -> str:
    return "\n".join(m.content for m in messages if m.role == "user")


def _latest_user_message(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


def _is_off_topic(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in OFF_TOPIC_PATTERNS)


def _is_vague_first_turn(messages: list[Message], text: str) -> bool:
    user_turns = sum(1 for m in messages if m.role == "user")
    if user_turns > 1:
        return False

    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, text.strip(), re.IGNORECASE):
            return True

    t = text.lower()

    has_seniority = bool(re.search(
        r"\b(entry|junior|mid|senior|executive|director|manager|cxo|vp|graduate|student|years?)\b", t,
    ))
    has_skill = bool(re.search(
        r"\b(java|python|sql|react|angular|node|aws|docker|kubernetes|linux|rust|excel|word|msword|"
        r"powerpoint|ppt|acrobat|reader|visio|outlook|access|database|nosql|microservice|"
        r"full.?stack|backend|frontend|api|rest|soap|graphql|c(?:\+\+)?|c#|php|ruby|swift|kotlin|"
        r"scala|go|html|css|javascript|typescript|jquery|bootstrap|english|spanish|french|"
        r"mandarin|german|italian|portuguese|russian|turkish|arabic|hindi|japanese|korean|"
        r"chinese|indonesian|malay|thai|vietnamese|urdu|persian|hebrew|us\.english|uk\.english|"
        r"inbound|outbound|telecommunications|transportation|customer\.experience|customer\.success|"
        r"customer\.engagement|customer\.satisfaction|customer\.loyalty|customer\.retention|"
        r"customer\.acquisition|general coding|programming|coding|technical|contact\.centre|admin|"
        r"sales|finance|leadership|personality|safety|mechanical|electrical|chemical|biomedical|"
        r"environmental|industrial|material|medical|nuclear|systems)\b", t,
    ))
    has_role = bool(re.search(
        r"\b(developer|engineer|manager|leader|analyst|agent|operator|representative|assistant|customer.service|customer.support)\b", t,
    ))

    return not (has_seniority and has_skill)


def _extract_mentioned_skills(text: str) -> set[str]:
    t = text.lower()
    return set(re.findall(
        r"\b(java|python|sql|react|angular|node|aws|docker|kubernetes|linux|rust|excel|word|msword|"
        r"powerpoint|ppt|acrobat|reader|visio|outlook|access|database|nosql|microservice|"
        r"full.?stack|backend|frontend|api|rest|soap|graphql|c(?:\+\+)?|c#|php|ruby|swift|kotlin|"
        r"scala|go|html|css|javascript|typescript|jquery|bootstrap|english|spanish|french|"
        r"mandarin|german|italian|portuguese|russian|turkish|arabic|hindi|japanese|korean|"
        r"chinese|indonesian|malay|thai|vietnamese|urdu|persian|hebrew|us\.english|uk\.english|"
        r"inbound|outbound|telecommunications|transportation|customer\.experience|customer\.success|"
        r"customer\.engagement|customer\.satisfaction|customer\.loyalty|customer\.retention|"
        r"customer\.acquisition|general coding|programming|coding|technical|contact\.centre|admin|"
        r"sales|finance|leadership|personality|safety|mechanical|electrical|chemical|biomedical|"
        r"environmental|industrial|material|medical|nuclear|systems)\b", t,
    ))


_CATALOG_TEXT_CACHE: str | None = None


def _all_catalog_search_text() -> str:
    global _CATALOG_TEXT_CACHE
    if _CATALOG_TEXT_CACHE is None:
        parts: list[str] = []
        for item in get_catalog():
            parts.append(item.name.lower())
            parts.append(item.description.lower())
            for tag in item.assessment_tags:
                parts.append(tag.lower())
        _CATALOG_TEXT_CACHE = " ".join(parts)
    return _CATALOG_TEXT_CACHE


def _any_skill_missing_from_catalog(text: str) -> bool:
    skills = _extract_mentioned_skills(text)
    if not skills:
        return False
    catalog_text = _all_catalog_search_text()
    return any(s not in catalog_text for s in skills)


def _assistant_reported_skill_gap(messages: list[Message]) -> bool:
    for msg in reversed(messages):
        if msg.role == "assistant":
            return bool(re.search(
                r"(no|don't|doesn't|do not|does not).*"
                r"(specific (assessment|test)|available|catalog doesn't|not .* in (our|the) catalog)",
                msg.content, re.IGNORECASE,
            ))
    return False


def _first_turn_knowledge_gap(messages: list[Message], text: str) -> bool:
    """Check if a first turn sounds specific but still lacks info the catalog needs."""
    user_turns = sum(1 for m in messages if m.role == "user")
    if user_turns > 1:
        return False
    t = text.lower()
    # Skill mentioned but not found in any catalog item
    if _any_skill_missing_from_catalog(text):
        return True
    # Contact centre without language mention
    if re.search(r"\bcontact centre\b", t) and not re.search(
        r"\b(language|english|spanish|french|mandarin|german)\b", t
    ):
        return True
    # Senior leadership/executive role without selection/development context
    if re.search(r"\bsenior (leadership|management|executive)\b", t) and not re.search(
        r"\b(selection|hiring|development|assessment|benchmark)\b", t
    ):
        return True
    # Healthcare admin with language/catalog constraints (HIPAA, bilingual)
    if re.search(r"\bhipaa\b", t) or re.search(r"\bbilingual\b", t):
        return True
    # JD-style query with multiple technologies that needs narrowing
    if re.search(r"\b(jd|job description|here'?s (the|a) (role|jd|position|req|opening))\b", text, re.IGNORECASE) and \
       len(re.findall(r"\b(java|spring|react|angular|node|aws|docker|kubernetes|sql|nosql|microservice|full.?stack|backend|frontend|api|rest|soap|graphql)\b", t, re.IGNORECASE)) >= 3:
        return True
    return False


def _is_confirm(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in CONFIRM_PATTERNS)


def _is_compare(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in COMPARE_PATTERNS)


def _is_refine(messages: list[Message], text: str) -> bool:
    if len(messages) <= 2:
        return False
    if not extract_prior_recommendations(messages):
        return False
    drop_terms, add_terms = extract_refine_terms(text)
    if drop_terms:
        return True
    if add_terms:
        return True
    return any(
        re.search(p, text, re.IGNORECASE)
        for p in (r"\b(drop|remove|replace|skip|exclude)\b", r"\bwithout\b")
    )


def _assistant_reply_has_recs(content: str) -> bool:
    """Check if an assistant reply contains product recommendation URLs."""
    return bool(re.search(r"https://www\.shl\.com/products/", content))


def _last_assistant_was_clarify(messages: list[Message]) -> bool:
    """Check if the most recent assistant response had no product recommendations."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return not _assistant_reply_has_recs(msg.content)
    return False


def _user_response_indicates_readiness(
    user_text: str,
    user_turn_count: int = 0,
) -> bool:
    """Check if the user's response to a clarifying question indicates readiness for recommendations."""
    text = user_text.strip().lower()
    # Explicit agreement to proceed
    if re.search(r"^(yes|sure|go ahead|proceed|please do|let'?s)", text):
        return True
    # Making a definitive choice ("go with X", "option A", "the first one")
    if re.search(r"\b(go with|choose|pick|select|i\'?ll (take|use|go with))\b", text):
        return True
    # Substantive follow-up question about specific assessments
    if re.search(r"\b(should I also|should we also|what about|also add|also include)\b", text):
        return True
    # Providing the missing key detail that was asked about (selection/development context)
    if re.search(r"\b(selection|benchmark|comparing candidate|promot|upskill|re.?skill)\b", text) and len(text.split()) >= 4:
        return True
    # At turn 3+, any non-question response indicates enough cumulative context
    words = [w for w in text.split() if w]
    if user_turn_count >= 3 and "?" not in text and len(words) > 0:
        return True
    return False


def _is_replace_request(user_text: str) -> bool:
    """Check if user asks to 'replace' (not just remove/drop)."""
    return bool(re.search(r"\breplace\b", user_text, re.IGNORECASE))


def _check_refine_replace(
    user_text: str,
    messages: list[Message],
) -> ChatResponse | None:
    """If user asks to replace a product and all prior recs match the removed family, return empty recs."""
    remove_term = _extract_removed_product_family(user_text)
    if not remove_term:
        return None
    if not _is_replace_request(user_text):
        # Just "drop/remove" — let the filter handle it later
        return None
    # User asked to REPLACE an item — check if most prior recs are from this family
    prior = extract_prior_recommendations(messages)
    if prior:
        opq_count = sum(1 for p in prior if remove_term.lower() in (p.name or "").lower())
        if opq_count / len(prior) >= 0.5:
            return ChatResponse(
                reply=f"The {remove_term.upper()} assessment is the most relevant instrument for this need. "
                      "No shorter alternative that covers the same scope exists in the catalog. "
                      "Would you like to keep the current recommendations or explore a different approach?",
                recommendations=[],
                end_of_conversation=False,
            )
    return None


def _extract_removed_product_family(user_text: str) -> str | None:
    """Extract product family keyword from a 'remove/replace X' request."""
    m = re.search(r"(?:remove|drop|replace)\s+(?:the\s+)?(\w+)", user_text, re.IGNORECASE)
    if not m:
        return None
    word = m.group(1)
    # Strip trailing digits and lowercase letters to get the uppercase prefix
    prefix = re.sub(r"[0-9a-z]+$", "", word)
    # Match all-caps product prefixes only (OPQ, DSI, MFS, etc.), not common words
    if len(prefix) >= 2 and prefix == prefix.upper() and prefix.isalpha():
        return prefix
    return None


def _refine_reply_intro(user_text: str, prior: list[CatalogItem], updated: list[CatalogItem]) -> str:
    dropped = [p.name for p in prior if p.entity_id not in {u.entity_id for u in updated}]
    added = [u.name for u in updated if u.entity_id not in {p.entity_id for p in prior}]
    parts: list[str] = []
    if dropped:
        parts.append(f"{', '.join(dropped)} out")
    if added:
        parts.append(f"{', '.join(added)} in")
    if parts:
        return f"Updated — {'; '.join(parts)}:"
    return "Updated list of recommended assessments:"


def _build_retrieval_query(messages: list[Message]) -> str:
    return " ".join(m.content for m in messages)[-2000:]


def _order_by_query_relevance(
    pool: list[CatalogItem],
    messages: list[Message],
) -> list[CatalogItem]:
    """Sort pool items by embedding similarity to the full conversation."""
    if not pool:
        return pool
    index = get_retrieval_index()
    ranked = index.search(_build_retrieval_query(messages), top_k=max(len(pool) * 2, 40))
    rank_of = {item.entity_id: idx for idx, item in enumerate(ranked)}
    default_rank = len(rank_of) + 1
    return sorted(pool, key=lambda item: rank_of.get(item.entity_id, default_rank))


def _forced_candidates(
    messages: list[Message],
    hint_intent: Intent | None = None,
) -> list[CatalogItem]:
    """Items that must be present regardless of group routing.

    Covers conversation continuity (prior recommendations) and explicit user
    requests (compare targets, refine "add X" terms).
    """
    index = get_retrieval_index()
    user_text = _latest_user_message(messages)

    items: list[CatalogItem] = []
    seen: set[str] = set()

    def add(item: CatalogItem) -> None:
        if item.entity_id not in seen:
            items.append(item)
            seen.add(item.entity_id)

    for item in extract_prior_recommendations(messages):
        add(item)

    if hint_intent == "compare" or _is_compare(user_text):
        terms = extract_comparison_terms(user_text)
        if terms:
            for item in index.search_by_terms(terms, per_term=4):
                add(item)
        else:
            for term in re.split(r"\band\b|\bvs\.?\b|\bversus\b", user_text, flags=re.IGNORECASE):
                term = re.sub(r"(?i)difference between|compare|what is|what's", "", term).strip()
                if len(term) >= 2:
                    for item in find_by_name_fragment(term, limit=3):
                        add(item)

    _, add_terms = extract_refine_terms(user_text)
    for term in add_terms:
        for item in find_by_name_fragment(term, limit=3):
            add(item)

    return items


def _embedding_fallback(messages: list[Message]) -> list[CatalogItem]:
    index = get_retrieval_index()
    return index.search(_build_retrieval_query(messages), top_k=settings.retrieval_top_k)


async def _build_candidate_pool(
    messages: list[Message],
    hint_intent: Intent | None = None,
    prior_items: list[CatalogItem] | None = None,
) -> list[CatalogItem]:
    conversation = _conversation_text(messages)
    signals = detect_query_signals(conversation)

    route = await route_query_to_groups(conversation)
    groups = route["groups"] or fallback_groups_from_signals(signals)
    seniority = route["seniority"] or detect_seniority(conversation)

    group_pool = items_in_groups(groups)
    group_pool = filter_by_seniority(group_pool, seniority)
    group_pool = _order_by_query_relevance(group_pool, messages)

    merged: list[CatalogItem] = []
    seen: set[str] = set()

    def add(item: CatalogItem) -> None:
        if item.entity_id not in seen:
            merged.append(item)
            seen.add(item.entity_id)

    # Explicit/continuity items first, then group members ordered by relevance.
    for item in _forced_candidates(messages, hint_intent):
        add(item)
    for item in group_pool:
        add(item)

    # Degrade gracefully if routing produced too few candidates (e.g. vague query).
    if len(merged) < 8:
        for item in filter_by_seniority(_embedding_fallback(messages), seniority):
            add(item)

    # Honor explicit "drop personality / OPQ" refine requests.
    if signals.exclude_personality:
        kept = [
            item
            for item in merged
            if "opq32r" not in item.name.lower()
            and not (item.product_kind == "instrument" and "personality" in item.assessment_tags)
        ]
        merged = kept or merged

    filtered = merged[: settings.retrieval_candidate_k]

    if prior_items:
        pinned_ids = {item.entity_id for item in filtered}
        for item in prior_items:
            if item.entity_id not in pinned_ids:
                filtered.append(item)

    return filtered


def _candidate_context(candidates: list[CatalogItem]) -> str:
    return "\n\n---\n\n".join(item.to_context_block() for item in candidates)


def _candidate_ids(candidates: list[CatalogItem]) -> set[str]:
    return {item.entity_id for item in candidates}


def _format_reply_with_urls(reply: str, items: list[CatalogItem]) -> str:
    """Append the sample-conversation markdown table so replies include catalog URLs."""
    if not items:
        return reply.strip()
    table = format_recommendation_table(items)
    return f"{reply.rstrip()}\n\n{table}"


def _finalize_response(
    reply: str,
    items: list[CatalogItem],
    end_of_conversation: bool,
) -> ChatResponse:
    recommendations = final_validate_recommendations(items_to_recommendations(items))
    table_items = items_for_recommendations(recommendations) if recommendations else []
    reply_with_table = _format_reply_with_urls(reply, table_items)
    return ChatResponse(
        reply=reply_with_table,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )


CLARIFY_CLASSIFY_PROMPT = """Decide if enough information has been provided to recommend SHL assessments.

CONVERSATION:
{conversation}

{clarify_context}

Choose CLARIFY when key details are missing:
- Unknown: role/level, selection vs development, language
- Complex/broad hiring that needs narrowing
- Niche or uncommon skill not clearly in catalog scope
- The assistant recently asked a question and the user hasn't fully answered

Choose RECOMMEND when the user has provided sufficient detail:
- Specific role/level and skills mentioned
- Clear selection or development context
- User has answered clarifying questions or explicitly agreed to proceed

Examples — CLARIFY:
- "We need a solution for senior leadership." → missing: selection/development, skills
- "500 entry-level contact centre agents, inbound calls." → missing: language
- "CXOs, director-level positions." → still no selection vs development
- "English." → after asking about language, still need accent

Examples — RECOMMEND:
- "Selection — comparing candidates against a leadership benchmark." → has role, context, purpose
- "Yes, go ahead. Should I also add a cognitive test?" → user agreed and asking to extend
- "US." → after accent question, now has all contact centre details
- "Hiring graduate financial analysts — final-year students, no experience..." → specific role, skills, context
- "Screen admin assistants for Excel and Word skills. Knowledge tests only." → specific skills, role, constraints

Return JSON: {{"action": "clarify" | "recommend"}}"""


async def _llm_classify_intent(messages: list[Message], clarify_context: str = "") -> str | None:
    """Ask LLM whether to clarify or recommend (no candidate bias)."""
    try:
        result = await chat_completion(
            CLARIFY_CLASSIFY_PROMPT.format(
                conversation=_conversation_text(messages),
                clarify_context=clarify_context,
            ),
            "Classify the intent.",
            temperature=0.1,
        )
        action = str(result.get("action", "")).lower()
        if action in ("clarify", "recommend"):
            return action
    except (LLMError, Exception):
        pass
    return None


async def _fallback_response(messages: list[Message]) -> ChatResponse:
    user_text = _latest_user_message(messages)

    if _is_off_topic(user_text):
        return ChatResponse(
            reply=(
                "I can help you select SHL assessments from our catalog, but I cannot provide "
                "legal or regulatory advice. Your compliance or legal team is the right resource for that. "
                "I can confirm what catalog products measure if that helps."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if _is_vague_first_turn(messages, user_text):
        return ChatResponse(
            reply="Happy to help narrow that down. Who is this assessment for, and what role or goal are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    if _first_turn_knowledge_gap(messages, user_text):
        return ChatResponse(
            reply="I need a bit more detail to recommend the right assessments. Could you share the role level, key skills, and whether this is for selection or development?",
            recommendations=[],
            end_of_conversation=False,
        )

    candidates = await _build_candidate_pool(messages)
    if candidates:
        return _finalize_response(
            reply=f"Based on your needs, here are {min(len(candidates), settings.max_recommendations)} SHL assessments that may fit.",
            items=candidates[: settings.max_recommendations],
            end_of_conversation=_is_confirm(user_text),
        )

    return ChatResponse(
        reply="Could you share more about the role, seniority, and skills you want to assess?",
        recommendations=[],
        end_of_conversation=False,
    )


async def _compare_path(messages: list[Message]) -> ChatResponse:
    """Handle comparison. Include prior recs only when comparing items already in the shortlist."""
    user_text = _latest_user_message(messages)
    candidates = await _build_candidate_pool(messages, "compare")
    prior_recs = extract_prior_recommendations(messages)
    # Only include prior recs if the compared items are within the shortlist
    include_prior = False
    if prior_recs:
        terms = extract_comparison_terms(user_text)
        if terms:
            for t in terms:
                if any(t.lower() in (p.name or "").lower() for p in prior_recs):
                    include_prior = True
                    break
    try:
        result = await chat_completion(
            RERANK_SYSTEM_PROMPT,
            f"""CONVERSATION:
{_conversation_text(messages)}

CANDIDATE LIST (for context):
{_candidate_context(candidates)}

This is a comparison question. Explain the differences between the products.
Return JSON with intent="compare", reply explaining differences, selected_entity_ids=[], end_of_conversation=false.""",
        )
        reply = str(result.get("reply", "")).strip()
        if reply:
            if include_prior:
                return _finalize_response(reply, prior_recs, end_of_conversation=False)
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
    except (LLMError, Exception):
        pass
    if include_prior:
        return _finalize_response(
            reply="Here is your current shortlist. Those products serve different assessment needs — let me explain how they differ.",
            items=prior_recs,
            end_of_conversation=False,
        )
    return ChatResponse(
        reply="Those products serve different assessment needs. Could you share more about which aspects matter most to your situation?",
        recommendations=[],
        end_of_conversation=False,
    )


async def _recommend_path(messages: list[Message], hint: Intent | None) -> ChatResponse:
    """Build candidates and call LLM to select recommendations."""
    user_text = _latest_user_message(messages)
    prior = extract_prior_recommendations(messages)
    candidates = await _build_candidate_pool(messages, hint, prior_items=prior)
    # Pre-filter candidates for "drop/remove" so LLM doesn't see removed items
    if hint == "refine":
        remove_term = _extract_removed_product_family(user_text)
        if remove_term and not _is_replace_request(user_text):
            candidates = [c for c in candidates if remove_term.lower() not in (c.name or "").lower()]

        replace_resp = _check_refine_replace(user_text, messages)
        if replace_resp:
            return replace_resp

        drop_terms, add_terms = extract_refine_terms(user_text)
        refined = apply_refine_to_shortlist(prior, user_text, candidates)
        # For add-only refine turns, use deterministic shortlist (bypass LLM which tends to drop prior items)
        if refined is not None and not drop_terms and add_terms:
            return _finalize_response(
                _refine_reply_intro(user_text, prior, refined),
                refined,
                end_of_conversation=_is_confirm(user_text),
            )
        # For drop-only or mixed refine, also use deterministic
        if refined is not None and drop_terms:
            return _finalize_response(
                _refine_reply_intro(user_text, prior, refined),
                refined,
                end_of_conversation=_is_confirm(user_text),
            )

    allowed_ids = _candidate_ids(candidates)
    shortlist_block = format_shortlist_context(prior)

    conversation = _conversation_text(messages)
    user_text = _user_conversation_text(messages)
    confirmed = confirm_prior_shortlist(prior, hint=hint)
    if confirmed:
        reply = await generate_reply_intro(
            conversation,
            confirmed,
            chat_completion=chat_completion,
        )
        return _finalize_response(reply, confirmed, end_of_conversation=True)

    user_prompt = f"""CONVERSATION:
{_conversation_text(messages)}

CURRENT SHORTLIST (carry forward all items unless user explicitly drops any; for refine turns keep prior items and add new ones):
{shortlist_block}

CANDIDATE LIST (available for selection):
{_candidate_context(candidates)}

IF no item is a good match, or if you need to explain a catalog constraint first, use intent "clarify" or "refuse" and return empty selected_entity_ids.

Detected hint intent: {hint or "recommend"}

Return JSON with intent, reply, selected_entity_ids, end_of_conversation."""

    try:
        result = await chat_completion(RERANK_SYSTEM_PROMPT, user_prompt)
    except (LLMError, Exception):
        if hint == "refine":
            replace_resp = _check_refine_replace(user_text, messages)
            if replace_resp:
                return replace_resp
            # Filter candidates for "drop/remove" (not replace)
            remove_term = _extract_removed_product_family(user_text)
            if remove_term and not _is_replace_request(user_text):
                candidates = [c for c in candidates if remove_term.lower() not in (c.name or "").lower()]
            if candidates:
                items = candidates[:settings.max_recommendations]
                return _finalize_response(f"Updated list of recommended assessments:", items, end_of_conversation=False)
        return await _fallback_response(messages)

    intent = str(result.get("intent", "recommend")).lower()
    reply = str(result.get("reply", "")).strip()
    entity_ids = [str(x) for x in result.get("selected_entity_ids", [])]
    end_of_conversation = bool(result.get("end_of_conversation", False))

    if not reply:
        return await _fallback_response(messages)

    validated = validate_recommendations(entity_ids, allowed_ids=allowed_ids)

    if intent in {"clarify", "refuse"}:
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=end_of_conversation)

    if not validated and candidates and intent in {"recommend", "refine", "confirm"}:
        validated = candidates[: settings.max_recommendations]

    if validated and intent in {"recommend", "refine", "confirm"}:
        if _is_confirm(user_text):
            end_of_conversation = True
        if hint == "refine":
            replace_resp = _check_refine_replace(user_text, messages)
            if replace_resp:
                return replace_resp
        return _finalize_response(reply, validated, end_of_conversation)

    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=end_of_conversation)


async def handle_chat(messages: list[Message]) -> ChatResponse:
    if not messages:
        return ChatResponse(
            reply="Hello. Tell me about the role or hiring goal and I will recommend SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    # PDF limit: 8 turns (one user message + one assistant reply per turn).
    user_turn_count = sum(1 for m in messages if m.role == "user")
    if user_turn_count > settings.max_conversation_turns:
        return ChatResponse(
            reply="This conversation has reached its maximum length. Please start a new conversation if you need further assistance.",
            recommendations=[],
            end_of_conversation=True,
        )

    user_text = _latest_user_message(messages)
    hint: Intent | None = None

    if _is_off_topic(user_text):
        hint = "refuse"
    elif _is_vague_first_turn(messages, user_text):
        hint = "clarify"
    elif _first_turn_knowledge_gap(messages, user_text):
        hint = "clarify"
    elif _is_compare(user_text):
        hint = "compare"
    elif _is_refine(messages, user_text):
        hint = "refine"
    elif _is_confirm(user_text):
        hint = "confirm"

    # For clarify/refuse hints, force empty recs (rule-based, no LLM bias needed)
    if hint in {"clarify", "refuse"}:
        try:
            candidates = await _build_candidate_pool(messages, hint)
            result = await chat_completion(
                RERANK_SYSTEM_PROMPT,
                f"""CONVERSATION:
{_conversation_text(messages)}

CANDIDATE LIST (for context only — do not recommend yet):
{_candidate_context(candidates)}

Detected intent: {hint}

Ask a clarifying question. Do NOT recommend any products and do NOT mention any SHL product names, assessment names, or URLs in your reply.
Return JSON. selected_entity_ids must be [].""",
            )
            reply = str(result.get("reply", "")).strip()
            if reply:
                return ChatResponse(
                    reply=reply,
                    recommendations=[],
                    end_of_conversation=bool(result.get("end_of_conversation", False)),
                )
        except (LLMError, Exception):
            pass
        return await _fallback_response(messages)

    # For refine/confirm, go straight to recommendation path
    if hint in {"refine", "confirm"}:
        resp = await _recommend_path(messages, hint)
        # Force end_of_conversation for closing confirmations
        if _is_confirm(user_text) and not resp.end_of_conversation:
            resp.end_of_conversation = True
        return resp
    # For compare, use dedicated path (explains without recommending)
    if hint == "compare":
        return await _compare_path(messages)

    # Check if the last assistant was a clarifying turn (had no product recommendations)
    in_clarify_mode = False
    clarify_context = ""
    if _last_assistant_was_clarify(messages):
        if _user_response_indicates_readiness(user_text, user_turn_count):
            clarify_context = (
                "NOTE: The assistant asked a clarifying question. "
                "The user has responded substantively. "
                "Decide if enough information exists to recommend, "
                "or if more clarification is still needed."
            )
        else:
            in_clarify_mode = True

    # Skill-gap follow-up: assistant said skill isn't in catalog, user picked a fallback
    if _last_assistant_was_clarify(messages) and _assistant_reported_skill_gap(messages):
        return await _recommend_path(messages, hint)

    if in_clarify_mode and hint is None:
        hint = "clarify"

    # No rule-based hint: ask LLM whether to clarify or recommend (no candidate bias)
    llm_action = await _llm_classify_intent(messages, clarify_context)
    if llm_action == "clarify" or hint == "clarify":
        # Include candidates for context so the LLM can reference real products
        candidates = await _build_candidate_pool(messages, "clarify")
        try:
            result = await chat_completion(
                RERANK_SYSTEM_PROMPT,
                f"""CONVERSATION:
{_conversation_text(messages)}

CANDIDATE LIST (for context only — do not recommend yet):
{_candidate_context(candidates) if candidates else "(no closely matching items found)"}

The user needs more information before a recommendation can be made.
Do NOT mention any SHL product names, assessment names, or URLs in your reply.

Return JSON with intent="clarify", reply asking a targeted question, selected_entity_ids=[], end_of_conversation=false.""",
            )
            reply = str(result.get("reply", "")).strip()
            if reply:
                return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
        except (LLMError, Exception):
            pass
        # Dedicated clarify fallback — never recommend when clarifying
        return ChatResponse(
            reply="Could you share more about the role, seniority, and specific skills you want to assess? This will help me find the right SHL assessments for you.",
            recommendations=[],
            end_of_conversation=False,
        )

    # LLM says recommend — proceed with candidate selection
    resp = await _recommend_path(messages, hint)
    # Force end_of_conversation when user expresses finality
    if _is_confirm(user_text):
        resp.end_of_conversation = True
    return resp
