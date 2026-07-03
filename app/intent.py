"""Extract structured hiring intent from user conversation via LLM."""

from __future__ import annotations

from app.llm import LLMError, chat_completion

INTENT_EXTRACT_PROMPT = """You extract structured hiring intent from conversation text.

Return JSON only:
{{
  "skills": ["skill1", "skill2"],
  "domains": ["domain1"],
  "clarification_needed": false
}}

Available domains:
- technical_hiring: hiring for technical/engineering/IT roles
- leadership: leadership, management, executive assessments
- personality: personality/behavior assessments
- development: development/reskilling/upskilling
- clerical: administrative/office/clerical roles
- sales: sales/customer-facing roles
- safety: safety-critical roles
- language: language proficiency testing
- contact_centre: contact center / customer service roles
- finance: finance/accounting roles

Rules:
- Extract specific skills (programming languages, tools, software, domain knowledge) from the query.
- Choose domains that best match the user's stated intent.
- If the user is too vague or no hiring intent is detectable, set clarification_needed: true and leave skills/domains empty.
- If skills are obvious (e.g. "Excel" → ["excel"]), include them even if the user didn't explicitly say "skill".

Examples:
  "Need Java developer with Spring Boot" → {{"skills": ["java", "spring boot"], "domains": ["technical_hiring"], "clarification_needed": false}}
  "Assess leadership capability for director role" → {{"skills": [], "domains": ["leadership"], "clarification_needed": false}}
  "Hiring admin assistant, needs Excel and Word" → {{"skills": ["excel", "word"], "domains": ["clerical"], "clarification_needed": false}}
  "I need help" → {{"skills": [], "domains": [], "clarification_needed": true}}
  "Re-skill our sales team" → {{"skills": [], "domains": ["development", "sales"], "clarification_needed": false}}
"""


async def extract_user_intent(conversation: str) -> dict:
    """Extract structured intent (skills, domains, clarification_needed) from conversation."""
    try:
        result = await chat_completion(
            INTENT_EXTRACT_PROMPT,
            f"CONVERSATION:\n{conversation}",
            temperature=0.0,
        )
        skills = [str(s).strip() for s in result.get("skills", []) if isinstance(s, str)]
        domains = [str(d).strip() for d in result.get("domains", []) if isinstance(d, str)]
        clarification_needed = bool(result.get("clarification_needed", False))
        return {
            "skills": skills,
            "domains": domains,
            "clarification_needed": clarification_needed,
        }
    except (LLMError, Exception):
        return {"skills": [], "domains": [], "clarification_needed": True}
