# Approach Document — SHL Assessment Recommender

## Design Choices

**Architecture — Retriever → Intent Extraction → Deterministic Scoring → LLM (explanation only).**  
A pipeline that grounds every recommendation in the catalog before the LLM sees the candidate pool. The retriever produces candidates, Gemini extracts structured hiring intent (skills + domains), a deterministic scoring function ranks items against those skills/domains, and the LLM only writes the reply explanation — it does not select items.

**Stateless API.**  
Each `POST /chat` carries full conversation history. No server-side session state. This simplifies deployment (any number of replicas) and matches the evaluator's replay harness.

**Action-based LLM schema.**  
The LLM outputs `{"intent", "reply", "selected_entity_ids", "end_of_conversation"}` rather than free text. Entity IDs are resolved against the candidate pool rather than names, reducing surface area for hallucination. Intent labels (`clarify`, `recommend`, `refine`, `compare`, `refuse`, `confirm`) let the agent's routing logic handle each case differently.

**Why not LangChain / LlamaIndex?**  
The pipeline is simple enough that a framework adds dependency risk without benefit. Raw FastAPI + httpx + sentence-transformers gives full control over every step and keeps cold-start fast on free-tier hosts.

## Retrieval Setup

**Hybrid search: embedding similarity + keyword scoring.**  
- `all-MiniLM-L6-v2` (384-dim) produces normalized embeddings for all 377 catalog items at startup.
- Semantic weight 0.65, keyword weight 0.35.
- Keyword scoring uses token overlap on name (×3), description (×1), and metadata fields (×1.5).
- Pre-computed embeddings cached to `data/embeddings.npy` for fast cold starts.

**Intent extraction** (`app/intent.py`) replaces the old keyword-taxonomy approach. Instead of mapping the query to a flat keyword list and doing set-overlap filtering, Gemini extracts structured intent:
- **skills**: specific abilities, tools, or domain knowledge ("java", "excel", "financial accounting")
- **domains**: high-level use cases ("technical_hiring", "leadership", "clerical", "development")
- **clarification_needed**: whether the query is too vague to act on

**Deterministic scoring** (`app/filters.py:_tag_score`) ranks catalog items against the extracted intent:
- Skill match in name/description → +5.0 per skill
- Skill match in keys/metadata → +4.0 per skill
- Domain match against assessment_tags → +2.0 to +3.0 per domain
- Opposite-level penalty: leadership domain penalizes "entry level" items (−4.0)

This replaces the earlier pipeline where the LLM mapped query→keywords and a set-overlap filter matched keywords→items. The old approach suffered from noisy generic keywords ("what", "level", "solution") and error propagation from the LLM to the downstream filter. The new approach uses a single LLM call for semantic understanding and deterministic scoring for item matching.

**Rule filter** (`filters.py`) applies additional signal-based boosts and penalties after intent scoring:
- Development queries → boost GSA/skills tags, penalize OPQ/personality.
- Hiring/selection queries → boost instruments.
- Explicit "drop OPQ"/"exclude personality" → deep penalty (-10).
- Extract refine terms ("add X", "drop Y") and blend into the candidate pool.

For comparison queries, terms are extracted via regex ("difference between X and Y", "compare X vs Y") and each term is searched independently with deduplication.

## Prompt Design

**Intent extraction prompt** (`INTENT_EXTRACT_PROMPT`) instructs Gemini to:
- Extract specific skills (programming languages, tools, software) from the query.
- Classify the use case into available domains.
- Set `clarification_needed: true` if the query is too vague.

**Reply-only prompt** (`REPLY_ONLY_PROMPT` in `selection.py`) instructs the LLM to:
- Write a 2-4 sentence conversational intro.
- NOT include markdown tables, numbered lists, or URLs.
- The system appends the product table separately.

Both Groq and Gemini are configured with `response_format: {"type": "json_object"}` to guarantee parseable output.

## Evaluation Approach

**Local replay harness** (`scripts/replay_conversations.py`) replays the 10 provided conversation traces against the agent turn by turn and computes Mean Recall@10. This is used for iterative development — the actual scoring is done by SHL's automated evaluator on holdout traces.

**Smoke tests** (`scripts/smoke_test.py`) validate:
- Schema compliance (response serialization).
- Catalog metadata accuracy (OPQ32r → instrument + personality).
- Final validation gate (fake names are stripped).
- Domain-specific filter behavior (development query → no OPQ in top result).
- Retrieval relevance (Java query → Java assessments).
- Fallback agent behavior (vague → clarification, legal → refusal).

## What Didn't Work

1. **Catalog keyword taxonomy.** The earlier approach extracted a flat keyword list from catalog items (791 keywords), used the LLM to map queries to keywords, and filtered items by set overlap. This produced generic keywords ("what", "level", "solution", "experience") that matched too many items and diluted precision. The taxonomy also required maintaining `keywords.json` and `item_keywords.json` files that drifted from the catalog. Replaced by intent extraction + deterministic scoring.

2. **Pure embedding search.** Top-10 by cosine similarity alone returned many irrelevant items for domain-specific queries (e.g., "re-skill sales team" returned ability tests, not development reports). Adding keyword scoring and rule-based filtering fixed this.

3. **LLM-as-retriever.** Early versions sent the full catalog in the prompt and asked the LLM to pick. Context windows were exhausted and the LLM hallucinated product names. Splitting into retriever → deterministic scoring eliminated hallucination and kept prompts small.

4. **Free-form LLM responses.** When the LLM wrote recommendation text freely, URLs and product descriptions drifted from the catalog. Switching to an entity-ID-based action schema and cross-referencing against the candidate pool solved this.

## AI Tools Used

- **LLM (Gemini 2.0 Flash):** Powers intent extraction and reply explanation. Not used for item selection.
- **SentenceTransformers:** Embedding model for semantic retrieval.
- **FAISS (implicit, via numpy dot product):** Similarity search over 377 embeddings — simple enough that explicit FAISS indexing wasn't needed.
- **FastAPI + Uvicorn:** API framework.

03-07-2026
Nothing is working. All approaches are returning irrelevant items. The whole keyword, comparison approach is failing. May be since there are a lot of assessment items.

Removed "keyword/taxonomy generation from the catalog items" approach since it was becoming too complex to compare hundreds of keywords with the user queries and the LLM's comparison results were hallucinated.

New approach, extracted domain, skills, experience from the user's query using gemini and compared it with the catalog items using scoring. It significantly improved the result. But the results were still irrelevant for a few conversations(C1.md, C2.md...).

Grouping the catalog data by domain, skill so it becomes easier and faster to compare.
