from __future__ import annotations

import json
import re

import numpy as np

from app.catalog import CatalogItem, find_by_name_fragment, get_catalog
from app.config import settings

_EMBEDDINGS: np.ndarray | None = None
_ENTITY_IDS: list[str] | None = None
_MODEL = None


_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "but", "not", "no", "from", "this", "that", "these", "those",
    "it", "its", "all", "each", "any", "also", "than", "then", "per", "via", "can",
    "will", "would", "could", "should", "may", "might", "shall", "about", "into",
    "over", "after", "before", "between", "under", "above", "below", "out", "off",
    "up", "down", "just", "very", "too", "so", "some", "such", "more", "most",
    "other", "what", "which", "who", "whom", "when", "where", "why", "how",
    "am", "he", "she", "they", "we", "you", "me", "him", "her", "us", "them",
})


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


def _url_slug_tokens(item: CatalogItem) -> set[str]:
    slug = item.url.rstrip("/").rsplit("/", 1)[-1] if "/" in item.url else ""
    return _tokenize(slug.replace("-", " "))


def _keyword_score(query: str, item: CatalogItem) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    name_tokens = _tokenize(item.name)
    desc_tokens = _tokenize(item.description[:500])
    field_tokens = _tokenize(" ".join(item.keys + item.job_levels + item.languages))
    slug_tokens = _url_slug_tokens(item)

    name_overlap = len(query_tokens & name_tokens) / len(query_tokens)
    desc_overlap = len(query_tokens & desc_tokens) / len(query_tokens)
    field_overlap = len(query_tokens & field_tokens) / len(query_tokens)
    slug_overlap = len(query_tokens & slug_tokens) / len(query_tokens) if slug_tokens else 0.0

    score = name_overlap * 3.0 + desc_overlap * 1.0 + field_overlap * 1.5 + slug_overlap * 1.5

    query_lower = query.lower()
    name_lower = item.name.lower()
    if query_lower in name_lower or name_lower in query_lower:
        score += 2.0
    name_clean = re.sub(r"\s*\([^)]*\)", "", name_lower).strip()
    if name_clean and (query_lower in name_clean or name_clean in query_lower):
        score += 2.0

    slug_lower = item.url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").lower() if "/" in item.url else ""
    if slug_lower and (query_lower in slug_lower or slug_lower in query_lower):
        score += 1.5

    for token in query_tokens:
        if len(token) >= 4 and token in name_lower:
            score += 1.0

    return score


class RetrievalIndex:
    def __init__(self) -> None:
        self.items: list[CatalogItem] = list(get_catalog())
        self.entity_id_to_idx = {item.entity_id: i for i, item in enumerate(self.items)}
        self.embeddings: np.ndarray | None = None

    def ensure_loaded(self) -> None:
        global _MODEL, _EMBEDDINGS, _ENTITY_IDS

        if self.embeddings is not None:
            return

        embeddings_path = settings.data_dir / "embeddings.npy"
        entity_ids_path = settings.data_dir / "entity_ids.json"

        if embeddings_path.exists() and entity_ids_path.exists():
            self.embeddings = np.load(embeddings_path)
            with entity_ids_path.open(encoding="utf-8") as f:
                stored_ids = json.load(f)
            if len(stored_ids) == len(self.items):
                id_order = [self.entity_id_to_idx[eid] for eid in stored_ids if eid in self.entity_id_to_idx]
                if len(id_order) == len(self.items):
                    self.embeddings = self.embeddings[id_order] if len(id_order) else self.embeddings
                    return

        self._build_embeddings()

    def _build_embeddings(self) -> None:
        global _MODEL
        from sentence_transformers import SentenceTransformer

        if _MODEL is None:
            _MODEL = SentenceTransformer(settings.embedding_model)

        texts = [item.to_search_text() for item in self.items]
        self.embeddings = _MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(settings.data_dir / "embeddings.npy", self.embeddings)
        with (settings.data_dir / "entity_ids.json").open("w", encoding="utf-8") as f:
            json.dump([item.entity_id for item in self.items], f)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        exclude_ids: set[str] | None = None,
        boost_ids: set[str] | None = None,
    ) -> list[CatalogItem]:
        self.ensure_loaded()
        k = top_k or settings.retrieval_top_k
        exclude_ids = exclude_ids or set()
        boost_ids = boost_ids or set()

        if self.embeddings is None:
            return self.items[:k]

        from sentence_transformers import SentenceTransformer

        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer(settings.embedding_model)

        query_vec = _MODEL.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        semantic = self.embeddings @ query_vec

        keyword = np.array([_keyword_score(query, item) for item in self.items], dtype=np.float32)
        if keyword.max() > 0:
            keyword = keyword / keyword.max()

        combined = semantic * 0.65 + keyword * 0.35

        for idx, item in enumerate(self.items):
            if item.entity_id in boost_ids:
                combined[idx] += 0.25
            if item.entity_id in exclude_ids:
                combined[idx] = -1.0

        top_indices = np.argsort(combined)[::-1]
        results: list[CatalogItem] = []
        for idx in top_indices:
            item = self.items[int(idx)]
            if item.entity_id in exclude_ids:
                continue
            results.append(item)
            if len(results) >= k:
                break
        return results

    def search_by_terms(self, terms: list[str], per_term: int = 3) -> list[CatalogItem]:
        seen: set[str] = set()
        results: list[CatalogItem] = []
        for term in terms:
            for item in find_by_name_fragment(term, limit=per_term):
                if item.entity_id not in seen:
                    results.append(item)
                    seen.add(item.entity_id)
            for item in self.search(term, top_k=per_term):
                if item.entity_id not in seen:
                    results.append(item)
                    seen.add(item.entity_id)
        return results


_index: RetrievalIndex | None = None


def get_retrieval_index() -> RetrievalIndex:
    global _index
    if _index is None:
        _index = RetrievalIndex()
    return _index

