"""Build data/catalog_groups.json — group the catalog by domain and assessment type.

Run once, and re-run only when shl_product_catalog.json changes:

    python scripts/build_groups.py

- Assessment-type groups come deterministically from the catalog 'keys' field.
- Domain groups are assigned by Gemini from each item's name/keys/description.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog import get_catalog  # noqa: E402
from app.config import settings  # noqa: E402
from app.groups import (  # noqa: E402
    ALL_GROUP_NAMES,
    DOMAIN_GROUP_NAMES,
    TYPE_GROUP_NAMES,
    classify_item_domains,
    deterministic_domain_groups,
    load_groups,
    type_groups_for_item,
)

BATCH_SIZE = 40
INTER_BATCH_DELAY_SECONDS = 5.0
MAX_RETRIES = 6


async def _classify_batch_with_retry(batch, batch_label: str) -> dict[str, list[str]]:
    delay = 8.0
    retryable = ("429", "500", "502", "503", "504", "timeout", "connect")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await classify_item_domains(batch)
        except Exception as exc:  # noqa: BLE001
            is_retryable = any(token in str(exc).lower() for token in retryable)
            if attempt == MAX_RETRIES or not is_retryable:
                print(f"  ! batch {batch_label} failed (attempt {attempt}): {exc}")
                return {}
            print(f"  … batch {batch_label} transient error, retrying in {delay:.0f}s (attempt {attempt})", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 90.0)
    return {}


async def _classify_all_domains(items) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    total = len(items)
    for start in range(0, total, BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        batch_label = f"{start}-{start + len(batch)}"
        batch_labels = await _classify_batch_with_retry(batch, batch_label)
        for item in batch:
            labels[item.entity_id] = batch_labels.get(item.entity_id, [])
        print(f"  classified {min(start + BATCH_SIZE, total)}/{total} items", flush=True)
        if start + BATCH_SIZE < total:
            await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)
    return labels


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build catalog_groups.json")
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq"],
        default=settings.llm_provider,
        help="LLM provider for the one-time domain classification (default: configured provider).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Assign domain groups from catalog metadata only (no LLM calls).",
    )
    args = parser.parse_args()
    # The build is a one-time offline job; use whichever provider is reliable.
    if not args.skip_llm:
        settings.llm_provider = args.provider

    catalog = list(get_catalog())
    classifier = "metadata" if args.skip_llm else args.provider
    print(f"Grouping {len(catalog)} catalog items (classifier: {classifier}) ...", flush=True)

    if args.skip_llm:
        domain_labels = {
            item.entity_id: deterministic_domain_groups(item) for item in catalog
        }
    else:
        print("Assigning domain groups ...", flush=True)
        domain_labels = await _classify_all_domains(catalog)
        for item in catalog:
            if not domain_labels.get(item.entity_id):
                domain_labels[item.entity_id] = deterministic_domain_groups(item)

    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in ALL_GROUP_NAMES}

    for item in catalog:
        entry = {"entity_id": item.entity_id, "name": item.name}
        for group in domain_labels.get(item.entity_id, []):
            grouped[group].append(entry)
        for group in type_groups_for_item(item):
            grouped[group].append(entry)

    for name in grouped:
        grouped[name].sort(key=lambda e: e["name"].lower())

    payload = {
        "groups": grouped,
        "_meta": {
            "domain_groups": list(DOMAIN_GROUP_NAMES),
            "type_groups": list(TYPE_GROUP_NAMES),
            "total_items": len(catalog),
        },
    }

    out_path = settings.data_dir / "catalog_groups.json"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    load_groups.cache_clear()

    print(f"\nSaved groups to {out_path}")
    for name in ALL_GROUP_NAMES:
        print(f"  {name}: {len(grouped[name])} items")
    unassigned = [
        item.name
        for item in catalog
        if not domain_labels.get(item.entity_id) and not type_groups_for_item(item)
    ]
    if unassigned:
        print(f"  ({len(unassigned)} items in no group)")


if __name__ == "__main__":
    asyncio.run(main())
