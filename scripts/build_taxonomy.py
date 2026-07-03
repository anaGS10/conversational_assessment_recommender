"""Build keywords.json and item_keywords.json from the SHL catalog."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.taxonomy import save_keyword_taxonomy_files  # noqa: E402


def main() -> None:
    print(f"Building keyword taxonomy from {settings.catalog_path} ...")
    data = save_keyword_taxonomy_files(settings.data_dir)
    tagged = sum(1 for kws in data.item_keywords.values() if kws)
    print(f"Saved {len(data.keywords)} keywords to {settings.data_dir / 'keywords.json'}")
    print(f"Tagged {tagged}/{len(data.item_keywords)} catalog items in item_keywords.json")


if __name__ == "__main__":
    main()
