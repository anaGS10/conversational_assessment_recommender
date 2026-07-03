"""Precompute embedding index for the SHL catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.retrieval import RetrievalIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHL catalog embedding index")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=settings.catalog_path,
        help="Path to catalog JSON",
    )
    args = parser.parse_args()

    if args.catalog != settings.catalog_path:
        settings.catalog_path = args.catalog

    print(f"Building index from {settings.catalog_path} ...")
    index = RetrievalIndex()
    index._build_embeddings()
    print(f"Saved embeddings to {settings.data_dir / 'embeddings.npy'}")
    print(f"Indexed {len(index.items)} products.")


if __name__ == "__main__":
    main()
