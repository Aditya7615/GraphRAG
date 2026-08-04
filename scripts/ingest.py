"""CLI ingestion: chunk, embed and index a folder or a single file.

    python scripts/ingest.py                          # ingests data/synthetic
    python scripts/ingest.py --dir data/synthetic
    python scripts/ingest.py --file report.pdf --type financial
    python scripts/ingest.py --reset                  # wipe index first
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.ingestion.pipeline import get_pipeline
from backend.ingestion.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--dir", help="Directory of documents to ingest.")
    parser.add_argument("--file", help="Single document to ingest.")
    parser.add_argument("--type", dest="doc_type", help="Override the document type tag.")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection and docstore first.")
    args = parser.parse_args()

    store = get_vector_store()
    if not store.ping():
        print(f"ERROR: Qdrant is not reachable at {settings.qdrant_url}.")
        print("Start it with: docker compose up -d qdrant")
        return 1

    pipeline = get_pipeline()
    if args.reset:
        print("Resetting collection and docstore...")
        pipeline.reset()

    if args.file:
        result = pipeline.ingest_file(args.file, doc_type=args.doc_type)
    else:
        target = Path(args.dir) if args.dir else settings.synthetic_data_dir
        if not target.is_dir():
            print(f"ERROR: {target} is not a directory. Run scripts/generate_synthetic_data.py first.")
            return 1
        result = pipeline.ingest_directory(target, doc_type=args.doc_type)

    print("\n" + "=" * 68)
    print(f"Documents ingested : {result.documents}")
    print(f"Parent chunks      : {result.parent_chunks}")
    print(f"Child chunks (vecs): {result.child_chunks}")
    print(f"Elapsed            : {result.latency_ms / 1000:.1f}s")
    print("=" * 68)
    for detail in result.details:
        print(
            f"  {detail['source']:<52} {detail['parent_chunks']:>3} parents / "
            f"{detail['child_chunks']:>4} children"
        )
    if result.skipped:
        print("\nSkipped:")
        for item in result.skipped:
            print(f"  - {item}")

    return 0 if result.documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
