#!/usr/bin/env python3
"""CLI script to ingest documents into the RAG pipeline.

Usage:
    python scripts/ingest.py --source path/to/documents
    python scripts/ingest.py --source path/to/file.pdf --reset
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    import json
    import urllib.error
    import urllib.request
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Path to a document file or directory of documents",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the vector store before ingestion",
    )
    args = parser.parse_args()

    # Try API ingestion first
    api_url = settings.api_url
    success = False

    try:
        health_url = f"{api_url.rstrip('/')}/healthz"
        # 0.5s timeout to check if API is running
        with urllib.request.urlopen(health_url, timeout=0.5) as response:
            if response.status == 200:
                health_data = json.loads(response.read().decode("utf-8"))
                if health_data.get("status") == "ok":
                    logging.info("Connected to API server at %s. Ingesting via API...", api_url)
                    ingest_url = f"{api_url.rstrip('/')}/ingest"
                    source_path = str(Path(args.source).resolve())
                    payload = {
                        "source": source_path,
                        "reset": args.reset,
                    }
                    req = urllib.request.Request(
                        ingest_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120.0) as i_resp:
                        if i_resp.status == 200:
                            res_data = json.loads(i_resp.read().decode("utf-8"))
                            print(
                                f"Ingested {res_data['chunks_ingested']} chunks. Total in store: {res_data['total_chunks']}"
                            )
                            success = True
    except Exception as e:
        logging.debug("API check failed or server offline (falling back to local): %s", e)

    if not success:
        logging.info(
            "API server offline or ingestion failed. Initialising local pipeline (this may take a few seconds)..."
        )
        pipeline = RAGPipeline()
        if args.reset:
            pipeline.reset()
            print("Vector store cleared.")
        count = pipeline.ingest(args.source)
        stats = pipeline.stats()
        print(f"Ingested {count} chunks. Total in store: {stats['chunks_in_store']}")


if __name__ == "__main__":
    main()
