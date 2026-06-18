#!/usr/bin/env python3
"""CLI script to query the RAG pipeline.

Usage:
    python scripts/query.py --question "What is RAG?"
    python scripts/query.py --question "How does hybrid search work?" --hybrid --reranker
    python scripts/query.py --question "..." --top-k 10 --no-citations
    python scripts/query.py --question "..." --provider anthropic
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

    from src.generation.citations import Citation

    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("--question", "-q", required=True, help="Your question")
    parser.add_argument(
        "--top-k", type=int, default=settings.top_k_final, help="Number of context chunks"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Enable hybrid search (BM25 + vector)"
    )
    parser.add_argument("--reranker", action="store_true", help="Enable cross-encoder re-ranker")
    parser.add_argument(
        "--no-citations", action="store_true", help="Suppress source citations in output"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider override (default: from RAG_LLM_PROVIDER env var)",
    )
    args = parser.parse_args()

    # Try API query first to avoid model loading cold-start
    api_url = settings.api_url
    answer = None
    citations = []

    try:
        health_url = f"{api_url.rstrip('/')}/healthz"
        # 0.5s timeout to quickly determine if server is running
        with urllib.request.urlopen(health_url, timeout=0.5) as response:
            if response.status == 200:
                health_data = json.loads(response.read().decode("utf-8"))
                if health_data.get("status") == "ok":
                    logging.info(
                        "Connected to API server at %s. Querying via API server (warm cache)...",
                        api_url,
                    )
                    query_url = f"{api_url.rstrip('/')}/query"
                    payload = {
                        "question": args.question,
                        "top_k": args.top_k,
                        "use_hybrid": args.hybrid,
                        "use_reranker": args.reranker,
                    }
                    req = urllib.request.Request(
                        query_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30.0) as q_resp:
                        if q_resp.status == 200:
                            res_data = json.loads(q_resp.read().decode("utf-8"))
                            answer = res_data["answer"]
                            citations = [
                                Citation(
                                    chunk_id=c["chunk_id"],
                                    source=c["source"],
                                    filename=c["filename"],
                                    text_snippet=c["text_snippet"],
                                    score=c["score"],
                                )
                                for c in res_data["citations"]
                            ]
    except Exception as e:
        logging.debug("API check failed or server offline (falling back to local): %s", e)

    if answer is None:
        logging.info(
            "API server offline or query failed. Initialising local pipeline (this may take a few seconds)..."
        )
        pipeline = RAGPipeline(llm_provider=args.provider)
        answer, citations = pipeline.query(
            args.question,
            top_k=args.top_k,
            use_hybrid=args.hybrid,
            use_reranker=args.reranker,
        )

    # Display answer
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    # Display citations
    if not args.no_citations and citations:
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
        for i, c in enumerate(citations, 1):
            score_info = f" (score: {c.score:.3f})" if c.score else ""
            print(f"\n[{i}] {c.filename}{score_info}")
            print(f"    Source: {c.source}")
            print(f"    Snippet: {c.text_snippet[:120]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
