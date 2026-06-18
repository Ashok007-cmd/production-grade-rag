#!/usr/bin/env python3
"""Download and cache embedding and reranker models.

This script is run during Docker build to ensure that all models are pre-baked
into the image, preventing runtime failures or cold-start overheads in offline environments.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_EMBEDDING = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

MODELS_RERANKER = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-large",
]


def cache_models() -> None:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers package is required to cache models. "
            "Please run 'pip install sentence-transformers'."
        )
        sys.exit(1)

    # Cache embedding models
    for model_name in MODELS_EMBEDDING:
        logger.info("Downloading and caching embedding model: %s", model_name)
        try:
            _ = SentenceTransformer(model_name)
            logger.info("Successfully cached embedding model: %s", model_name)
        except Exception as exc:
            logger.exception("Failed to cache embedding model: %s", model_name)
            sys.exit(1)

    # Cache reranker models
    for model_name in MODELS_RERANKER:
        logger.info("Downloading and caching reranker model: %s", model_name)
        try:
            _ = CrossEncoder(model_name)
            logger.info("Successfully cached reranker model: %s", model_name)
        except Exception as exc:
            logger.exception("Failed to cache reranker model: %s", model_name)
            sys.exit(1)

    logger.info("All models cached successfully.")


if __name__ == "__main__":
    cache_models()
