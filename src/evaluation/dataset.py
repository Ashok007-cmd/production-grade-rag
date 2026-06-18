"""Golden evaluation dataset management."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A single evaluation example with question, reference answer, and expected sources."""

    question: str
    reference_answer: str
    expected_sources: list[str] = field(default_factory=list)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            import hashlib

            self.id = hashlib.md5(self.question.encode()).hexdigest()[:12]


class GoldenDataset:
    """Manages a curated "golden" evaluation dataset.

    Each entry contains a question, a reference answer, and expected source documents.
    The dataset is stored as JSONL for easy versioning and CI integration.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else Path("data/golden_dataset/dataset.jsonl")
        self._examples: list[EvalExample] = []

    # ------------------------------------------------------------------
    # Loading / Saving
    # ------------------------------------------------------------------

    def load(self) -> list[EvalExample]:
        """Load the golden dataset from the JSONL file."""
        if not self.path.exists():
            logger.warning("Golden dataset not found at %s", self.path)
            self._examples = []
            return self._examples

        self._examples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._examples.append(EvalExample(**data))

        logger.info("Loaded %d evaluation examples from %s", len(self._examples), self.path)
        return self._examples

    def save(self, examples: list[EvalExample] | None = None) -> None:
        """Save the dataset to a JSONL file."""
        if examples is not None:
            self._examples = examples

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for ex in self._examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

        logger.info("Saved %d examples to %s", len(self._examples), self.path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def examples(self) -> list[EvalExample]:
        """Return loaded examples (auto-loads if needed)."""
        if not self._examples:
            self.load()
        return self._examples

    def add(self, example: EvalExample) -> None:
        """Add a single example and persist."""
        self.examples.append(example)
        self.save()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EvalExample:
        return self.examples[index]

    # ------------------------------------------------------------------
    # Built-in datasets
    # ------------------------------------------------------------------

    @staticmethod
    def create_sample_dataset() -> list[EvalExample]:
        """Create a meticulously curated golden dataset for evaluation based on the sample documents."""
        return [
            EvalExample(
                question="What is Retrieval-Augmented Generation (RAG)?",
                reference_answer=(
                    "Retrieval-Augmented Generation (RAG) is a powerful AI framework that enhances "
                    "large language models (LLMs) by connecting them to external knowledge sources. "
                    "Instead of relying solely on the knowledge captured during training, RAG systems "
                    "dynamically retrieve relevant information from a knowledge base and inject it "
                    "into the LLM's context window at inference time."
                ),
                expected_sources=["rag_intro.txt"],
            ),
            EvalExample(
                question="Explain the three main stages of the RAG process.",
                reference_answer=(
                    "The RAG process consists of three main stages: "
                    "1. Ingestion: Documents are loaded, split into chunks, embedded into vector representations "
                    "using a sentence transformer, and stored in a vector database. "
                    "2. Retrieval: When a query is asked, it is embedded using the same model, and the vector "
                    "database is searched for the most semantically similar document chunks. "
                    "3. Generation: Retrieved chunks are formatted as context and passed to the LLM alongside the "
                    "question, and the LLM generates an answer based solely on this context, citing sources."
                ),
                expected_sources=["rag_intro.txt"],
            ),
            EvalExample(
                question="What are the key benefits of Retrieval-Augmented Generation?",
                reference_answer=(
                    "The key benefits of RAG are: "
                    "- Accuracy: Grounding answers in retrieved documents reduces hallucinations and improves factual accuracy. "
                    "- Freshness: Knowledge bases can be updated without retraining or fine-tuning the LLM. "
                    "- Transparency: Source citations allow users to verify claims and trace information back to original documents. "
                    "- Cost-Effectiveness: RAG avoids the expense of continual model retraining while keeping answers current."
                ),
                expected_sources=["rag_intro.txt"],
            ),
            EvalExample(
                question="What is BM25 Keyword Search and what are its strengths and weaknesses?",
                reference_answer=(
                    "BM25 is a classic information retrieval algorithm that ranks documents based on term frequency "
                    "and inverse document frequency. It excels at exact keyword matching, making it ideal for queries "
                    "containing specific technical terms, proper names, or identifiers. It is fast, interpretable, "
                    "and requires no training data. Its primary weakness is that it struggles with synonyms, "
                    "paraphrases, and semantic relationships (e.g. searching for 'automobile' won't match 'cars')."
                ),
                expected_sources=["hybrid_search.txt"],
            ),
            EvalExample(
                question="How does vector semantic search differ from BM25?",
                reference_answer=(
                    "Vector search uses neural embeddings to capture semantic meaning, mapping sentences with similar "
                    "meanings to nearby points in a high-dimensional vector space regardless of whether they share "
                    "exact keywords. This makes it robust to vocabulary mismatches and capable of understanding "
                    "nuanced queries. Unlike BM25, it can easily capture synonymy, but the trade-off is that it "
                    "can occasionally miss exact keyword matches that BM25 would retrieve easily."
                ),
                expected_sources=["hybrid_search.txt"],
            ),
            EvalExample(
                question="What is Reciprocal Rank Fusion (RRF) and what is the typical value of its constant parameter?",
                reference_answer=(
                    "Reciprocal Rank Fusion (RRF) is a method for combining multiple ranking lists (like BM25 and "
                    "vector search) into a single relevance score. The score is calculated using the formula: "
                    "Score(d) = sum( 1 / (k + rank_i(d)) ), where rank_i(d) is the rank position of document d in "
                    "ranking list i, and k is a constant typically set to 60 that controls how quickly rank "
                    "contributions diminish."
                ),
                expected_sources=["hybrid_search.txt"],
            ),
            EvalExample(
                question="What does the hybrid search alpha parameter control, and what is a common starting point value?",
                reference_answer=(
                    "The alpha parameter controls the balance between vector and keyword search contributions in "
                    "hybrid search. An alpha of 0.6 means 60% weight is placed on vector search and 40% on BM25 keyword "
                    "search. This value is a common starting point for many domains."
                ),
                expected_sources=["hybrid_search.txt"],
            ),
            EvalExample(
                question="How do bi-encoders and cross-encoders differ in a RAG pipeline?",
                reference_answer=(
                    "A bi-encoder independently encodes the query and each document into fixed-size vectors, and "
                    "similarity is computed via a dot product or cosine distance. This is highly efficient and suitable "
                    "for initial retrieval. A cross-encoder jointly encodes the query and document together through a "
                    "single transformer pass, allowing for deep attention-based interactions between every token of the "
                    "query and document. Cross-encoders are much more precise but significantly slower."
                ),
                expected_sources=["reranker.txt"],
            ),
            EvalExample(
                question="Why is a cross-encoder used as a re-ranking step rather than for initial retrieval?",
                reference_answer=(
                    "Cross-encoders are too computationally expensive and slow to use for initial retrieval over a "
                    "large corpus, because they require a separate forward pass for every query-document pair. Instead, "
                    "they are applied as a re-ranking step: a bi-encoder first retrieves a broad set of candidates "
                    "(e.g., top-20), and the cross-encoder scores only those candidate pairs (requiring fewer forward passes) "
                    "to produce a precise top-5."
                ),
                expected_sources=["reranker.txt"],
            ),
            EvalExample(
                question="Name three popular cross-encoder models for re-ranking in RAG.",
                reference_answer=(
                    "Three popular cross-encoder models are: "
                    "1. `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good accuracy, ~80MB). "
                    "2. `cross-encoder/ms-marco-MiniLM-L-12-v2` (higher accuracy, slower). "
                    "3. `BAAI/bge-reranker-v2-m3` (multilingual, strong performance)."
                ),
                expected_sources=["reranker.txt"],
            ),
            EvalExample(
                question="What is the most important metric for evaluating RAG system quality?",
                reference_answer=(
                    "Faithfulness (or groundedness) is the most important metric for RAG systems. It measures whether "
                    "the generated answer contains only claims that are directly supported by the retrieved context. "
                    "This is critical because unfaithful answers may look fluent and plausible while containing factually "
                    "incorrect hallucinations."
                ),
                expected_sources=["evaluation.txt"],
            ),
            EvalExample(
                question="Explain the four dimensions of RAG quality evaluation.",
                reference_answer=(
                    "The four dimensions of RAG quality are: "
                    "1. Faithfulness (Groundedness): Whether the answer is supported by the retrieved context. "
                    "2. Answer Relevance: Whether the answer addresses the user's question. "
                    "3. Context Precision: How much of the retrieved context is relevant. "
                    "4. Context Recall: Whether all relevant documents in the corpus are retrieved."
                ),
                expected_sources=["evaluation.txt"],
            ),
            EvalExample(
                question="How does LLM-as-judge evaluation work for faithfulness?",
                reference_answer=(
                    "In LLM-as-judge evaluation, an evaluator LLM is given the retrieved context and generated answer, "
                    "and is asked to identify any claims in the answer that are not supported by the context. This judge "
                    "produces a binary faithful/unfaithful judgment, a continuous score between 0.0 and 1.0, and a list "
                    "of unsupported claims. This automated method correlates well with human judgment when using capable models."
                ),
                expected_sources=["evaluation.txt"],
            ),
            EvalExample(
                question="How can RAG evaluation be integrated into Continuous Integration (CI) pipelines?",
                reference_answer=(
                    "RAG evaluation can be integrated into CI pipelines by running offline evaluations against a golden dataset. "
                    "If a code or configuration change causes the average faithfulness score to drop below a configured threshold "
                    "(e.g., 0.7), the CI build fails. This acts as a quality gate, preventing regressions from reaching production."
                ),
                expected_sources=["evaluation.txt"],
            ),
        ]
