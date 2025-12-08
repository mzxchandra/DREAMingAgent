"""
ALCF configuration for Research Agent.

Separate from src/config.py to avoid coupling with main system config.
Uses dataclass pattern to match src/config.py style.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ALCFConfig:
    """Configuration for ALCF inference gateway and embeddings."""

    # === LLM Configuration ===
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_BASE_URL",
            "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
        )
    )

    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_LLM_MODEL",
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    )

    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("RESEARCH_AGENT_LLM_TEMPERATURE", "0.7"))
    )

    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("RESEARCH_AGENT_LLM_MAX_TOKENS", "1000"))
    )

    # === Embedding Configuration ===
    embedding_provider: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_EMBEDDING_PROVIDER",
            "huggingface_api"
        )
    )

    # ALCF embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_EMBEDDING_MODEL",
            "mistralai/Mistral-7B-Instruct-v0.3-embed"
        )
    )

    # HuggingFace local embeddings
    huggingface_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_HUGGINGFACE_EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
    )

    # HuggingFace API embeddings
    huggingface_api_token: str = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_API_TOKEN", "")
    )

    huggingface_api_model: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_HUGGINGFACE_API_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
    )

    # === ChromaDB Configuration ===
    chroma_persist_directory: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_CHROMA_PERSIST_DIRECTORY",
            "./chroma_db"
        )
    )

    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv(
            "RESEARCH_AGENT_CHROMA_COLLECTION_NAME",
            "ecoli_literature"
        )
    )

    # === Retrieval Configuration ===
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RESEARCH_AGENT_RETRIEVAL_TOP_K", "5"))
    )

    retrieval_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv(
            "RESEARCH_AGENT_RETRIEVAL_SIMILARITY_THRESHOLD",
            "0.7"
        ))
    )


# Global singleton
_alcf_config: Optional[ALCFConfig] = None


def get_alcf_config() -> ALCFConfig:
    """Get or create global ALCF config instance."""
    global _alcf_config
    if _alcf_config is None:
        _alcf_config = ALCFConfig()
    return _alcf_config


def set_alcf_config(config: ALCFConfig):
    """Override global ALCF config instance."""
    global _alcf_config
    _alcf_config = config
