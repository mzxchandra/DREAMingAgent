"""Configuration for ALCF API endpoints and models."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class ALCFConfig(BaseSettings):
    """Configuration for ALCF inference gateway."""

    # ALCF Inference Gateway endpoints
    # Sophia cluster (vLLM): https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1
    # Metis cluster (API): https://inference-api.alcf.anl.gov/resource_server/metis/api/v1
    base_url: str = Field(
        default="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        description="ALCF inference gateway base URL"
    )

    # Available models on Sophia cluster
    llm_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        description="LLM model for text generation"
    )

    embedding_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3-embed",
        description="Embedding model for vector generation (ALCF)"
    )

    # Embedding provider: "alcf", "huggingface" (local), or "huggingface_api" (server-side)
    embedding_provider: str = Field(
        default="huggingface_api",
        description="Embedding provider to use (alcf, huggingface, or huggingface_api)"
    )

    # HuggingFace local embedding model (used when embedding_provider="huggingface")
    # Good options: sentence-transformers/all-mpnet-base-v2, BAAI/bge-small-en-v1.5, intfloat/e5-base-v2
    huggingface_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model for local sentence-transformers"
    )

    # HuggingFace Inference API settings (used when embedding_provider="huggingface_api")
    huggingface_api_token: str = Field(
        default=os.getenv("HUGGINGFACE_API_TOKEN", ""),
        description="HuggingFace API token (read access) for Inference API"
    )
    huggingface_api_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model for server-side Inference API"
    )
    huggingface_api_url: str = Field(
        default=" https://router.huggingface.co/pipeline/feature-extraction",
        description="HuggingFace Inference API base URL"
    )

    # Alternative models
    llm_model_alt_sophia: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # Sophia cluster
    llm_model_metis: str = "gpt-oss-120b-131072"  # Metis cluster
    base_url_metis: str = "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"

    # LLM parameters
    llm_temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    llm_max_tokens: int = Field(default=1000, description="Max tokens for LLM generation")

    # ChromaDB parameters
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="Directory for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="ecoli_literature",
        description="Collection name for literature documents"
    )

    # Retrieval parameters
    retrieval_top_k: int = Field(default=5, description="Number of documents to retrieve")
    retrieval_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval"
    )

    class Config:
        env_prefix = "RESEARCH_AGENT_"
        env_file = ".env"


# Global config instance
config = ALCFConfig()
