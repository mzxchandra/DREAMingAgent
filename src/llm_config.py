"""
LLM Configuration for DREAMing Agent

Provides initialization and configuration for Argonne-hosted LLM models
via the LangChain ChatOpenAI interface.
"""

from langchain_openai import ChatOpenAI
from src.utils.inference_auth_token import get_access_token


def create_argonne_llm(
    base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    model_name: str = "openai/gpt-oss-20b",
    temperature: float = 0.1,
    max_tokens: int = 8192
) -> ChatOpenAI:
    """
    Create ChatOpenAI instance configured for Argonne-hosted LLM.

    Args:
        base_url: Argonne inference API endpoint
        model_name: Model identifier (e.g., "openai/gpt-oss-20b")
        temperature: Sampling temperature (low for reproducibility)
        max_tokens: Maximum tokens in response

    Returns:
        Configured ChatOpenAI instance

    Why temperature=0.1?
        Scientific decisions need consistency across runs
        Low temperature reduces hallucination risk
        Reproducibility is critical for evaluation

    Why max_tokens=8192?
        Large TF regulons (50+ targets) need space for detailed decisions
        ~80 tokens per edge * 50 edges + overhead = ~4000-6000 tokens
    """
    api_key = get_access_token()

    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
