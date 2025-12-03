"""Unified ALCF client for LLM and embedding operations."""

from typing import List, Optional, Dict, Any
from openai import OpenAI
import requests

from .auth import get_alcf_token
from .config import config

# Lazy import for sentence-transformers (only if using HuggingFace local)
_sentence_transformer_model = None


def _get_sentence_transformer(model_name: str):
    """Lazy load sentence transformer model."""
    global _sentence_transformer_model

    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    return _sentence_transformer_model


class ALCFClient:
    """
    Unified client for ALCF inference gateway.

    Handles chat completions via ALCF and embeddings via either ALCF or HuggingFace.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        huggingface_embedding_model: Optional[str] = None,
        huggingface_api_token: Optional[str] = None,
        huggingface_api_model: Optional[str] = None,
        huggingface_api_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize ALCF client.

        Args:
            base_url: ALCF API base URL (defaults to config.base_url)
            llm_model: LLM model name (defaults to config.llm_model)
            embedding_model: ALCF embedding model name (defaults to config.embedding_model)
            embedding_provider: "alcf", "huggingface", or "huggingface_api" (defaults to config.embedding_provider)
            huggingface_embedding_model: HuggingFace local model name (defaults to config.huggingface_embedding_model)
            huggingface_api_token: HuggingFace API token (defaults to config.huggingface_api_token)
            huggingface_api_model: HuggingFace API model name (defaults to config.huggingface_api_model)
            huggingface_api_url: HuggingFace API URL (defaults to config.huggingface_api_url)
            temperature: Sampling temperature (defaults to config.llm_temperature)
            max_tokens: Maximum tokens to generate (defaults to config.llm_max_tokens)
        """
        self.base_url = base_url or config.base_url
        self.llm_model = llm_model or config.llm_model
        self.embedding_model = embedding_model or config.embedding_model
        self.embedding_provider = embedding_provider or config.embedding_provider
        self.huggingface_embedding_model = huggingface_embedding_model or config.huggingface_embedding_model
        self.huggingface_api_token = huggingface_api_token or config.huggingface_api_token
        self.huggingface_api_model = huggingface_api_model or config.huggingface_api_model
        self.huggingface_api_url = huggingface_api_url or config.huggingface_api_url
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else config.llm_max_tokens

    def _get_client(self) -> OpenAI:
        """Get OpenAI client with fresh ALCF token."""
        token = get_alcf_token()
        return OpenAI(
            api_key=token,
            base_url=self.base_url
        )

    # ========== LLM Methods ==========

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides instance default)
            max_tokens: Maximum tokens to generate (overrides instance default)
            model: LLM model to use (overrides instance default)
            **kwargs: Additional arguments to pass to OpenAI API

        Returns:
            Generated text content
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self.llm_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def chat_with_system(
        self,
        user_message: str,
        system_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion with system message.

        Args:
            user_message: User message content
            system_message: System message content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Generated text content
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return self.chat(messages, temperature, max_tokens, **kwargs)

    # ========== Embedding Methods ==========

    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Embedding model to use (overrides instance default)

        Returns:
            List of floats representing the embedding vector
        """
        return self.embed_batch([text], model=model)[0]

    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use (overrides instance default)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Use HuggingFace local embeddings if configured
        if self.embedding_provider == "huggingface":
            model_name = model or self.huggingface_embedding_model
            sentence_model = _get_sentence_transformer(model_name)
            # sentence-transformers returns numpy arrays, convert to lists
            embeddings = sentence_model.encode(texts, convert_to_numpy=True)
            return [embedding.tolist() for embedding in embeddings]

        # Use HuggingFace Inference API
        if self.embedding_provider == "huggingface_api":
            model_name = model or self.huggingface_api_model

            if not self.huggingface_api_token:
                raise ValueError(
                    "HuggingFace API token is required for huggingface_api provider. "
                    "Set HUGGINGFACE_API_TOKEN environment variable or "
                    "RESEARCH_AGENT_HUGGINGFACE_API_TOKEN."
                )

            headers = {"Authorization": f"Bearer {self.huggingface_api_token}"}

            # HuggingFace Inference API URL (new router endpoint)
            api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

            # HuggingFace Inference API payload
            payload = {"inputs": texts}

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"HuggingFace API request failed with status {response.status_code}: "
                    f"{response.text}\n"
                    f"URL: {api_url}\n"
                    f"Model: {model_name}"
                )

            # API returns embeddings directly as nested lists
            embeddings = response.json()

            # Handle both single and batch responses
            if embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]

            return embeddings

        # Otherwise use ALCF embeddings
        client = self._get_client()
        response = client.embeddings.create(
            model=model or self.embedding_model,
            input=texts,
            encoding_format="float"  # Required for ALCF
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings


# Global client instance
_alcf_client: Optional[ALCFClient] = None


def get_alcf_client(
    base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    huggingface_embedding_model: Optional[str] = None,
    huggingface_api_token: Optional[str] = None,
    huggingface_api_model: Optional[str] = None,
    huggingface_api_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> ALCFClient:
    """
    Get or create global ALCF client instance.

    Args:
        base_url: ALCF API base URL
        llm_model: LLM model name
        embedding_model: ALCF embedding model name
        embedding_provider: "alcf", "huggingface", or "huggingface_api"
        huggingface_embedding_model: HuggingFace local embedding model name
        huggingface_api_token: HuggingFace API token
        huggingface_api_model: HuggingFace API model name
        huggingface_api_url: HuggingFace API URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        ALCFClient instance
    """
    global _alcf_client

    # Create new client if none exists or if parameters differ
    if _alcf_client is None:
        _alcf_client = ALCFClient(
            base_url=base_url,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            huggingface_embedding_model=huggingface_embedding_model,
            huggingface_api_token=huggingface_api_token,
            huggingface_api_model=huggingface_api_model,
            huggingface_api_url=huggingface_api_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif (
        (base_url is not None and base_url != _alcf_client.base_url) or
        (llm_model is not None and llm_model != _alcf_client.llm_model) or
        (embedding_model is not None and embedding_model != _alcf_client.embedding_model) or
        (embedding_provider is not None and embedding_provider != _alcf_client.embedding_provider) or
        (huggingface_embedding_model is not None and huggingface_embedding_model != _alcf_client.huggingface_embedding_model) or
        (huggingface_api_token is not None and huggingface_api_token != _alcf_client.huggingface_api_token) or
        (huggingface_api_model is not None and huggingface_api_model != _alcf_client.huggingface_api_model) or
        (huggingface_api_url is not None and huggingface_api_url != _alcf_client.huggingface_api_url) or
        (temperature is not None and temperature != _alcf_client.temperature) or
        (max_tokens is not None and max_tokens != _alcf_client.max_tokens)
    ):
        _alcf_client = ALCFClient(
            base_url=base_url,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            huggingface_embedding_model=huggingface_embedding_model,
            huggingface_api_token=huggingface_api_token,
            huggingface_api_model=huggingface_api_model,
            huggingface_api_url=huggingface_api_url,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return _alcf_client
