"""
LLM Client Configuration

Handles Google Gemini API initialization and response generation.
"""

import os
from typing import Optional
from loguru import logger

# Global client instance
_client = None


def get_llm_client():
    """
    Get or create the Google Genai client.
    
    Requires GEMINI_API_KEY environment variable to be set.
    
    Returns:
        Google Genai Client instance
    """
    global _client
    
    if _client is None:
        try:
            from google import genai
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning(
                    "GEMINI_API_KEY not set. LLM features will be disabled. "
                    "Set the environment variable to enable AI reasoning."
                )
                return None
            
            _client = genai.Client(api_key=api_key)
            logger.info("Google Genai client initialized successfully")
            
        except ImportError:
            logger.warning(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Genai client: {e}")
            return None
    
    return _client


def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> Optional[str]:
    """
    Generate a response from the LLM.
    
    Args:
        prompt: The user prompt / question
        system_prompt: Optional system instructions
        model: Model name (default: gemini-2.5-flash)
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum response length
        
    Returns:
        Generated text response, or None if LLM unavailable
    """
    client = get_llm_client()
    
    if client is None:
        logger.debug("LLM client not available, returning None")
        return None
    
    try:
        # Combine system prompt with user prompt if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        else:
            full_prompt = prompt
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None


def generate_structured_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gemini-2.5-flash"
) -> Optional[dict]:
    """
    Generate a JSON-structured response from the LLM.
    
    Args:
        prompt: The prompt requesting structured output
        system_prompt: System instructions
        model: Model name
        
    Returns:
        Parsed JSON dict, or None if failed
    """
    import json
    
    response = generate_response(
        prompt=prompt + "\n\nRespond with valid JSON only.",
        system_prompt=system_prompt,
        model=model,
        temperature=0.1  # Lower temp for structured output
    )
    
    if response is None:
        return None
    
    try:
        # Try to extract JSON from response
        # Handle cases where model wraps JSON in markdown code blocks
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM JSON response: {e}")
        logger.debug(f"Raw response: {response}")
        return None

