"""
Configuration for the DREAMing Agent System.

Controls LLM usage and other system-wide settings.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the agentic reconciliation system."""
    
    # LLM Settings
    use_llm: bool = True
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    
    # Reconciler Settings
    use_llm_reconciler: bool = True
    use_batch_mode: bool = True  # Batch LLM calls for efficiency
    
    # Context Agent Settings
    use_llm_context: bool = True
    
    # Thresholds
    z_score_high: float = 4.0
    z_score_moderate: float = 2.0
    z_score_low: float = 1.0
    
    # Processing
    batch_size: int = 5
    max_iterations: int = 100
    min_samples_for_analysis: int = 10
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables."""
        return cls(
            use_llm=os.environ.get("DREAMING_USE_LLM", "true").lower() == "true",
            llm_model=os.environ.get("DREAMING_LLM_MODEL", "gemini-2.5-flash"),
            use_llm_reconciler=os.environ.get("DREAMING_LLM_RECONCILER", "true").lower() == "true",
            use_llm_context=os.environ.get("DREAMING_LLM_CONTEXT", "true").lower() == "true",
        )


# Global config instance
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Get the global configuration."""
    global _config
    if _config is None:
        _config = AgentConfig.from_env()
    return _config


def set_config(config: AgentConfig):
    """Set the global configuration."""
    global _config
    _config = config
    
    # Update module-level flags
    from . import nodes
    nodes.reconciler.USE_LLM_REASONING = config.use_llm_reconciler
    nodes.reconciler.USE_BATCH_MODE = config.use_batch_mode
    nodes.context_agent.USE_LLM_CONTEXT = config.use_llm_context
    nodes.batch_manager.BATCH_SIZE = config.batch_size


def disable_llm():
    """Disable all LLM features (use rule-based fallbacks)."""
    config = get_config()
    config.use_llm = False
    config.use_llm_reconciler = False
    config.use_llm_context = False
    set_config(config)


def enable_llm():
    """Enable all LLM features."""
    config = get_config()
    config.use_llm = True
    config.use_llm_reconciler = True
    config.use_llm_context = True
    set_config(config)

