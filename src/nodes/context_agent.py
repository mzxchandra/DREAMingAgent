"""
Context Agent Node: AI-Powered Sample Filtering

Uses LLM (Google Gemini) to intelligently determine which experimental 
conditions are relevant for analyzing a given transcription factor.

Falls back to a knowledge base mapping when LLM is unavailable.
"""

from typing import Dict, Any, List, Set, Optional
import pandas as pd
from loguru import logger

from ..state import AgentState
from ..llm.client import generate_structured_response
from ..llm.prompts import CONTEXT_AGENT_SYSTEM_PROMPT, format_context_prompt


# ============================================================================
# Configuration
# ============================================================================

# Whether to use LLM for context determination
USE_LLM_CONTEXT = True

# Minimum samples required for statistical validity
MIN_SAMPLES = 10


# ============================================================================
# Fallback Knowledge Base (used when LLM unavailable)
# ============================================================================

TF_CONDITION_MAP = {
    # Anaerobic regulators
    "fnr": ["anaerobic", "anaerobiosis", "oxygen", "fermentation", "no_oxygen"],
    "arca": ["anaerobic", "microaerobic", "oxygen"],
    
    # Stress response
    "rpoh": ["heat", "heat_shock", "temperature", "42c", "thermal"],
    "rpos": ["stationary", "starvation", "stress", "osmotic"],
    "soxs": ["oxidative", "superoxide", "paraquat", "redox"],
    "soxr": ["oxidative", "superoxide", "paraquat", "redox"],
    "oxyr": ["oxidative", "h2o2", "peroxide", "hydrogen_peroxide"],
    
    # Carbon metabolism
    "crp": ["glucose", "carbon", "camp", "catabolite", "lactose", "glycerol"],
    "fis": ["growth", "exponential", "rich_media"],
    "mlc": ["glucose", "pts", "carbon"],
    
    # Nitrogen metabolism
    "nac": ["nitrogen", "ammonia", "glutamine"],
    "ntrc": ["nitrogen", "ammonia", "limiting"],
    
    # Metal homeostasis
    "fur": ["iron", "fe", "metal"],
    "zur": ["zinc", "zn", "metal"],
    
    # DNA damage
    "lexa": ["dna_damage", "uv", "sos", "mitomycin", "norfloxacin"],
    "reca": ["dna_damage", "uv", "sos"],
    
    # Acid stress
    "gadr": ["acid", "ph", "low_ph", "acidic"],
    "gadx": ["acid", "ph", "low_ph", "acidic"],
    
    # Arabinose metabolism
    "arac": ["arabinose", "ara", "l-arabinose"],
    
    # Biofilm / motility
    "flhdc": ["flagella", "motility", "biofilm"],
}


# ============================================================================
# Main Context Agent Node
# ============================================================================

def context_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Context Agent Node: AI-Powered Sample Filtering.
    
    Uses LLM to intelligently determine which experimental conditions
    are relevant for analyzing the current transcription factors.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with active_sample_indices
    """
    logger.info("=== CONTEXT AGENT NODE: AI-Powered Sample Selection ===")
    
    current_batch_tfs = state.get("current_batch_tfs", [])
    metadata = state.get("metadata", pd.DataFrame())
    expression_matrix = state.get("expression_matrix", pd.DataFrame())
    bnumber_to_name = state.get("bnumber_to_gene_name", {})
    
    if metadata.empty or expression_matrix.empty:
        logger.warning("No metadata or expression data available")
        return {
            "active_sample_indices": list(expression_matrix.columns),
            "current_context": "no_metadata_available"
        }
    
    # Collect keywords for all TFs in batch
    all_keywords: Set[str] = set()
    context_descriptions: List[str] = []
    
    for tf in current_batch_tfs:
        tf_name = bnumber_to_name.get(tf, tf).lower()
        
        # Try LLM first
        if USE_LLM_CONTEXT:
            llm_result = _get_context_from_llm(tf_name)
            
            if llm_result:
                keywords = llm_result.get("relevant_conditions", [])
                all_keywords.update(k.lower() for k in keywords)
                
                bio_role = llm_result.get("biological_role", "")
                reasoning = llm_result.get("reasoning", "")
                context_descriptions.append(f"{tf_name}: {bio_role}")
                
                logger.info(f"[AI] TF {tf_name}: {keywords}")
                logger.debug(f"[AI] Reasoning: {reasoning}")
                continue
        
        # Fallback to knowledge base
        keywords = _get_context_from_knowledge_base(tf_name)
        if keywords:
            all_keywords.update(keywords)
            logger.info(f"[KB] TF {tf_name}: {keywords}")
    
    # If no specific context found, use all samples
    if not all_keywords:
        logger.info("No specific context found for TFs, using all samples")
        return {
            "active_sample_indices": list(expression_matrix.columns),
            "current_context": "all_conditions"
        }
    
    # Filter metadata based on keywords
    relevant_samples = _filter_samples_by_keywords(
        metadata,
        all_keywords,
        expression_matrix.columns
    )
    
    # Ensure minimum sample size
    if len(relevant_samples) < MIN_SAMPLES:
        logger.warning(
            f"Only {len(relevant_samples)} relevant samples found "
            f"(minimum: {MIN_SAMPLES}). Using all samples."
        )
        relevant_samples = list(expression_matrix.columns)
        context_description = f"fallback_to_all (keywords: {', '.join(list(all_keywords)[:5])})"
    else:
        context_description = "; ".join(context_descriptions) if context_descriptions else ", ".join(list(all_keywords)[:5])
    
    logger.info(
        f"Context filtering: {len(relevant_samples)} samples selected "
        f"from {len(expression_matrix.columns)} total"
    )
    
    return {
        "active_sample_indices": relevant_samples,
        "current_context": context_description
    }


# ============================================================================
# LLM-Powered Context Determination
# ============================================================================

def _get_context_from_llm(tf_name: str) -> Optional[Dict[str, Any]]:
    """
    Use LLM to determine relevant conditions for a TF.
    
    Args:
        tf_name: Name of the transcription factor
        
    Returns:
        Dictionary with relevant_conditions, biological_role, etc.
        or None if LLM fails
    """
    prompt = format_context_prompt(tf_name)
    
    response = generate_structured_response(
        prompt=prompt,
        system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT
    )
    
    if response and "relevant_conditions" in response:
        return response
    
    return None


# ============================================================================
# Knowledge Base Fallback
# ============================================================================

def _get_context_from_knowledge_base(tf_name: str) -> List[str]:
    """
    Get relevant conditions from hardcoded knowledge base.
    
    Args:
        tf_name: Name of the transcription factor
        
    Returns:
        List of relevant condition keywords
    """
    tf_lower = tf_name.lower()
    
    # Direct match
    if tf_lower in TF_CONDITION_MAP:
        return TF_CONDITION_MAP[tf_lower]
    
    # Partial match
    for known_tf, keywords in TF_CONDITION_MAP.items():
        if known_tf in tf_lower or tf_lower in known_tf:
            return keywords
    
    return []


# ============================================================================
# Sample Filtering
# ============================================================================

def _filter_samples_by_keywords(
    metadata: pd.DataFrame,
    keywords: Set[str],
    available_columns: List[str]
) -> List[str]:
    """
    Filter experiment IDs based on keyword matching in metadata.
    """
    matching_experiments = set()
    
    # Search all text columns in metadata
    for col in metadata.columns:
        if metadata[col].dtype == 'object':
            for idx, value in metadata[col].items():
                if pd.notna(value):
                    value_lower = str(value).lower()
                    for keyword in keywords:
                        if keyword in value_lower:
                            matching_experiments.add(str(idx))
                            break
    
    # Also check experiment IDs themselves
    for col in available_columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if keyword in col_lower:
                matching_experiments.add(col)
                break
    
    # Filter to columns that exist in expression matrix
    available_set = set(str(c) for c in available_columns)
    valid_samples = [s for s in matching_experiments if s in available_set]
    
    return valid_samples


# ============================================================================
# Utility Functions
# ============================================================================

def get_tf_context_description(tf_name: str) -> str:
    """Get human-readable context description for a TF."""
    # Try LLM
    if USE_LLM_CONTEXT:
        result = _get_context_from_llm(tf_name)
        if result:
            return f"{tf_name}: {result.get('biological_role', 'Unknown function')}"
    
    # Fallback
    tf_lower = tf_name.lower()
    if tf_lower in TF_CONDITION_MAP:
        conditions = TF_CONDITION_MAP[tf_lower]
        return f"{tf_name} responds to: {', '.join(conditions)}"
    
    return f"{tf_name}: unknown biological context"
