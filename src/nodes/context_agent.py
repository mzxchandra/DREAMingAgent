"""
Context Agent Node: Intelligent sample filtering.

Filters M3D experiments based on TF-specific biological context,
ensuring that correlation analysis uses relevant conditions.
"""

from typing import Dict, Any, List, Set
import pandas as pd
from loguru import logger

from ..state import AgentState


# Known TF to condition mappings (biological context)
# This acts as a simplified "Gensor Unit" - mapping TFs to their activating conditions
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


def context_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Context Agent Node: Filters samples based on TF context.
    
    This node is the "Contextualizer" of the system. It:
    1. Identifies the biological function of current TFs
    2. Filters M3D metadata to find relevant experiments
    3. Returns sample indices for context-aware analysis
    
    If no specific context is found for a TF, all samples are used.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with active_sample_indices
    """
    logger.info("=== CONTEXT AGENT NODE: Filtering relevant samples ===")
    
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
    
    # Collect all relevant keywords for the current TF batch
    all_keywords: Set[str] = set()
    
    for tf in current_batch_tfs:
        # Convert b-number to name for lookup
        tf_name = bnumber_to_name.get(tf, tf).lower()
        
        # Check TF condition map
        if tf_name in TF_CONDITION_MAP:
            keywords = TF_CONDITION_MAP[tf_name]
            all_keywords.update(keywords)
            logger.info(f"TF {tf_name}: Context keywords = {keywords}")
        else:
            # Try partial matching
            for known_tf, keywords in TF_CONDITION_MAP.items():
                if known_tf in tf_name or tf_name in known_tf:
                    all_keywords.update(keywords)
                    logger.info(f"TF {tf_name} (partial match to {known_tf}): {keywords}")
                    break
    
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
    
    # Ensure minimum sample size for statistical validity
    MIN_SAMPLES = 10
    
    if len(relevant_samples) < MIN_SAMPLES:
        logger.warning(
            f"Only {len(relevant_samples)} relevant samples found "
            f"(minimum: {MIN_SAMPLES}). Using all samples."
        )
        relevant_samples = list(expression_matrix.columns)
        context_description = f"fallback_to_all (keywords: {', '.join(all_keywords)})"
    else:
        context_description = f"filtered ({len(relevant_samples)} samples): {', '.join(all_keywords)}"
    
    logger.info(
        f"Context filtering: {len(relevant_samples)} samples selected "
        f"from {len(expression_matrix.columns)} total"
    )
    
    return {
        "active_sample_indices": relevant_samples,
        "current_context": context_description
    }


def _filter_samples_by_keywords(
    metadata: pd.DataFrame,
    keywords: Set[str],
    available_columns: List[str]
) -> List[str]:
    """
    Filter experiment IDs based on keyword matching in metadata.
    
    Searches all text columns in metadata for keyword matches.
    """
    matching_experiments = set()
    
    # Convert metadata columns to searchable strings
    for col in metadata.columns:
        if metadata[col].dtype == 'object':  # Text columns only
            for idx, value in metadata[col].items():
                if pd.notna(value):
                    value_lower = str(value).lower()
                    # Check if any keyword matches
                    for keyword in keywords:
                        if keyword in value_lower:
                            matching_experiments.add(str(idx))
                            break
    
    # Also check experiment IDs themselves (column names in expression matrix)
    for col in available_columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if keyword in col_lower:
                matching_experiments.add(col)
                break
    
    # Filter to only include columns that exist in expression matrix
    available_set = set(str(c) for c in available_columns)
    valid_samples = [s for s in matching_experiments if s in available_set]
    
    return valid_samples


def get_tf_context_description(tf_name: str) -> str:
    """
    Get human-readable context description for a TF.
    
    Args:
        tf_name: Name of the transcription factor
        
    Returns:
        Description of the TF's biological context
    """
    tf_lower = tf_name.lower()
    
    if tf_lower in TF_CONDITION_MAP:
        conditions = TF_CONDITION_MAP[tf_lower]
        return f"{tf_name} responds to: {', '.join(conditions)}"
    
    return f"{tf_name}: unknown biological context"

