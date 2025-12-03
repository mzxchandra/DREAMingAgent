"""
Analysis Agent Node: The Computational Heart

Implements the Context Likelihood of Relatedness (CLR) algorithm for
gene regulatory network inference. This node computes mutual information
between transcription factors and all potential target genes, then applies
CLR normalization to identify statistically significant regulatory relationships.

This is the core statistical engine of the reconciliation system.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from loguru import logger

from ..state import AgentState, EdgeAnalysis


# ============================================================================
# Configuration Constants
# ============================================================================

# Z-score threshold for significance
Z_SCORE_THRESHOLD_HIGH = 4.0      # High confidence
Z_SCORE_THRESHOLD_MODERATE = 2.0  # Moderate confidence

# Minimum samples for reliable MI estimation
MIN_SAMPLES_FOR_MI = 15

# Minimum mean expression for TF to be considered "expressed"
MIN_TF_EXPRESSION = 4.0  # log2 intensity

# Number of neighbors for MI estimation (k-NN based estimator)
MI_N_NEIGHBORS = 3

# Random state for reproducibility
RANDOM_STATE = 42


# ============================================================================
# Main Analysis Agent Node
# ============================================================================

def analysis_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Analysis Agent Node: Computes CLR-corrected Mutual Information.
    
    This is the computational heart of the system. It:
    1. Extracts TF expression vectors from the filtered sample set
    2. Computes Mutual Information (MI) between TF and all genes
    3. Applies CLR normalization to obtain z-scores
    4. Filters results by significance threshold
    5. Returns statistical confidence for each potential edge
    
    The agent does not "reason" biologically - it reasons statistically.
    Biology-aware decisions are made by the Reconciler node.
    
    Args:
        state: Current agent state with:
            - current_batch_tfs: List of TFs (b-numbers) to analyze
            - active_sample_indices: Context-filtered experiment columns
            - expression_matrix: Full M3D expression matrix
            
    Returns:
        Updated state with analysis_results populated
    """
    logger.info("=== ANALYSIS AGENT NODE: Computing CLR-corrected MI ===")
    
    # Extract inputs from state
    tf_batch = state.get("current_batch_tfs", [])
    valid_samples = state.get("active_sample_indices", [])
    expression_matrix = state.get("expression_matrix", pd.DataFrame())
    literature_graph = state.get("literature_graph")
    
    if expression_matrix.empty:
        logger.error("Expression matrix is empty, cannot perform analysis")
        return {
            "analysis_results": {},
            "errors": state.get("errors", []) + ["Empty expression matrix"]
        }
    
    if not tf_batch:
        logger.warning("No TFs in current batch")
        return {"analysis_results": {}}
    
    # Slice expression matrix to valid samples
    available_samples = [s for s in valid_samples if s in expression_matrix.columns]
    
    if len(available_samples) < MIN_SAMPLES_FOR_MI:
        logger.warning(
            f"Insufficient samples ({len(available_samples)}) for MI estimation. "
            f"Minimum required: {MIN_SAMPLES_FOR_MI}. Using all samples."
        )
        available_samples = list(expression_matrix.columns)
    
    logger.info(f"Analyzing {len(tf_batch)} TFs across {len(available_samples)} samples")
    
    # Create sliced matrix
    data_slice = expression_matrix[available_samples]
    
    # Store results
    analysis_results = state.get("analysis_results", {}).copy()
    
    # Process each TF in the batch
    for tf in tf_batch:
        logger.info(f"Processing TF: {tf}")
        
        tf_result = _analyze_single_tf(
            tf=tf,
            data_slice=data_slice,
            literature_graph=literature_graph,
            n_samples=len(available_samples)
        )
        
        analysis_results[tf] = tf_result
    
    logger.info(f"Analysis complete for {len(tf_batch)} TFs")
    
    return {"analysis_results": analysis_results}


# ============================================================================
# Core Analysis Functions
# ============================================================================

def _analyze_single_tf(
    tf: str,
    data_slice: pd.DataFrame,
    literature_graph: Any,
    n_samples: int
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze regulatory relationships for a single TF.
    
    Args:
        tf: TF identifier (b-number)
        data_slice: Expression matrix slice (genes x samples)
        literature_graph: NetworkX graph of literature interactions
        n_samples: Number of samples being analyzed
        
    Returns:
        Dictionary mapping target genes to their analysis results
    """
    # Check if TF exists in the expression data
    if tf not in data_slice.index:
        logger.warning(f"TF {tf} not found in expression matrix")
        return {"error": "TF_NOT_IN_MATRIX"}
    
    # Get TF expression vector
    tf_vector = data_slice.loc[tf].values.astype(np.float64)
    
    # Check if TF is expressed (detection call)
    tf_mean_expression = np.mean(tf_vector)
    if tf_mean_expression < MIN_TF_EXPRESSION:
        logger.info(
            f"TF {tf} has low expression (mean={tf_mean_expression:.2f}). "
            "May be contextually inactive."
        )
        return {
            "status": "TF_LOW_EXPRESSION",
            "mean_expression": float(tf_mean_expression)
        }
    
    # Get target gene matrix (all genes except the TF itself)
    target_genes = [g for g in data_slice.index if g != tf]
    target_matrix = data_slice.loc[target_genes].values.astype(np.float64)
    
    # Compute MI and CLR z-scores
    mi_scores, z_scores = _compute_clr_scores(
        tf_vector=tf_vector,
        target_matrix=target_matrix
    )
    
    # Compute p-values from z-scores
    p_values = _zscore_to_pvalue(z_scores)
    
    # Build results dictionary
    results = {}
    
    for i, gene in enumerate(target_genes):
        z = z_scores[i]
        mi = mi_scores[i]
        p = p_values[i]
        
        # Determine significance level
        if z >= Z_SCORE_THRESHOLD_HIGH:
            significance = "High"
        elif z >= Z_SCORE_THRESHOLD_MODERATE:
            significance = "Moderate"
        else:
            significance = "Low"
        
        # Only store significant results to save memory
        # But always store results for known literature targets
        is_literature_target = (
            literature_graph is not None and 
            literature_graph.has_edge(tf, gene)
        )
        
        if z >= Z_SCORE_THRESHOLD_MODERATE or is_literature_target:
            results[gene] = {
                "mi": float(mi),
                "z_score": float(z),
                "p_value": float(p),
                "significance": significance,
                "sample_count": n_samples,
                "is_literature_target": is_literature_target
            }
    
    # Add summary statistics
    results["_summary"] = {
        "total_genes_tested": len(target_genes),
        "significant_high": sum(1 for z in z_scores if z >= Z_SCORE_THRESHOLD_HIGH),
        "significant_moderate": sum(1 for z in z_scores if z >= Z_SCORE_THRESHOLD_MODERATE),
        "tf_mean_expression": float(tf_mean_expression),
        "tf_std_expression": float(np.std(tf_vector))
    }
    
    logger.info(
        f"TF {tf}: {results['_summary']['significant_high']} high, "
        f"{results['_summary']['significant_moderate']} moderate significance hits"
    )
    
    return results


def _compute_clr_scores(
    tf_vector: np.ndarray,
    target_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Context Likelihood of Relatedness (CLR) scores.
    
    The CLR algorithm normalizes raw mutual information by the background
    distribution, helping distinguish direct from indirect regulatory
    relationships.
    
    Steps:
    1. Compute MI between TF and all targets
    2. Calculate z-scores: z = (MI - μ) / σ
    3. Apply max(0, z) to handle negative z-scores
    
    For the full CLR algorithm on complete networks:
    z_ij = sqrt(z_i² + z_j²)
    
    For single TF analysis (this function), we use simplified row-z normalization.
    
    Args:
        tf_vector: TF expression values (n_samples,)
        target_matrix: Target gene expression (n_genes x n_samples)
        
    Returns:
        Tuple of (MI scores, z-scores)
    """
    n_targets = target_matrix.shape[0]
    
    # Handle edge cases
    if np.std(tf_vector) < 1e-10:
        logger.warning("TF has constant expression, returning zeros")
        return np.zeros(n_targets), np.zeros(n_targets)
    
    # Compute MI between TF and all targets using sklearn
    # sklearn expects: X = (n_samples, n_features), y = (n_samples,)
    X = target_matrix.T  # Transpose to (n_samples, n_genes)
    y = tf_vector
    
    # Identify and handle constant features
    feature_stds = np.std(X, axis=0)
    valid_mask = feature_stds > 1e-10
    
    mi_scores = np.zeros(n_targets)
    
    if np.sum(valid_mask) > 0:
        # Only compute MI for non-constant genes
        mi_valid = mutual_info_regression(
            X[:, valid_mask],
            y,
            n_neighbors=MI_N_NEIGHBORS,
            random_state=RANDOM_STATE
        )
        mi_scores[valid_mask] = np.maximum(mi_valid, 0)  # MI should be non-negative
    
    # Compute z-scores (CLR row normalization)
    # Use only valid (non-zero) MI values for statistics
    nonzero_mask = mi_scores > 0
    
    if np.sum(nonzero_mask) < 2:
        logger.warning("Too few non-zero MI values for z-score calculation")
        return mi_scores, np.zeros(n_targets)
    
    # Calculate background statistics
    mu = np.mean(mi_scores[nonzero_mask])
    sigma = np.std(mi_scores[nonzero_mask])
    
    if sigma < 1e-10:
        logger.warning("MI values have near-zero variance")
        return mi_scores, np.zeros(n_targets)
    
    # Compute z-scores
    z_scores = (mi_scores - mu) / sigma
    
    # Apply CLR max(0, z) transformation
    z_scores = np.maximum(z_scores, 0)
    
    return mi_scores, z_scores


def _zscore_to_pvalue(z_scores: np.ndarray) -> np.ndarray:
    """
    Convert z-scores to one-tailed p-values.
    
    Uses the standard normal distribution.
    
    Args:
        z_scores: Array of z-scores
        
    Returns:
        Array of p-values
    """
    return 1 - stats.norm.cdf(z_scores)


# ============================================================================
# Advanced Analysis Functions
# ============================================================================

def compute_full_clr_matrix(
    expression_matrix: pd.DataFrame,
    sample_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute full CLR matrix for all gene pairs.
    
    This is computationally expensive (O(n²)) but provides the most
    accurate CLR scores by normalizing over both rows and columns.
    
    Formula: z_ij = sqrt(z_i² + z_j²)
    where z_i = max(0, (MI_ij - μ_i) / σ_i)
    
    Args:
        expression_matrix: Full expression matrix (genes x samples)
        sample_subset: Optional list of sample columns to use
        
    Returns:
        DataFrame of CLR z-scores (genes x genes)
    """
    if sample_subset:
        data = expression_matrix[sample_subset].values
    else:
        data = expression_matrix.values
    
    genes = expression_matrix.index.tolist()
    n_genes = len(genes)
    
    logger.info(f"Computing full CLR matrix for {n_genes} genes...")
    
    # Compute pairwise MI matrix
    mi_matrix = np.zeros((n_genes, n_genes))
    
    for i in range(n_genes):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{n_genes} genes")
        
        for j in range(i + 1, n_genes):
            mi = _compute_mi_pair(data[i, :], data[j, :])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    # Apply CLR normalization
    # Row-wise z-scores
    row_means = np.mean(mi_matrix, axis=1, keepdims=True)
    row_stds = np.std(mi_matrix, axis=1, keepdims=True)
    row_stds = np.where(row_stds < 1e-10, 1.0, row_stds)
    z_rows = np.maximum(0, (mi_matrix - row_means) / row_stds)
    
    # Column-wise z-scores
    col_means = np.mean(mi_matrix, axis=0, keepdims=True)
    col_stds = np.std(mi_matrix, axis=0, keepdims=True)
    col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)
    z_cols = np.maximum(0, (mi_matrix - col_means) / col_stds)
    
    # Combined CLR score
    clr_matrix = np.sqrt(z_rows**2 + z_cols**2)
    
    return pd.DataFrame(clr_matrix, index=genes, columns=genes)


def _compute_mi_pair(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information between two vectors.
    """
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    mi = mutual_info_regression(
        x.reshape(-1, 1),
        y,
        n_neighbors=MI_N_NEIGHBORS,
        random_state=RANDOM_STATE
    )[0]
    
    return max(0.0, mi)


def analyze_tf_regulon(
    tf: str,
    known_targets: List[str],
    expression_matrix: pd.DataFrame,
    sample_indices: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Focused analysis of a TF's known regulon.
    
    This is more efficient when we only care about validating
    known literature interactions rather than discovering new ones.
    
    Args:
        tf: TF identifier
        known_targets: List of known target gene identifiers
        expression_matrix: Expression matrix
        sample_indices: Optional sample subset
        
    Returns:
        Analysis results for the known regulon
    """
    if sample_indices:
        data = expression_matrix[sample_indices]
    else:
        data = expression_matrix
    
    if tf not in data.index:
        return {"error": "TF_NOT_IN_MATRIX"}
    
    # Filter to genes that exist in matrix
    valid_targets = [g for g in known_targets if g in data.index]
    
    if not valid_targets:
        return {"error": "NO_VALID_TARGETS"}
    
    tf_vector = data.loc[tf].values
    target_data = data.loc[valid_targets].values
    
    mi_scores, z_scores = _compute_clr_scores(tf_vector, target_data)
    p_values = _zscore_to_pvalue(z_scores)
    
    results = {}
    for i, gene in enumerate(valid_targets):
        results[gene] = {
            "mi": float(mi_scores[i]),
            "z_score": float(z_scores[i]),
            "p_value": float(p_values[i]),
            "significance": "High" if z_scores[i] >= Z_SCORE_THRESHOLD_HIGH else
                          "Moderate" if z_scores[i] >= Z_SCORE_THRESHOLD_MODERATE else "Low"
        }
    
    return results


def compute_correlation_complement(
    tf: str,
    expression_matrix: pd.DataFrame,
    sample_indices: List[str]
) -> Dict[str, float]:
    """
    Compute Spearman correlation as a complement to MI.
    
    Correlation provides directional information (+/-) that MI lacks.
    This can be used to validate the sign of regulation (activation vs repression).
    
    Args:
        tf: TF identifier
        expression_matrix: Expression matrix
        sample_indices: Sample subset
        
    Returns:
        Dictionary mapping genes to Spearman correlation coefficients
    """
    data = expression_matrix[sample_indices]
    
    if tf not in data.index:
        return {}
    
    tf_vector = data.loc[tf].values
    correlations = {}
    
    for gene in data.index:
        if gene == tf:
            continue
        
        gene_vector = data.loc[gene].values
        
        # Compute Spearman correlation
        corr, p = stats.spearmanr(tf_vector, gene_vector)
        
        if not np.isnan(corr):
            correlations[gene] = {
                "spearman_rho": float(corr),
                "spearman_p": float(p)
            }
    
    return correlations

