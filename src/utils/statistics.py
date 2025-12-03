"""
Statistical Functions for Gene Regulatory Network Inference.

Implements:
- Mutual Information (MI) calculation
- Context Likelihood of Relatedness (CLR) algorithm
- Z-score normalization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from loguru import logger


def compute_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> float:
    """
    Compute mutual information between two continuous variables.
    
    Uses sklearn's mutual_info_regression which implements a k-nearest
    neighbor based estimator (Kraskov et al., 2004).
    
    Args:
        x: First variable (1D array)
        y: Second variable (1D array)
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed for reproducibility
        
    Returns:
        Mutual information value (non-negative)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Ensure same length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # Handle constant arrays
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    # Reshape for sklearn
    X = x.reshape(-1, 1)
    
    mi = mutual_info_regression(
        X, y,
        n_neighbors=n_neighbors,
        random_state=random_state
    )[0]
    
    return max(0.0, mi)  # MI should be non-negative


def compute_mutual_information_matrix(
    data: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute pairwise mutual information matrix for all variables.
    
    Args:
        data: 2D array where rows are variables (genes) and columns are samples
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed
        
    Returns:
        Symmetric MI matrix (n_genes x n_genes)
    """
    n_genes = data.shape[0]
    mi_matrix = np.zeros((n_genes, n_genes))
    
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            mi = compute_mutual_information(
                data[i, :], data[j, :],
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    return mi_matrix


def compute_pairwise_mi_matrix(
    tf_vector: np.ndarray,
    target_matrix: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute mutual information between one TF and all target genes.
    
    This is optimized for the common case where we want MI between
    one regulator and many potential targets.
    
    Args:
        tf_vector: Expression vector of the TF (1D, n_samples)
        target_matrix: Expression matrix of targets (n_targets x n_samples)
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed
        
    Returns:
        1D array of MI values for each target
    """
    tf_vector = np.asarray(tf_vector).ravel()
    target_matrix = np.asarray(target_matrix)
    
    n_targets = target_matrix.shape[0]
    mi_scores = np.zeros(n_targets)
    
    # Use sklearn's vectorized implementation
    # Transpose target_matrix for sklearn: (n_samples, n_features)
    X = target_matrix.T  # Now: (n_samples, n_targets)
    y = tf_vector
    
    # Filter out constant features to avoid warnings
    valid_mask = np.std(X, axis=0) > 1e-10
    
    if not any(valid_mask):
        return mi_scores
    
    X_valid = X[:, valid_mask]
    
    mi_valid = mutual_info_regression(
        X_valid, y,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    
    # Map back to full array
    mi_scores[valid_mask] = mi_valid
    
    return np.maximum(mi_scores, 0.0)  # Ensure non-negative


def compute_clr_scores(
    mi_matrix: np.ndarray,
    tf_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute Context Likelihood of Relatedness (CLR) scores.
    
    CLR normalizes MI values by the background distribution for each gene,
    helping to distinguish direct from indirect regulatory relationships.
    
    Formula: z_ij = sqrt(z_i^2 + z_j^2)
    where z_i = max(0, (MI_ij - μ_i) / σ_i)
    
    Args:
        mi_matrix: Symmetric MI matrix (n_genes x n_genes)
        tf_indices: Optional list of TF row indices. If provided, only
                   compute CLR for TF rows (faster).
    
    Returns:
        CLR z-score matrix (same shape as input)
    """
    n = mi_matrix.shape[0]
    
    # Compute row-wise z-scores
    # For each gene i, z_i(j) = (MI_ij - mean_i) / std_i
    row_means = np.mean(mi_matrix, axis=1, keepdims=True)
    row_stds = np.std(mi_matrix, axis=1, keepdims=True)
    
    # Avoid division by zero
    row_stds = np.where(row_stds < 1e-10, 1.0, row_stds)
    
    # Row-wise z-scores
    z_rows = (mi_matrix - row_means) / row_stds
    z_rows = np.maximum(z_rows, 0.0)  # CLR uses max(0, z)
    
    # Column-wise z-scores
    col_means = np.mean(mi_matrix, axis=0, keepdims=True)
    col_stds = np.std(mi_matrix, axis=0, keepdims=True)
    col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)
    
    z_cols = (mi_matrix - col_means) / col_stds
    z_cols = np.maximum(z_cols, 0.0)
    
    # CLR score: sqrt(z_i^2 + z_j^2)
    clr_matrix = np.sqrt(z_rows**2 + z_cols**2)
    
    return clr_matrix


def compute_clr_for_tf(
    tf_expression: np.ndarray,
    all_gene_expression: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLR-corrected z-scores for one TF against all genes.
    
    This is the optimized single-TF version used by the Analysis Agent.
    
    Args:
        tf_expression: TF expression vector (n_samples,)
        all_gene_expression: All genes expression matrix (n_genes x n_samples)
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed
        
    Returns:
        Tuple of:
        - MI scores (n_genes,)
        - Z-scores (n_genes,)
    """
    # Compute MI between TF and all genes
    mi_scores = compute_pairwise_mi_matrix(
        tf_expression,
        all_gene_expression,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    
    # Compute z-scores based on background distribution
    # The background is the distribution of MI values for this TF
    mu = np.mean(mi_scores)
    sigma = np.std(mi_scores)
    
    if sigma < 1e-10:
        z_scores = np.zeros_like(mi_scores)
    else:
        z_scores = (mi_scores - mu) / sigma
    
    return mi_scores, z_scores


def mi_to_pvalue(z_score: float, n_samples: int) -> float:
    """
    Convert z-score to approximate p-value.
    
    Uses the standard normal distribution as an approximation.
    For more accurate p-values, permutation testing should be used.
    
    Args:
        z_score: The z-score
        n_samples: Number of samples (for reference, not used in basic calc)
        
    Returns:
        Two-tailed p-value
    """
    # One-tailed p-value for positive z-scores
    p_value = 1 - stats.norm.cdf(z_score)
    return p_value


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
        
    Returns:
        Boolean array indicating which hypotheses are rejected
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    # BH threshold: p_i <= (i/n) * alpha
    thresholds = np.arange(1, n + 1) / n * alpha
    
    # Find largest k where p_(k) <= threshold_k
    rejected = sorted_pvals <= thresholds
    
    # Find the cutoff
    if not np.any(rejected):
        return np.zeros(n, dtype=bool)
    
    k = np.max(np.where(rejected)[0])
    
    # Reject all hypotheses with p-value <= p_(k)
    result = np.zeros(n, dtype=bool)
    result[sorted_indices[:k+1]] = True
    
    return result


def discretize_expression(
    data: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'quantile'
) -> np.ndarray:
    """
    Discretize continuous expression values for MI calculation.
    
    Some MI estimators work better with discretized data.
    
    Args:
        data: Continuous expression data (genes x samples or 1D)
        n_bins: Number of bins
        strategy: 'uniform', 'quantile', or 'kmeans'
        
    Returns:
        Discretized data (same shape)
    """
    original_shape = data.shape
    data_flat = data.reshape(-1, 1) if data.ndim == 1 else data.T
    
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy=strategy
    )
    
    discretized = discretizer.fit_transform(data_flat)
    
    if original_shape != discretized.shape:
        discretized = discretized.T.reshape(original_shape)
    
    return discretized.astype(int)


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix.
    
    Provided as an alternative to MI for quick comparisons.
    
    Args:
        data: Expression matrix (genes x samples)
        
    Returns:
        Correlation matrix (genes x genes)
    """
    return np.corrcoef(data)


def compute_spearman_correlation(
    tf_vector: np.ndarray,
    target_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute Spearman rank correlation between TF and all targets.
    
    More robust to outliers than Pearson correlation.
    
    Args:
        tf_vector: TF expression (n_samples,)
        target_matrix: Target expression (n_genes x n_samples)
        
    Returns:
        Array of Spearman correlations
    """
    n_genes = target_matrix.shape[0]
    correlations = np.zeros(n_genes)
    
    for i in range(n_genes):
        corr, _ = stats.spearmanr(tf_vector, target_matrix[i, :])
        correlations[i] = corr if not np.isnan(corr) else 0.0
    
    return correlations

