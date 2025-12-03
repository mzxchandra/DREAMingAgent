#!/usr/bin/env python3
"""
Demo: Analysis Agent in Isolation

This script demonstrates the Analysis Agent's CLR/MI calculation
without running the full LangGraph workflow.

Useful for:
- Understanding the statistical methodology
- Testing with custom data
- Educational purposes
"""

import numpy as np
import pandas as pd
from loguru import logger
import sys

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.nodes.analysis_agent import (
    _compute_clr_scores,
    _analyze_single_tf,
    compute_full_clr_matrix,
    Z_SCORE_THRESHOLD_HIGH,
    Z_SCORE_THRESHOLD_MODERATE
)


def create_demo_data(
    n_samples: int = 50,
    n_genes: int = 100,
    n_true_targets: int = 10,
    correlation_strength: float = 0.7,
    noise_level: float = 0.3
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Create synthetic expression data with known regulatory relationships.
    
    Args:
        n_samples: Number of experimental conditions
        n_genes: Total number of genes
        n_true_targets: Number of true TF targets
        correlation_strength: Strength of regulatory signal (0-1)
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (tf_expression, all_gene_expression, true_target_indices)
    """
    np.random.seed(42)
    
    # Generate TF expression (varying across conditions)
    tf_expression = np.random.normal(8.0, 2.0, n_samples)
    
    # Generate target genes with correlated expression
    true_targets = list(range(n_true_targets))
    all_gene_expression = np.zeros((n_genes, n_samples))
    
    for i in range(n_genes):
        if i in true_targets:
            # True targets: correlated with TF
            noise = np.random.normal(0, noise_level, n_samples)
            all_gene_expression[i, :] = (
                correlation_strength * tf_expression + 
                (1 - correlation_strength) * np.random.normal(8, 1.5, n_samples) +
                noise
            )
        else:
            # Non-targets: random expression
            all_gene_expression[i, :] = np.random.normal(8.0, 1.5, n_samples)
    
    return tf_expression, all_gene_expression, true_targets


def demo_clr_calculation():
    """Demonstrate CLR z-score calculation."""
    logger.info("=" * 60)
    logger.info("DEMO: CLR (Context Likelihood of Relatedness) Calculation")
    logger.info("=" * 60)
    
    # Create synthetic data
    logger.info("\n1. Creating synthetic expression data...")
    tf_expr, gene_expr, true_targets = create_demo_data(
        n_samples=50,
        n_genes=100,
        n_true_targets=10,
        correlation_strength=0.7
    )
    
    logger.info(f"   - TF expression: {len(tf_expr)} samples")
    logger.info(f"   - Gene matrix: {gene_expr.shape[0]} genes x {gene_expr.shape[1]} samples")
    logger.info(f"   - True targets (embedded signal): genes 0-9")
    
    # Compute CLR scores
    logger.info("\n2. Computing Mutual Information and CLR z-scores...")
    mi_scores, z_scores = _compute_clr_scores(tf_expr, gene_expr)
    
    logger.info(f"   - MI range: [{mi_scores.min():.4f}, {mi_scores.max():.4f}]")
    logger.info(f"   - Z-score range: [{z_scores.min():.4f}, {z_scores.max():.4f}]")
    
    # Analyze results
    logger.info("\n3. Results Analysis:")
    
    # Count significant hits
    high_sig = np.sum(z_scores >= Z_SCORE_THRESHOLD_HIGH)
    moderate_sig = np.sum(z_scores >= Z_SCORE_THRESHOLD_MODERATE)
    
    logger.info(f"   - High significance (z >= {Z_SCORE_THRESHOLD_HIGH}): {high_sig} genes")
    logger.info(f"   - Moderate significance (z >= {Z_SCORE_THRESHOLD_MODERATE}): {moderate_sig} genes")
    
    # Check true positive rate
    true_positive_high = sum(1 for i in true_targets if z_scores[i] >= Z_SCORE_THRESHOLD_HIGH)
    true_positive_mod = sum(1 for i in true_targets if z_scores[i] >= Z_SCORE_THRESHOLD_MODERATE)
    
    logger.info(f"\n4. Recovery of True Targets:")
    logger.info(f"   - True positives (high): {true_positive_high}/{len(true_targets)}")
    logger.info(f"   - True positives (moderate): {true_positive_mod}/{len(true_targets)}")
    
    # Show top hits
    logger.info("\n5. Top 15 Genes by Z-score:")
    sorted_indices = np.argsort(z_scores)[::-1]
    
    for rank, idx in enumerate(sorted_indices[:15], 1):
        is_true = "✓ TRUE TARGET" if idx in true_targets else ""
        logger.info(
            f"   {rank:2d}. Gene {idx:3d}: MI={mi_scores[idx]:.4f}, "
            f"z={z_scores[idx]:.4f} {is_true}"
        )
    
    # Calculate precision/recall
    predicted_positives = set(np.where(z_scores >= Z_SCORE_THRESHOLD_MODERATE)[0])
    true_positive_set = set(true_targets)
    
    tp = len(predicted_positives & true_positive_set)
    fp = len(predicted_positives - true_positive_set)
    fn = len(true_positive_set - predicted_positives)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\n6. Performance Metrics (at moderate threshold):")
    logger.info(f"   - Precision: {precision:.2%}")
    logger.info(f"   - Recall: {recall:.2%}")
    logger.info(f"   - F1 Score: {f1:.2%}")


def demo_with_dataframe():
    """Demonstrate analysis with pandas DataFrame input."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Analysis Agent with DataFrame Input")
    logger.info("=" * 60)
    
    # Create expression matrix as DataFrame
    np.random.seed(123)
    
    n_samples = 30
    n_genes = 50
    
    # Gene names (b-numbers)
    genes = [f"b{i:04d}" for i in range(n_genes)]
    tf_gene = "b0001"  # First gene is the TF
    
    # Sample names
    samples = [f"exp_{i:03d}" for i in range(n_samples)]
    
    # Generate expression data
    expression_data = np.random.normal(8.0, 1.5, (n_genes, n_samples))
    
    # Add regulatory signal for first 5 targets
    tf_idx = 0
    tf_expression = expression_data[tf_idx, :]
    
    for target_idx in range(1, 6):
        noise = np.random.normal(0, 0.3, n_samples)
        expression_data[target_idx, :] = 0.6 * tf_expression + 0.4 * expression_data[target_idx, :] + noise
    
    # Create DataFrame
    df = pd.DataFrame(expression_data, index=genes, columns=samples)
    
    logger.info(f"\n1. Expression Matrix:")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   TF: {tf_gene}")
    logger.info(f"   True targets: b0001 regulates b0002-b0005")
    
    # Run analysis
    logger.info("\n2. Running _analyze_single_tf()...")
    
    results = _analyze_single_tf(
        tf=tf_gene,
        data_slice=df,
        literature_graph=None,
        n_samples=n_samples
    )
    
    # Display results
    logger.info("\n3. Results:")
    
    if "_summary" in results:
        summary = results["_summary"]
        logger.info(f"   Total genes tested: {summary['total_genes_tested']}")
        logger.info(f"   High significance hits: {summary['significant_high']}")
        logger.info(f"   Moderate significance hits: {summary['significant_moderate']}")
    
    # Show significant hits
    logger.info("\n4. Significant Interactions:")
    for gene, data in sorted(
        [(g, d) for g, d in results.items() if not g.startswith("_") and isinstance(d, dict)],
        key=lambda x: x[1].get("z_score", 0),
        reverse=True
    )[:10]:
        z = data.get("z_score", 0)
        mi = data.get("mi", 0)
        sig = data.get("significance", "")
        logger.info(f"   {tf_gene} → {gene}: z={z:.3f}, MI={mi:.4f} [{sig}]")


def demo_correlation_vs_mi():
    """Compare correlation and mutual information."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Correlation vs Mutual Information")
    logger.info("=" * 60)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create different relationship types
    x = np.linspace(0, 4 * np.pi, n_samples)
    
    relationships = {
        "Linear Positive": np.random.normal(0, 0.3, n_samples) + x,
        "Linear Negative": np.random.normal(0, 0.3, n_samples) - x,
        "Quadratic (U-shape)": (x - 2*np.pi)**2 + np.random.normal(0, 2, n_samples),
        "Sinusoidal": np.sin(x) + np.random.normal(0, 0.2, n_samples),
        "Random (No relation)": np.random.normal(0, 1, n_samples)
    }
    
    logger.info("\nRelationship Type         | Pearson Corr | Mutual Info")
    logger.info("-" * 60)
    
    from scipy import stats
    
    for name, y in relationships.items():
        # Compute Pearson correlation
        corr, _ = stats.pearsonr(x, y)
        
        # Compute MI
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
        
        logger.info(f"{name:25s} | {corr:+.4f}      | {mi:.4f}")
    
    logger.info("\nKey insight: MI captures non-linear relationships that correlation misses!")
    logger.info("- Quadratic: Low correlation but high MI")
    logger.info("- Sinusoidal: ~Zero correlation but high MI")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    
    print("\n" + "=" * 70)
    print("DREAMing Agent - Analysis Agent Demonstration")
    print("=" * 70)
    
    demo_clr_calculation()
    demo_with_dataframe()
    demo_correlation_vs_mi()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

