"""
Unit tests for the Analysis Agent.

Tests the core CLR/MI calculation logic.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLRCalculation:
    """Test the CLR z-score calculation."""
    
    def test_compute_clr_scores_basic(self):
        """Test basic CLR calculation with known data."""
        from src.nodes.analysis_agent import _compute_clr_scores
        
        # Create simple test data
        np.random.seed(42)
        n_samples = 50
        n_targets = 20
        
        # TF expression
        tf_vector = np.random.normal(8, 2, n_samples)
        
        # Target matrix - first 5 are correlated with TF
        target_matrix = np.random.normal(8, 1.5, (n_targets, n_samples))
        
        # Add correlation for first 5 targets
        for i in range(5):
            target_matrix[i, :] = 0.7 * tf_vector + 0.3 * target_matrix[i, :] + np.random.normal(0, 0.2, n_samples)
        
        # Compute CLR scores
        mi_scores, z_scores = _compute_clr_scores(tf_vector, target_matrix)
        
        # Assertions
        assert len(mi_scores) == n_targets
        assert len(z_scores) == n_targets
        assert all(mi >= 0 for mi in mi_scores)  # MI should be non-negative
        assert all(z >= 0 for z in z_scores)     # CLR uses max(0, z)
        
        # Correlated targets should have higher z-scores
        mean_z_correlated = np.mean(z_scores[:5])
        mean_z_random = np.mean(z_scores[5:])
        assert mean_z_correlated > mean_z_random, "Correlated targets should have higher z-scores"
    
    def test_compute_clr_constant_tf(self):
        """Test CLR with constant TF expression (edge case)."""
        from src.nodes.analysis_agent import _compute_clr_scores
        
        # Constant TF (no variance)
        tf_vector = np.ones(50) * 8.0
        target_matrix = np.random.normal(8, 1.5, (10, 50))
        
        mi_scores, z_scores = _compute_clr_scores(tf_vector, target_matrix)
        
        # Should return zeros for constant TF
        assert all(mi == 0 for mi in mi_scores)
        assert all(z == 0 for z in z_scores)
    
    def test_analyze_single_tf(self):
        """Test the full single-TF analysis."""
        from src.nodes.analysis_agent import _analyze_single_tf
        
        # Create DataFrame
        np.random.seed(42)
        genes = [f"b{i:04d}" for i in range(50)]
        samples = [f"exp_{i:03d}" for i in range(30)]
        
        expression_data = np.random.normal(8, 1.5, (50, 30))
        df = pd.DataFrame(expression_data, index=genes, columns=samples)
        
        # Analyze
        results = _analyze_single_tf(
            tf="b0001",
            data_slice=df,
            literature_graph=None,
            n_samples=30
        )
        
        # Check structure
        assert "_summary" in results
        assert "total_genes_tested" in results["_summary"]
        assert results["_summary"]["total_genes_tested"] == 49  # All except TF
    
    def test_analyze_single_tf_not_in_matrix(self):
        """Test handling of TF not in expression matrix."""
        from src.nodes.analysis_agent import _analyze_single_tf
        
        df = pd.DataFrame(
            np.random.normal(8, 1.5, (10, 20)),
            index=[f"b{i:04d}" for i in range(10)],
            columns=[f"exp_{i:03d}" for i in range(20)]
        )
        
        results = _analyze_single_tf(
            tf="b9999",  # Not in matrix
            data_slice=df,
            literature_graph=None,
            n_samples=20
        )
        
        assert results == {"error": "TF_NOT_IN_MATRIX"}


class TestStatisticalFunctions:
    """Test utility statistical functions."""
    
    def test_mutual_information_basic(self):
        """Test basic MI calculation."""
        from src.utils.statistics import compute_mutual_information
        
        np.random.seed(42)
        
        # Perfectly correlated
        x = np.linspace(0, 10, 100)
        y = x + np.random.normal(0, 0.1, 100)
        mi_correlated = compute_mutual_information(x, y)
        
        # Random (independent)
        y_random = np.random.normal(0, 1, 100)
        mi_random = compute_mutual_information(x, y_random)
        
        # Correlated should have higher MI
        assert mi_correlated > mi_random
    
    def test_pairwise_mi_matrix(self):
        """Test pairwise MI calculation for TF vs all targets."""
        from src.utils.statistics import compute_pairwise_mi_matrix
        
        np.random.seed(42)
        n_samples = 50
        
        tf_vector = np.random.normal(8, 2, n_samples)
        target_matrix = np.random.normal(8, 1.5, (20, n_samples))
        
        mi_scores = compute_pairwise_mi_matrix(tf_vector, target_matrix)
        
        assert len(mi_scores) == 20
        assert all(mi >= 0 for mi in mi_scores)
    
    def test_zscore_to_pvalue(self):
        """Test z-score to p-value conversion."""
        from src.nodes.analysis_agent import _zscore_to_pvalue
        
        z_scores = np.array([0, 1, 2, 3, 4])
        p_values = _zscore_to_pvalue(z_scores)
        
        # Higher z-score = lower p-value
        assert all(p_values[i] > p_values[i+1] for i in range(len(p_values)-1))
        
        # z=0 should give p=0.5
        assert abs(p_values[0] - 0.5) < 0.01


class TestDataParsers:
    """Test data parsing utilities."""
    
    def test_synthetic_data_creation(self):
        """Test synthetic test data generation."""
        from src.utils.parsers import create_synthetic_test_data
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = create_synthetic_test_data(
                n_genes=50,
                n_experiments=30,
                n_tfs=5,
                output_dir=tmpdir
            )
            
            # Check files were created
            assert paths["network"].exists()
            assert paths["gene_product"].exists()
            assert paths["expression"].exists()
            assert paths["metadata"].exists()
    
    def test_normalize_effect(self):
        """Test interaction effect normalization."""
        from src.utils.parsers import _normalize_effect
        
        assert _normalize_effect("+") == "+"
        assert _normalize_effect("-") == "-"
        assert _normalize_effect("activator") == "+"
        assert _normalize_effect("repressor") == "-"
        assert _normalize_effect("dual") == "+-"
        assert _normalize_effect("unknown") == "?"


class TestReconciler:
    """Test reconciliation logic."""
    
    def test_determine_status_validated(self):
        """Test validation status determination."""
        from src.nodes.reconciler import _determine_status
        
        # Strong evidence + high data = Validated
        status, notes = _determine_status(
            evidence_level="Strong",
            z_score=5.0,
            tf_name="test_tf",
            gene_name="test_gene"
        )
        assert status == "Validated"
    
    def test_determine_status_silent(self):
        """Test conditional silence status."""
        from src.nodes.reconciler import _determine_status
        
        # Strong evidence + low data = ConditionSilent
        status, notes = _determine_status(
            evidence_level="Strong",
            z_score=0.5,
            tf_name="test_tf",
            gene_name="test_gene"
        )
        assert status == "ConditionSilent"
    
    def test_determine_status_false_positive(self):
        """Test probable false positive status."""
        from src.nodes.reconciler import _determine_status
        
        # Weak evidence + low data = ProbableFalsePos
        status, notes = _determine_status(
            evidence_level="Weak",
            z_score=0.3,
            tf_name="test_tf",
            gene_name="test_gene"
        )
        assert status == "ProbableFalsePos"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


