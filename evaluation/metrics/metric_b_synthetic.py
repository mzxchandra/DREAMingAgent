from pathlib import Path
from typing import Dict, Any, Set, List
import pandas as pd
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

from .base_metric import BaseMetric
from ..utils.metrics_calculator import compute_classification_metrics, compute_detection_rate
from ..utils.plot_generator import plot_confusion_matrix, plot_bar_chart, plot_scatter
from src.utils.parsers import create_synthetic_test_data
from src.workflow import run_reconciliation

logger = logging.getLogger(__name__)

class MetricBSynthetic(BaseMetric):
    """Metric B: Synthetic Ground Truth Recovery."""

    def prepare_data(self) -> Dict[str, Any]:
        logger.info("Generating synthetic data...")
        paths = create_synthetic_test_data(
            n_genes=self.config.synthetic_n_genes,
            n_experiments=self.config.synthetic_n_experiments,
            n_tfs=self.config.synthetic_n_tfs,
            output_dir=self.config.data_dir / "synthetic"
        )
        ground_truth_edges = set()
        network_path = paths["network"]
        
        # Read Ground Truth
        df = pd.read_csv(network_path, sep='\t', comment='#')
        for _, row in df.iterrows():
            tf_name = row.iloc[1] 
            target_name = row.iloc[4] 
            edge_id = f"{tf_name}→{target_name}"
            ground_truth_edges.add(edge_id)
            
        logger.info(f"Extracted {len(ground_truth_edges)} ground truth edges.")
        return {"paths": paths, "ground_truth_edges": ground_truth_edges}

    def run_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.ground_truth_edges = data["ground_truth_edges"] # Store for scoring
        paths = data["paths"]
        logger.info("Running reconciliation...")
        final_state = run_reconciliation(
            regulondb_network_path=str(paths["network"]),
            regulondb_gene_product_path=str(paths["gene_product"]),
            m3d_expression_path=str(paths["expression"]),
            m3d_metadata_path=str(paths["metadata"])
        )
        return {
            "reconciliation_log": final_state.get("reconciliation_log", [])
        }

    def compute_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        logs = results["reconciliation_log"]
        ground_truth = self.ground_truth_edges
        
        tp = 0 
        fn = 0
        reconciled_map = {}
        for entry in logs:
            edge_id = f"{entry['source_tf_name']}→{entry['target_gene_name']}"
            reconciled_map[edge_id] = entry['reconciliation_status']
            
        for edge_id in ground_truth:
            status = reconciled_map.get(edge_id)
            if status in ["Validated", "ConditionSilent", "NovelHypothesis"]:
                tp += 1
            else:
                fn += 1 # ProbableFalsePositive or Missing
                
        recall = tp / len(ground_truth) if ground_truth else 0.0
        return {"recall": recall, "tp": tp, "fn": fn, "total_edges": len(ground_truth)}

    def generate_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        paths = []
        logs = results["reconciliation_log"]
        counts = defaultdict(int)
        for entry in logs:
            counts[entry['reconciliation_status']] += 1
        
        # 1. Status Distribution
        p1 = output_dir / "metric_b_status.png"
        if counts:
            plt.figure(figsize=(6, 6))
            plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
            plt.title("Status Distribution")
            plt.savefig(p1)
            plt.close()
            paths.append(p1)
            
        return paths
