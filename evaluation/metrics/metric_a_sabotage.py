from pathlib import Path
from typing import Dict, Any, List, Set
import logging
import pandas as pd
import matplotlib.pyplot as plt

from .base_metric import BaseMetric
from ..utils.graph_manipulation import (
    load_network_as_graph, 
    inject_false_edges, 
    delete_true_edges, 
    save_graph_to_regulondb_format
)
from ..utils.metrics_calculator import compute_classification_metrics
from ..utils.plot_generator import plot_confusion_matrix, plot_bar_chart
from src.workflow import run_reconciliation

logger = logging.getLogger(__name__)

class MetricASabotage(BaseMetric):
    """Metric A: Sabotage Test (False Positive Injection & Deletion)."""

    def prepare_data(self) -> Dict[str, Any]:
        """Generate sabotaged data based on REAL RegulonDB network."""
        logger.info("Preparing sabotage data from Real Network...")
        
        # 1. Defined paths to Real Data (using default locations or config)
        # Assuming data is in project root/data or similar. 
        # For now, we expect them to be available where main.py usually finds them.
        # We'll use the defaults from LoaderConfig/main.py if not specified in config.
        # Hardcoding standard paths for this context or expecting them in config Data Dir?
        # Let's assume standard locations relative to project root.
        project_root = Path(".").resolve()
        data_dir = project_root / "data"
        if not data_dir.exists():
             data_dir = project_root # Fallback to root if data/ doesn't exist
             
        network_path = data_dir / "NetworkRegulatorGene.tsv"
        gene_product_path = data_dir / "GeneProductAllIdentifiersSet.tsv"
        expression_path = data_dir / "E_coli_v4_Build_6_exps.tab"
        metadata_path = data_dir / "E_coli_v4_Build_6_meta.tab"
        
        if not network_path.exists():
            raise FileNotFoundError(f"Real network file not found at {network_path}")

        # 2. Load Real Graph
        G = load_network_as_graph(network_path)
        initial_edge_count = G.number_of_edges()
        logger.info(f"Loaded real network with {initial_edge_count} edges.")
        
        # 3. Inject False Positives (Sabotage)
        # We inject edges that do NOT exist in RegulonDB (presumed false)
        G_injected, injected_ids = inject_false_edges(
            G, 
            n_edges=self.config.sabotage_n_false_positives,
            evidence_level='W'
        )
        
        # 4. Save Sabotaged Network
        sabotage_dir = self.config.data_dir / "sabotage_real"
        sabotage_dir.mkdir(parents=True, exist_ok=True)
        sabotaged_network_path = sabotage_dir / "network_sabotaged.txt"
        
        save_graph_to_regulondb_format(G_injected, sabotaged_network_path)
        
        # Return paths (using REAL expression/metadata, but SABOTAGED network)
        return {
            "network_path": sabotaged_network_path,
            "gene_product_path": gene_product_path,
            "expression_path": expression_path,
            "metadata_path": metadata_path,
            "injected_ids": injected_ids,
            "deleted_ids": set() # We are skipping deletion for now as recovering derived edges is hard without context
        }

    def run_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow on sabotaged network."""
        self.meta = data # store for scoring
        
        logger.info("Running reconciliation on sabotaged network...")
        
        final_state = run_reconciliation(
            regulondb_network_path=str(data["network_path"]),
            regulondb_gene_product_path=str(data["gene_product_path"]),
            m3d_expression_path=str(data["expression_path"]),
            m3d_metadata_path=str(data["metadata_path"]),
            max_iterations=self.config.max_iterations,
            target_tfs=self.config.limit_tfs if self.config.limit_tfs else None
        )
        
        return {
            "reconciliation_log": final_state.get("reconciliation_log", []),
            "novel_hypotheses": final_state.get("novel_hypotheses", []),
            "false_positive_candidates": final_state.get("false_positive_candidates", [])
        }

    def compute_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Score FP detection and Edge Recovery."""
        injected = self.meta["injected_ids"]
        deleted = self.meta["deleted_ids"]
        
        # 1. FP Detection Analysis
        # We want the system to flag 'injected' edges as "ProbableFalsePositive"
        
        detected_fp = set()
        # Look in reconciliation log
        for entry in results["reconciliation_log"]:
            edge_id = f"{entry['source_tf_name']}→{entry['target_gene_name']}"
            if entry['reconciliation_status'] == "ProbableFalsePositive":
                detected_fp.add(edge_id)
                
        # Also check false_positive_candidates list (redundant but safe)
        for cand in results["false_positive_candidates"]:
            # cand format might differ, assuming it matches edge_id construction
            # usually it's a dict
            if isinstance(cand, dict):
                edge_id = f"{cand.get('source_tf_name')}→{cand.get('target_gene_name')}"
                detected_fp.add(edge_id)
        
        tp_fp = len(injected.intersection(detected_fp)) # Correctly flagged injected edges
        fn_fp = len(injected - detected_fp) # Missed injected edges (marked valid)
        
        # We can also calculate False Positives for the detector (flagged a true edge as FP)
        remaining_truth = self.meta["ground_truth_remaining"]
        fp_detector = len(detected_fp.intersection(remaining_truth))
        
        # 2. Edge Recovery Analysis (Novel)
        # We want the system to find 'deleted' edges as "NovelHypothesis"
        # Note: In the current workflow, Novel Hypotheses come from high CLR scores that are NOT in the input network.
        # Since we deleted them from the input, they count as "not in input" but "high CLR".
        
        recovered = set()
        for novel in results["novel_hypotheses"]:
             # novel is likely a dict or object
             # In workflow, novel_hypotheses is a list of dicts: source, target, etc.
             tf = novel.get('source_tf_name', novel.get('source'))
             target = novel.get('target_gene_name', novel.get('target'))
             edge_id = f"{tf}→{target}"
             recovered.add(edge_id)
             
        tp_rec = len(deleted.intersection(recovered))
        fn_rec = len(deleted - recovered)
        
        return {
            "fp_detection_rate": tp_fp / len(injected) if injected else 0.0,
            "fp_precision": tp_fp / (tp_fp + fp_detector) if (tp_fp + fp_detector) > 0 else 0.0,
            "recovery_rate": tp_rec / len(deleted) if deleted else 0.0,
            "tp_fp": tp_fp,
            "fn_fp": fn_fp,
            "fp_detector": fp_detector
        }

    def generate_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        scores = self.compute_scores(results)
        paths = []
        
        # 1. FP Detection Confusion Matrix (Subset: Only Injected vs Real)
        # We simplify the universe to (Injected U Flagged)
        # True Class: Injected (Positive), Real (Negative)
        # Predicted: Flagged FP (Positive), Validated (Negative)
        
        tp = scores["tp_fp"] # Injected AND Flagged
        fn = scores["fn_fp"] # Injected BUT Validated
        fp = scores["fp_detector"] # Real BUT Flagged
        # TN = Real AND Validated?
        # Use total remaining truth count
        tn = len(self.meta["ground_truth_remaining"]) - fp
        
        cm_path = output_dir / "metric_a_fp_confusion_matrix.png"
        plot_confusion_matrix(tp, fp, tn, fn, cm_path, title="FP Detection Performance")
        paths.append(cm_path)
        
        # 2. Rates Bar Chart
        rates_path = output_dir / "metric_a_rates.png"
        plot_bar_chart(
            ["FP Detection Rate", "Edge Recovery Rate"],
            [scores["fp_detection_rate"], scores["recovery_rate"]],
            rates_path,
            title="Sabotage Test Performance",
            ylim=(0, 1.0)
        )
        paths.append(rates_path)
        
        return paths
