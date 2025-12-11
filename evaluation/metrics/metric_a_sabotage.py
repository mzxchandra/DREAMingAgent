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
        # If running on multiple TFs, we want Global Sabotage RESTRICTED to those TFs
        # If running on single TF, we focus sabotage around that TF
        if self.config.limit_tfs:
             # Fix for ID Mismatch: Graph uses Names (araC), limit_tfs uses IDs (b0064)
             # We need to map IDs -> Names to ensure focus_node filtering works
             from src.utils.parsers import parse_gene_product_mapping
             _, bnumber_to_name, _ = parse_gene_product_mapping("data/GeneProductAllIdentifiersSet.tsv")
             
             focus_tf = []
             for tf_id in self.config.limit_tfs:
                 name = bnumber_to_name.get(tf_id)
                 if name:
                     focus_tf.append(name.lower())
                 else:
                     focus_tf.append(tf_id.lower()) # Fallback to ID
        else:
            focus_tf = None # Truly Global (any node in graph)
        
        G_injected, injected_ids = inject_false_edges(
            G, 
            n_edges=self.config.sabotage_n_false_positives,
            evidence_level='W',
            focus_node=focus_tf
        )
        
        # 3.1 Delete True Edges (Sabotage - Recovery Test)
        # We delete edges that DO exist in RegulonDB (presumed true) 
        # to see if the system recovers them as "Novel" or "Validated"
        G_sabotaged, deleted_ids = delete_true_edges(
            G_injected, # Use the graph that already has injections
            n_edges=self.config.sabotage_n_false_positives, # Recycle text N for deletion count for now
            evidence_level='S', # Delete STRONG edges so we know they should be there
            focus_node=focus_tf
        )
        
        logger.info(f"Sabotage Summary for {focus_tf if focus_tf else 'Global'}:")
        logger.info(f"  Injected {len(injected_ids)} False Positives: {injected_ids}")
        logger.info(f"  Deleted {len(deleted_ids)} True Edges: {deleted_ids}")

        # 4. Save Sabotaged Network
        sabotage_dir = self.config.data_dir / "sabotage_real"
        sabotage_dir.mkdir(parents=True, exist_ok=True)
        sabotaged_network_path = sabotage_dir / "network_sabotaged.txt"
        
        save_graph_to_regulondb_format(G_sabotaged, sabotaged_network_path)
        
        # Return paths (using REAL expression/metadata, but SABOTAGED network)
        return {
            "network_path": sabotaged_network_path,
            "gene_product_path": gene_product_path,
            "expression_path": expression_path,
            "metadata_path": metadata_path,
            "injected_ids": injected_ids,
            "deleted_ids": deleted_ids, # Now we have deleted IDs to track
            "ground_truth_remaining": set(G.edges[u, v]['edge_id'] for u, v in G.edges()) - set(deleted_ids)
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
        injected = set(self.meta["injected_ids"])
        deleted = set(self.meta["deleted_ids"])
        
        # 1. FP Detection Analysis
        # We want the system to flag 'injected' edges as "ProbableFalsePositive"
        
        detected_fp = set()
        # Look in reconciliation log
        for entry in results["reconciliation_log"]:
            edge_id = f"{entry['source_tf_name']}→{entry['target_gene_name']}"
            if entry['reconciliation_status'] == "ProbableFalsePos":
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

    def _generate_details_report(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """Generate a detailed markdown report of the sabotage targets."""
        report_path = output_dir / "metric_a_sabotage_details.md"
        
        injected = self.meta.get("injected_ids", [])
        deleted = self.meta.get("deleted_ids", [])
        
        # Helper to find edge in log
        log_map = {}
        for entry in results.get("reconciliation_log", []):
            eid = f"{entry['source_tf_name']}→{entry['target_gene_name']}"
            log_map[eid] = {
                "status": entry.get('reconciliation_status', 'Unknown'),
                "z_score": entry.get('m3d_z_score', 0.0)
            }
            
        lines = ["# Metric A: Sabotage Detail Report\n"]
        lines.append(f"**Focus TF:** {self.config.limit_tfs[0] if self.config.limit_tfs else 'Global'}\n")
        
        lines.append("## 1. Injected False Positives (Goal: Detect as ProbableFalsePos)\n")
        lines.append("| Edge ID | Previous Value | System Prediction | Z-Score | Outcome |")
        lines.append("|---|---|---|---|---|")
        
        for eid in injected:
            data = log_map.get(eid, {"status": "Not Reviewed", "z_score": "N/A"})
            pred = data["status"]
            z = data["z_score"]
            z_str = f"{z:.2f}" if isinstance(z, (int, float)) else str(z)
            
            # Outcome logic
            if pred == "ProbableFalsePos":
                outcome = "✅ DETECTED"
            elif pred == "Not Reviewed":
                outcome = "⚠️ SKIPPED"
            else:
                outcome = "❌ FAILED (Accepted)"
            
            lines.append(f"| {eid} | Non-Existent | {pred} | {z_str} | {outcome} |")
            
        lines.append("\n## 2. Deleted True Edges (Goal: Recover as NovelHypothesis)\n")
        lines.append("| Edge ID | Previous Value | System Prediction | Z-Score | Outcome |")
        lines.append("|---|---|---|---|---|")
        
        for eid in deleted:
            data = log_map.get(eid, {"status": "Not Found (Low Signal)", "z_score": "N/A"})
            pred = data["status"]
            z = data["z_score"]
            z_str = f"{z:.2f}" if isinstance(z, (int, float)) else str(z)
            
            # Outcome logic
            if pred == "NovelHypothesis":
                outcome = "✅ RECOVERED"
            elif pred == "Not Found (Low Signal)":
                outcome = "❌ MISSED (Low Stats)"
            else:
                outcome = f"⚠️ {pred}" 
            
            lines.append(f"| {eid} | Existing (True) | {pred} | {z_str} | {outcome} |")
            
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
            
        return report_path

    def generate_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        scores = self.compute_scores(results)
        paths = []
        
        # 0. Generate Detailed Report
        self._generate_details_report(results, output_dir)
        
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
