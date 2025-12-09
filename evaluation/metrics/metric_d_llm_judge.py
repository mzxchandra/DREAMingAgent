from pathlib import Path
from typing import Dict, Any, List
import logging
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from .base_metric import BaseMetric
from ..judges.alcf_judge import ALCFJudge
from ..utils.plot_generator import plot_histogram, plot_bar_chart
from src.workflow import run_reconciliation

logger = logging.getLogger(__name__)

class MetricDLLMJudge(BaseMetric):
    """Metric D: LLM-as-a-Judge Evaluation (Real Data)."""

    def prepare_data(self) -> Dict[str, Any]:
        """Prepare real data subset for evaluation."""
        logger.info("Preparing real data for LLM Judge...")
        
        # 1. Source files (Real Data assumed in data/)
        # Using default paths or config if set
        network_path = Path("data/NetworkRegulatorGene.tsv")
        gene_prod_path = Path("data/GeneProductAllIdentifiersSet.tsv")
        expr_path = Path("data/E_coli_v4_Build_6_exps.tab")
        meta_path = Path("data/E_coli_v4_Build_6_meta.tab")
        
        if not network_path.exists():
            raise FileNotFoundError(f"Real data not found at {network_path}. Cannot run Metric D.")

        # 2. Filter Network for Target TFs (e.g., FNR, ArcA)
        # We process fewer TFs to verify "Biological Accuracy" deeply without 
        # running the whole network (which takes too long for eval loop).
        
        filtered_dir = self.config.data_dir / "metric_d_subset"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        subset_network_path = filtered_dir / "NetworkRegulatorGene_subset.tsv"
        
        target_tfs = set(self.config.judge_tfs) 
        # e.g., defined in config as ["FNR", "ArcA", "CRP"]
        # Wait, the config names might not match file names.
        # RDB names are usually mixed.
        # We need to filter by 'regulatorName' column (index 1).
        
        df = pd.read_csv(network_path, sep='\t', comment='#')
        # 1)regulatorId 2)regulatorName ...
        
        # Filter rows where regulatorName is in our list
        subset_df = df[df.iloc[:, 1].isin(target_tfs)]
        
        if len(subset_df) == 0:
            logger.warning(f"No edges found for TFs {target_tfs}. Using top 5 TFs instead.")
            top_tfs = df.iloc[:, 1].value_counts().head(5).index.tolist()
            subset_df = df[df.iloc[:, 1].isin(top_tfs)]
            
        # Save filtered network
        # We need to reconstruct the header
        header_lines = [line for line in open(network_path) if line.startswith('#') or line.startswith('1)')]
        
        with open(subset_network_path, 'w') as f:
            f.writelines(header_lines)
            subset_df.to_csv(f, sep='\t', header=False, index=False)
            
        logger.info(f"Created subset network with {len(subset_df)} edges for TFs: {subset_df.iloc[:, 1].unique()}")
        
        return {
            "network": subset_network_path,
            "gene_product": gene_prod_path,
            "expression": expr_path,
            "metadata": meta_path
        }

    def run_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow and judge results."""
        
        # 1. Run Workflow
        logger.info("Running reconciliation on real data subset...")
        # check if they exist, otherwise set to None
        safe_expr_path = str(expr_path) if expr_path.exists() else None
        safe_meta_path = str(meta_path) if meta_path.exists() else None

        if not safe_expr_path:
            logger.warning(f"M3D expression data not found at {expr_path}. Running without expression data.")
            
        final_state = run_reconciliation(
            regulondb_network_path=str(data["network"]), 
            regulondb_gene_product_path=str(data["gene_product"]),
            m3d_expression_path=safe_expr_path,
            m3d_metadata_path=safe_meta_path
            # Ensure LLM is enabled if possible for better explanations, 
            # but usually arguments control this. 
            # We assume the user creates environment variables or relies on default config.
        )
        
        logs = final_state.get("reconciliation_log", [])
        if not logs:
            logger.warning("No reconciliation logs found!")
            return {"judge_results": []}

        # 2. Sample Explanations (N=30)
        sample_size = min(len(logs), self.config.judge_sample_size)
        sample_logs = random.sample(logs, sample_size)
        
        # 3. Judge Logic
        judge = ALCFJudge(model_name=self.config.judge_model)
        judge_results = []
        
        logger.info(f"Judging {sample_size} explanations...")
        
        for i, entry in enumerate(sample_logs):
            edge_id = f"{entry['source_tf_name']}→{entry['target_gene_name']}"
            explanation = entry.get('notes', "No explanation provided.")
            
            # Skip if empty explanation
            if not explanation or explanation == "No explanation provided.":
                continue
                
            score = judge.evaluate_explanation(
                edge_id=edge_id,
                status=entry.get('reconciliation_status', 'Unknown'),
                lit_evidence=entry.get('regulondb_evidence', 'Unknown'),
                z_score=entry.get('m3d_z_score', 0.0),
                explanation=explanation
            )
            
            score['edge_id'] = edge_id
            judge_results.append(score)
            
            if (i+1) % 5 == 0:
                logger.info(f"Judged {i+1}/{sample_size}...")
                
        return {"judge_results": judge_results}

    def compute_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute aggregate judge scores."""
        scores = results["judge_results"]
        if not scores:
            return {}
            
        df = pd.DataFrame(scores)
        
        metrics = {
            "mean_bio_accuracy": float(df['biological_accuracy'].mean()),
            "mean_stat_reasoning": float(df['statistical_reasoning'].mean()),
            "mean_clarity": float(df['clarity'].mean()),
            "mean_overall": float(df['overall_quality'].mean()),
            "high_quality_rate": float((df['overall_quality'] >= 4).mean())
        }
        
        return metrics

    def generate_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        scores = results["judge_results"]
        if not scores:
            return []
            
        df = pd.DataFrame(scores)
        paths = []
        
        # 1. Histogram of Overall Scores
        hist_path = output_dir / "metric_d_score_dist.png"
        plot_histogram(
            df['overall_quality'].tolist(), 
            hist_path, 
            xlabel="Overall Quality (1-5)",
            title="LLM Judge Score Distribution",
            bins=5
        )
        paths.append(hist_path)
        
        # 2. Component Breakdown
        means = [
            df['biological_accuracy'].mean(),
            df['statistical_reasoning'].mean(),
            df['clarity'].mean(),
            df['overall_quality'].mean()
        ]
        cats = ["Bio Accuracy", "Stat Reasoning", "Clarity", "Overall"]
        
        bar_path = output_dir / "metric_d_components.png"
        plot_bar_chart(cats, means, bar_path, title="Average Component Scores", ylim=(1, 5))
        paths.append(bar_path)
        
        return paths
