from pathlib import Path
from typing import Dict, Any, List
import json

def generate_metric_report(metric_name: str, results: Dict[str, Any], output_path: Path):
    """Generate a detailed report for a single metric."""
    scores = results["scores"]
    plot_paths = results["plot_paths"]
    
    lines = [
        f"# Metric Report: {metric_name}",
        "",
        "## Scores",
        "| Metric | Value |",
        "|--------|-------|"
    ]
    
    for k, v in scores.items():
        val = f"{v:.4f}" if isinstance(v, float) else v
        lines.append(f"| {k} | {val} |")
        
    lines.append("")
    lines.append("## Plots")
    for p in plot_paths:
        p_obj = Path(p)
        # Use relative path for markdown
        rel_path = p_obj.name # Assuming plots are in same folder or subfolder
        # Actually usually plots are in same folder as report or subfolder. 
        # If output_path is `outputs/reports/metric.md`, and plots are `outputs/plot.png`
        # We need relative path logic. 
        # For simplicity, let's assume plots are in `../` relative to report directory if report is in `reports/`
        # But `runner.py` usually puts everything in `outputs/`.
        # Let's just use the filename if they are in the same dir.
        lines.append(f"![{rel_path}]({rel_path})")
        lines.append("")
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

def generate_combined_report(all_results: List[Dict[str, Any]], output_path: Path):
    """Generate a summary report for all metrics."""
    lines = [
        "# DREAMing Agent Evaluation Summary",
        "",
        "## Overview",
        f"Evaluated {len(all_results)} metrics.",
        "",
    ]
    
    for res in all_results:
        name = res["metric"]
        scores = res["scores"]
        
        lines.append(f"### {name}")
        for k, v in scores.items():
             if "rate" in k or "accuracy" in k or "precision" in k or "recall" in k or "f1" in k:
                 val = f"{v:.2%}" if isinstance(v, float) else v
             else:
                 val = f"{v:.3f}" if isinstance(v, float) else v
             lines.append(f"- **{k}**: {val}")
        lines.append("")
        
        # Embed first plot if available
        if res["plot_paths"]:
            p_name = Path(res["plot_paths"][0]).name
            lines.append(f"![{p_name}]({p_name})")
            lines.append("")
            
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
