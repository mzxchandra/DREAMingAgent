import json
import re
import sys
from pathlib import Path

# Fix import path to include src
sys.path.append(str(Path(".").resolve()))
from src.utils.parsers import parse_gene_product_mapping

def reconstruct_json():
    input_path = Path("evaluation/outputs_scaled_100/metric_a_sabotage_details.md")
    output_path = Path("evaluation/outputs_scaled_100/reconciliation_results.json")
    gene_product_path = "data/GeneProductAllIdentifiersSet.tsv"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # Load Mapping (Name -> B-Number)
    print("Loading gene mapping...")
    name_to_bnumber, _, _ = parse_gene_product_mapping(gene_product_path)

    print(f"Reading {input_path}...")
    with open(input_path, "r") as f:
        lines = f.readlines()

    reconciliation_log = []
    
    # Regex to capture table row content
    # | Edge ID | Previous Value | System Prediction | Z-Score | Outcome |
    row_pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")

    for line in lines:
        match = row_pattern.search(line)
        if match:
            edge_id = match.group(1).strip()
            prev_value = match.group(2).strip()
            prediction = match.group(3).strip()
            z_score_str = match.group(4).strip()
            outcome = match.group(5).strip()
            
            # Skip Headers
            if edge_id == "Edge ID" or "---" in edge_id:
                continue

            # Parse Edge ID
            if "→" in edge_id:
                source_name, target_name = edge_id.split("→")
            else:
                source_name, target_name = edge_id, "Unknown"

            # Parse Z-Score
            try:
                z_score = float(z_score_str)
            except ValueError:
                z_score = 0.0

            # Lookup IDs
            source_tf = name_to_bnumber.get(source_name.lower(), "Unknown")
            target_gene = name_to_bnumber.get(target_name.lower(), "Unknown")

            # Construct JSON Entry matching STALE FILE schema
            entry = {
                "source_tf": source_tf, # ID
                "target_gene": target_gene, # ID
                "source_tf_name": source_name,
                "target_gene_name": target_name,
                "regulondb_evidence": "SabotageTest", # Context
                "regulondb_effect": "?",
                "m3d_z_score": z_score,
                "m3d_mi": 0.0, # Not preserved in markdown report
                "reconciliation_status": prediction,
                "context_tags": ["Sabotage"],
                "notes": f"[Reconstructed] Previous State: {prev_value}. Outcome: {outcome}. Z-Score: {z_score}. (Derived from detailed report)"
            }
            reconciliation_log.append(entry)

    # Filter for derived lists (mock logic for structure completeness)
    novel_hypotheses = [e for e in reconciliation_log if e["reconciliation_status"] == "NovelHypothesis"]
    false_positives = [e for e in reconciliation_log if e["reconciliation_status"] == "ProbableFalsePos"]

    final_output = {
        "reconciliation_log": reconciliation_log,
        "novel_hypotheses": novel_hypotheses,
        "false_positive_candidates": false_positives,
        "summary": {
            "total_edges": len(reconciliation_log),
            "novel_count": len(novel_hypotheses),
            "false_positive_count": len(false_positives)
        }
    }

    print(f"Extracted {len(reconciliation_log)} entries.")
    
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Saved strictly formatted JSON to {output_path}")

if __name__ == "__main__":
    reconstruct_json()
