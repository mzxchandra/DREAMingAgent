
import os
import sys
import argparse
import pandas as pd
import logging
from tqdm import tqdm
from typing import Set, Tuple, Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.time_travel_utils import (
    load_tsv_map, Parser2016, Parser2024, DescriptionFetcher, 
    compute_targeted_clr_row_z, BIG_5_MAP, BIG_5_BNUMS
)
from src.llm_config import create_argonne_llm
from langchain_core.prompts import ChatPromptTemplate
from src.nodes.reconciler import reconciler_node
from src.state import AgentState
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Big 5 TFs (b-numbers)
# FNR: b1334
# ArcA: b4401
# CRP: b3357
# Fur: b0683
# SoxS: b4062
BIG_5_MAP = {
    "FNR": "b1334",
    "ArcA": "b4401",
    "CRP": "b3357",
    "Fur": "b0683",
    "SoxS": "b4062"
}
REVERSE_BIG_5 = {v: k for k, v in BIG_5_MAP.items()}

def run_system_b(
    llm, 
    tf_name: str, 
    gene_desc: str, 
    z_score: float
) -> str:
    """
    Run System B (Blind Baseline).
    Prompt with TF Name, Gene Description, and CLR Score.
    """
    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert biologist studying E. coli gene regulation.\n"
        "Task: Determine if Transcription Factor '{tf_name}' regulates the gene described as '{gene_desc}'.\n"
        "Evidence: Context Likelihood of Relatedness (CLR) Z-score is {z_score:.2f} (Z > 2.0 suggests interaction).\n\n"
        "Answer strictly with YES or NO."
    )
    
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "tf_name": tf_name,
            "gene_desc": gene_desc,
            "z_score": z_score
        })
        content = response.content.strip().upper()
        if "YES" in content:
            return "YES"
        return "NO"
    except Exception as e:
        logger.error(f"System B LLM Error: {e}")
        return "ERROR"

def main():
    parser = argparse.ArgumentParser(description="Time Travel Evaluation Harness")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--candidates", type=int, default=50, help="Max candidates per TF")
    parser.add_argument("--dry-run", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    # Paths
    regulon_path = os.path.join(args.data_dir, "regulon9.0")
    future_net_path = os.path.join(args.data_dir, "NetworkRegulatorGene.tsv")
    future_id_map = os.path.join(args.data_dir, "GeneProductAllIdentifiersSet.tsv")
    data_dir = args.data_dir # Define data_dir for use in the new snippet
    regulon_path = os.path.join(data_dir, "regulon9.0")
    future_net_path = os.path.join(data_dir, "NetworkRegulatorGene.tsv")
    future_id_map = os.path.join(data_dir, "GeneProductAllIdentifiersSet.tsv")
    expr_file = os.path.join(data_dir, "E_coli_v4_Build_6_exps.tab")
    product_file = os.path.join(regulon_path, "product.txt")
    synonym_file = os.path.join(regulon_path, "object_synonym.txt")

    # 1. Load Ground Truths (Past & Future)
    logger.info("Loading Knowledge Bases (Epoch A & B)...")
    
    # Epoch A: 2016 (Past)
    # Filter for Big 5 TFs (using b-numbers)
    # BIG_5_MAP keys are names ('FNR'), values are b-numbers ('b1334'). Parsers return b-numbers.
    big_5_bnums = BIG_5_BNUMS
    
    past_interactions = Parser2016(
        os.path.join(data_dir, "regulon9.0"), 
        big_5_tfs=big_5_bnums
    )
    logger.info(f"Past Edges (Big 5): {len(past_interactions)}")
    
    # Epoch B: 2024 (Future)
    future_interactions = Parser2024(
        os.path.join(data_dir, "NetworkRegulatorGene.tsv"),
        os.path.join(data_dir, "GeneProductAllIdentifiersSet.tsv"),
        big_5_tfs=big_5_bnums
    )
    logger.info(f"Future Edges (Big 5): {len(future_interactions)}")
    
    # Discovery Set: Edges in Future but NOT in Past
    discovery_set = future_interactions - past_interactions
    discovery_set = future_interactions - past_interactions
    logger.info(f"Target Discovery Set: {len(discovery_set)}")

    # Build Literature Graph (NetworkX) for Agent System A
    literature_graph = nx.DiGraph()
    for src, tgt in past_interactions:
        literature_graph.add_edge(src, tgt, evidence_type="Confirmed", effect="?")
    logger.info(f"Built Literature Graph with {literature_graph.number_of_edges()} edges.")

    # 2. Compute Candidates (Epoch C evidence)
    logger.info("Loading Expression Data & Computing CLR...")
    expr_df = pd.read_csv(expr_file, sep='\t', index_col=0)
    
    # Standardize Index: Extract b-number from "gene_b1234_suffix"
    import re
    def extract_bnum(s):
        match = re.search(r'(b\d{4})', str(s))
        return match.group(1) if match else None
        
    # Create new index map
    new_index = []
    valid_rows = []
    for idx in expr_df.index:
        bnum = extract_bnum(idx)
        if bnum:
            new_index.append(bnum)
            valid_rows.append(True)
        else:
            valid_rows.append(False)
            
    # Filter and reindex
    expr_df = expr_df[valid_rows]
    expr_df.index = new_index
    # Handle duplicates (keep first?)
    expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
    
    logger.info(f"Matrix standardized to {len(expr_df)} genes (b-numbers).")
    clr_df = compute_targeted_clr_row_z(expr_df, list(big_5_bnums))
    
    # Filter Candidates
    # 1. Z-score >= 2.0 (Moderate)
    # 2. Not in Past Edges (Novelty)
    # 3. Top N per TF
    
    candidates = []
    
    logger.info("Filtering Candidates...")
    for tf_bnum in big_5_bnums:
        tf_data = clr_df[clr_df['TF'] == tf_bnum].copy()
        
        # Filter Z
        tf_data = tf_data[tf_data['score'] >= 2.0]
        
        # Filter Known
        # (TF, Target) pair
        known_targets = {target for (src, target) in past_interactions if src == tf_bnum}
        tf_data = tf_data[~tf_data['Target'].isin(known_targets)]
        
        # Sort & Limit
        tf_data = tf_data.sort_values('score', ascending=False).head(args.candidates)
        
        candidates.append(tf_data)
        
    candidates_df = pd.concat(candidates)
    logger.info(f"Generated {len(candidates_df)} candidates for evaluation.")
    
    # 4. Prepare System A (Anti-Gravity Agent)
    logger.info("Initializing System A (Reconciler Agent)...")
    
    # Format Analysis Results for Agent: {tf: {target: {z_score, mi}}}
    analysis_results_dict = {}
    for tf in big_5_bnums:
        tf_data = clr_df[clr_df['TF'] == tf]
        tf_results = {}
        for _, row in tf_data.iterrows():
            tf_results[row['Target']] = {
                "z_score": row['score'],
                "mi": row['mi'] if 'mi' in row else 0.0 # Handle if MI not present
            }
        analysis_results_dict[tf] = tf_results

    # Construct Agent State
    state = AgentState(
        literature_graph=literature_graph,
        gene_name_to_bnumber={}, # Not needed as we use b-nums internally
        bnumber_to_gene_name=REVERSE_BIG_5, # Partial map for logging
        regulondb_to_bnumber={},
        expression_matrix=expr_df,
        metadata=pd.DataFrame(), # Empty metadata for now
        tf_queue=list(big_5_bnums),
        current_batch_tfs=list(big_5_bnums),
        active_sample_indices=[],
        current_context="E. coli K-12 Aerobic Glucose M9", # Standard context
        analysis_results=analysis_results_dict,
        reconciliation_log=[],
        novel_hypotheses=[],
        false_positive_candidates=[],
        iteration_count=0,
        max_iterations=1,
        errors=[],
        status="processing"
    )
    
    # Run Agent (Reconciler Node)
    try:
        agent_output = reconciler_node(state)
        reconciliation_log = agent_output.get("reconciliation_log", [])
        logger.info(f"System A complete. Log size: {len(reconciliation_log)}")
    except Exception as e:
        logger.error(f"System A Failed: {e}")
        reconciliation_log = []

    # Map Agent Predictions (TF, Target) -> Status
    agent_predictions = {}
    for entry in reconciliation_log:
        key = (entry['source_tf'], entry['target_gene'])
        agent_predictions[key] = entry['reconciliation_status']

    if args.dry_run:
        logger.info("DRY RUN: Limiting to top 5 candidates total.")
        candidates_df = candidates_df.head(5)

    # 5. Initialize System B (Blind Baseline)
    llm = create_argonne_llm()
    desc_fetcher = DescriptionFetcher(product_file, synonym_file)
    
    results = []

    # 6. Evaluation Loop
    logger.info("Starting Evaluation Loop...")
    for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df)):
        tf_bnum = row['TF']
        target_bnum = row['Target']
        z_score = row['score']
        
        tf_name = REVERSE_BIG_5.get(tf_bnum, tf_bnum)
        target_desc = desc_fetcher.get(target_bnum, "Unknown Gene Product")
        
        # System B (Blind)
        sys_b_pred = run_system_b(llm, tf_name, target_desc, z_score)
        
        # System A (Agentic)
        # Check if agent marked it as "NovelHypothesis" (Discovery) or "Validated" (if recovering)
        # Since these are candidates (high Z, not in past), we hope for "NovelHypothesis"
        agent_status = agent_predictions.get((tf_bnum, target_bnum), "Pending")
        sys_a_pred = agent_status
        
        # Ground Truth Check
        is_hit = (tf_bnum, target_bnum) in discovery_set
        
        results.append({
            "TF": tf_name,
            "TF_ID": tf_bnum,
            "Target_ID": target_bnum,
            "Target_Desc": target_desc,
            "Z_Score": z_score,
            "System_B_Pred": sys_b_pred,
            "System_A_Pred": sys_a_pred,
            "Ground_Truth_Discovery": is_hit
        })
        
    # 7. Save Results
    res_df = pd.DataFrame(results)
    print("\n=== Evaluation Results ===")
    print(res_df.head())
    
    out_file = "time_travel_results.csv"
    res_df.to_csv(out_file, index=False)
    logger.info(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
