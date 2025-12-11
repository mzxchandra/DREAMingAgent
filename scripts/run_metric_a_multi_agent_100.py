
import sys
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.config import EvaluationConfig
from evaluation.metrics.metric_a_sabotage import MetricASabotage
from src.utils.parsers import parse_tf_gene_network, parse_m3d_expression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def select_top_tfs(n: int = 100) -> list[str]:
    """Select top N TFs present in both Network and Expression data."""
    logger.info(f"Selecting top {n} TFs...")
    
    # Load Data Headers Only for speed / checking keys
    network_path = "data/NetworkRegulatorGene.tsv"
    expression_path = "data/E_coli_v4_Build_6_exps.tab"
    
    # 1. Get TFs from Graph
    graph, _ = parse_tf_gene_network(network_path)
    graph_tfs = {n for n, d in graph.nodes(data=True) if d.get('node_type') == 'TF'}
    
    # 2. Get TFs from Expression
    expr_df = parse_m3d_expression(expression_path)
    valid_tfs = []
    
    # 3. Intersection using Gene Product Mapping
    from src.utils.parsers import parse_gene_product_mapping
    name_to_bnumber, _, regulondb_to_bnumber = parse_gene_product_mapping("data/GeneProductAllIdentifiersSet.tsv")
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'TF':
            rdb_id = node
            # Try 1: Direct ID Map
            bnum = regulondb_to_bnumber.get(rdb_id)
            
            # Try 2: Name Map
            if not bnum:
                name = data.get('name', '').lower()
                bnum = name_to_bnumber.get(name)
                
            # Try 3: Check if ID is already b-number (unlikely in RDB but possible in mixed data)
            if not bnum and rdb_id.startswith('b') and rdb_id[1:].isdigit():
                bnum = rdb_id
            
            if bnum and bnum in expr_df.index:
                valid_tfs.append(bnum) # Use b-number
                
    # Sort and take top N
    valid_tfs = sorted(list(set(valid_tfs)))[:n]
    logger.info(f"Found {len(valid_tfs)} valid TFs. Returning top {n}.")
    return valid_tfs

def main():
    """
    Run Multi-Agent Metric A on 100 TFs.
    
    Usage:
        caffeinate -i ../aienv/bin/python scripts/run_metric_a_multi_agent_100.py
    """
    logger.info("Starting Scaled Metric A (100 TFs) - Agentic Workflow")
    
    # 1. Select TFs
    tfs = select_top_tfs(100)
    if not tfs:
        logger.error("No TFs found!")
        return

    # 2. Configure
    output_dir = Path("evaluation/outputs_scaled_100")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = EvaluationConfig(
        output_dir=output_dir,
        max_iterations=25, # 100 TFs / 5 per batch = 20 batches + buffer
        limit_tfs=tfs
    )
    
    # Override Sabotage Counts
    config.sabotage_n_false_positives = 50 # Total Global Injections
    
    # 3. Instantiate and Run
    metric = MetricASabotage(config)
    
    try:
        result = metric.execute(output_dir)
        
        # Reports
        from evaluation.utils.report_generator import generate_metric_report
        report_path = output_dir / f"report_{metric.name}.md"
        generate_metric_report(metric.name, result, report_path)
        
        logger.info(f"DONE. Report at {report_path}")
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
