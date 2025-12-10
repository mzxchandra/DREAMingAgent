import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from loguru import logger

# Setup paths and env
load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.nodes.reviewer_agent import reviewer_agent_node
from src.state import AgentState
from src.utils.vector_store import get_vector_store
import networkx as nx

def main():
    print("=== Verifying Reviewer Agent Literature Integration ===")
    
    # 1. Setup Mock State
    tf_id = "b4451" # ryhb
    target_id = "b1000" # arbitrary target (mock)
    
    # Create literature graph with ONE edge that exists
    lit_graph = nx.DiGraph()
    lit_graph.add_edge(tf_id, target_id, effect="repression", evidence_type="strong")
    
    state = {
        "current_batch_tfs": [tf_id],
        "literature_graph": lit_graph,
        "analysis_results": {
            tf_id: {
                target_id: {"clr_zscore": 0.0, "mi": 0.0} # No data support
            }
        },
        "metadata": None,
        "bnumber_to_gene_name": {
            "b4451": "ryhb",
            "b1000": "mock_target"
        },
        "reconciliation_log": []
    }
    
    # 2. Run Reviewer Node
    print(f"Running Reviewer Agent for {tf_id} (ryhb)...")
    try:
        result = reviewer_agent_node(state)
        
        # 3. Inspect Results
        log = result.get("reconciliation_log", [])
        print(f"\nGenerated {len(log)} log entries.")
        
        for entry in log:
            print(f"\n[Decision for {entry['target_gene_name']}]")
            print(f"Status: {entry['reconciliation_status']}")
            print(f"Notes: {entry['notes']}")
            
            # Check for context usage
            if "context" in entry['notes'].lower() or "literature" in entry['notes'].lower():
                print("SUCCESS: Literature context referenced in notes.")
            else:
                print("WARNING: contextual reference missing?")
                
    except Exception as e:
        print(f"Reviewer Agent Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
