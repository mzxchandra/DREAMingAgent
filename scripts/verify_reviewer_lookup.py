
import sys
import logging
from loguru import logger
import networkx as nx

# Configure logger first
logger.remove()
logger.add(sys.stdout, level="INFO")

from src.nodes.reviewer_agent import prepare_subgraph_data

def test_fuzzy_lookup():
    print("=== Testing Reviewer Agent Fuzzy Lookup ===")
    
    # 1. Mock Data
    tf_bnumber = "b4062"
    target_bnumber = "b1922"
    
    # Mock Stats: Use the exact key format seen in verify_analysis_math (flia_b1922_15)
    tf_stats = {
        "flia_b1922_15": {
            "z_score": 2.05,
            "mi": 0.35,
            "p_value": 0.02
        },
        "other_b1111_1": {
            "z_score": 0.1,
            "mi": 0.01
        }
    }
    
    # Mock Literature Graph
    lit_graph = nx.DiGraph()
    lit_graph.add_edge(tf_bnumber, target_bnumber, effect="+", evidence_type="strong")
    
    # Mock Bnumber map
    bnumber_to_name = {
        "b4062": "soxS",
        "b1922": "fliA"
    }
    
    print(f"Mock Stats Keys: {list(tf_stats.keys())}")
    print(f"Target B-number: {target_bnumber}")
    
    # 2. Construct State
    mock_state = {
        "literature_graph": lit_graph,
        "analysis_results": {tf_bnumber: tf_stats},
        "metadata": None,
        "bnumber_to_gene_name": bnumber_to_name
    }

    # 3. Run Function
    # Note: prepare_subgraph_data(state, tf_bnumber)
    edges = prepare_subgraph_data(mock_state, tf_bnumber)
    
    # 3. Verify
    print(f"\nResult Edges: {len(edges)}")
    
    matched = False
    for edge in edges:
        if edge["target"] == target_bnumber:
            stats = edge.get("statistics", {})
            print(f"Target {target_bnumber} Stats: {stats}")
            
            # Check if stats match mocked values
            if stats.get("z_score") == 2.05:
                print("SUCCESS: Exact stats retrieved via fuzzy match!")
                matched = True
            else:
                print(f"FAILURE: Stats do not match. Got {stats}")
                
    if not matched:
        print("FAILURE: Target not found or stats zero.")

if __name__ == "__main__":
    test_fuzzy_lookup()
