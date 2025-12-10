import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Setup paths and env
load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def main():
    print("=== Verifying Analysis Agent Math Engine ===")
    
    # Deferred Imports
    import pandas as pd
    import numpy as np
    from src.nodes.analysis_agent import _analyze_single_tf
    
    # Check imports First
    try:
        from scipy import stats
        from sklearn.feature_selection import mutual_info_regression
        print("SUCCESS: Scipy and Sklearn imported successfully.")
    except ImportError as e:
        print(f"FAILURE: Could not import dependencies: {e}")
        return
    
    # 1. Load Data
    print("Loading M3D Expression Data...")
    try:
        df = pd.read_csv("data/E_coli_v4_Build_6_exps.tab", sep="\t", index_col=0)
        # Clean index
        df.index = df.index.astype(str).str.strip().str.lower()
        print(f"Loaded matrix: {df.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Setup TF and Target
    tf_id = "b4451" # ryhb
    tf_name = "ryhb"
    
    if tf_id not in df.index:
        print(f"TF {tf_id} not found in index. Searching...")
        matches = [x for x in df.index if "ryhb" in x]
        print(f"Possible matches: {matches}")
        if matches:
            tf_id = matches[0]
            print(f"Using {tf_id}")
        else:
            print("Cannot proceed without TF data.")
            return

    # 3. Running Analysis Logic
    print(f"\nRunning CLR Analysis for {tf_id}...")
    
    # Mock literature graph (not needed for math, but required by signature)
    import networkx as nx
    lit_graph = nx.DiGraph()
    
    # Run analysis on a subset of highly variable genes for speed
    # Calculate variances
    variances = df.var(axis=1)
    top_genes = variances.nlargest(500).index.tolist()
    if tf_id not in top_genes:
        top_genes.append(tf_id)
        
    data_slice = df.loc[top_genes]
    print(f"Sliced data to top 500 variable genes for quick verification.")
    
    results = _analyze_single_tf(
        tf_id=tf_id,
        lookup_id=tf_id,
        data_slice=data_slice,
        literature_graph=lit_graph,
        n_samples=len(df.columns)
    )
    
    # 4. Print Results
    print("\n[Statistical Results Sample]")
    if "error" in results:
        print(f"Analysis Error: {results}")
    elif "status" in results and results["status"] == "TF_LOW_EXPRESSION":
        print(f"TF Low Expression: {results}")
    else:
        # Sort by Z-score
        # Debug: Print first key to see structure
        first_key = list(results.keys())[0]
        print(f"Debug: First result key '{first_key}': {results[first_key]}")
        
        # Sort by Z-score (handle nested structure if needed)
        try:
             sorted_genes = sorted(results.items(), key=lambda x: x[1].get('z_score', 0), reverse=True)
        except Exception as e:
             print(f"Sort failed: {e}")
             return
        print(f"Total Targets Analyzed: {len(results)}")
        print("\nTop 5 Significant Interactions (CLR Z-Score > 2.0):")
        
        count = 0
        for gene, stats in sorted_genes:
            if stats.get('z_score', 0) > 2.0:
                print(f"  Target: {gene:<15} | CLR Z: {stats.get('z_score', 0):.4f} | MI: {stats.get('mi', 0):.4f}")
                count += 1
                if count >= 5: break
        
        if count == 0:
            print("  No targets crossed Z=2.0 threshold in this subset.")
            # Print top 3 anyway
            print("\nTop 3 raw scores:")
            for gene, stats in sorted_genes[:3]:
                 print(f"  Target: {gene:<15} | CLR Z: {stats.get('z_score', 0):.4f} | MI: {stats.get('mi', 0):.4f}")
                 
        print("\nSUCCESS: Calculated CLR Z-scores and Mutual Information.")

if __name__ == "__main__":
    main()
