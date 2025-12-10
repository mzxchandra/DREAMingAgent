
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.pathway_assessor.assessor import PathwayAssessor
from loguru import logger

def find_correlated_pair():
    assessor = PathwayAssessor()
    df = assessor.m3d_data
    
    genes = list(df.index)
    # Check a subset to avoid N^2
    np.random.seed(42)
    
    print("Searching for pairs...")
    
    high_corr_pair = None
    low_corr_pair = None
    
    # Try more random pairs
    for _ in range(2000):
        g1 = np.random.choice(genes)
        g2 = np.random.choice(genes)
        if g1 == g2: continue
        
        vec_a = df.loc[g1]
        vec_b = df.loc[g2]
        
        valid = ~np.isnan(vec_a) & ~np.isnan(vec_b)
        if valid.sum() > 20:
            corr = np.corrcoef(vec_a[valid], vec_b[valid])[0, 1]
            
            if high_corr_pair is None and corr > 0.85:
                n1 = str(g1).split('_')[0]
                n2 = str(g2).split('_')[0]
                print(f"High Corr Pair: {n1} - {n2} (Corr: {corr:.4f})")
                high_corr_pair = (n1, n2)
                
            if low_corr_pair is None and abs(corr) < 0.05:
                n1 = str(g1).split('_')[0]
                n2 = str(g2).split('_')[0]
                print(f"Low Corr Pair: {n1} - {n2} (Corr: {corr:.4f})")
                low_corr_pair = (n1, n2)
        
        if high_corr_pair and low_corr_pair:
            break
            
    if not high_corr_pair:
        print("Could not find high correlation pair.")
    if not low_corr_pair:
        print("Could not find low correlation pair.")

if __name__ == "__main__":
    find_correlated_pair()
