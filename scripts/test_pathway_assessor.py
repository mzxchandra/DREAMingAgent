
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.pathway_assessor.assessor import PathwayAssessor
from loguru import logger

def test():
    logger.info("Initializing PathwayAssessor...")
    assessor = PathwayAssessor()
    
    cases = [
        "AraC regulates araA",
        "AraC regulates lacZ",
        "rplC regulates rpsK",
        "yagX regulates gntT",
        "NonExistentGeneA regulates NonExistentGeneB"
    ]
    
    for case in cases:
        print(f"\n--- Testing: {case} ---")
        result = assessor.assess_pathway(case)
        print(f"Result: {result}")

if __name__ == "__main__":
    test()
