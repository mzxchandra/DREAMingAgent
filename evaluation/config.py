from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class EvaluationConfig:
    """Configuration for DREAMing Agent Evaluation Framework."""
    
    # Paths
    data_dir: Path = Path("evaluation/data")
    output_dir: Path = Path("evaluation/outputs")
    
    # ensure directories exist
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Metric A: Sabotage
    sabotage_n_false_positives: int = 5 # Reduced for smoke test
    sabotage_n_deletions: int = 5 # Reduced for smoke test

    # Metric B: Real Data
    # (No specific params needed yet, uses real data paths)

    # Metric D: LLM Judge
    judge_model: str = "openai/gpt-oss-120b"
    judge_sample_size: int = 30
    judge_tfs: List[str] = field(default_factory=lambda: ["FNR", "ArcA", "CRP"]) # TFs to test for Metric D (Real Data)
    
    # Execution Limits (for testing/smoke tests)
    max_iterations: int = 100
    limit_tfs: List[str] = field(default_factory=list) # Empty means all

    # General
    random_seed: int = 42
