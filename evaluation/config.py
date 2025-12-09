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
    sabotage_n_false_positives: int = 20
    sabotage_n_deletions: int = 15

    # Metric B: Synthetic
    synthetic_n_genes: int = 100
    synthetic_n_tfs: int = 10
    synthetic_n_experiments: int = 50

    # Metric D: LLM Judge
    judge_model: str = "openai/gpt-oss-120b"
    judge_sample_size: int = 30
    judge_tfs: List[str] = field(default_factory=lambda: ["FNR", "ArcA", "CRP"]) # TFs to test for Metric D (Real Data)

    # General
    random_seed: int = 42
