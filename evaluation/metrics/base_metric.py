from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any
import logging
import json

from ..config import EvaluationConfig

logger = logging.getLogger(__name__)

class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def prepare_data(self) -> Dict[str, Any]:
        """Prepare data for the evaluation run.
        
        Returns:
            Dict containing paths or data objects needed for evaluation.
        """
        pass

    @abstractmethod
    def run_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the core evaluation logic.
        
        Args:
            data: Output from prepare_data()
            
        Returns:
            Dict containing raw evaluation results.
        """
        pass

    @abstractmethod
    def compute_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute quantitative scores from results.
        
        Args:
            results: Output from run_evaluation()
            
        Returns:
            Dict of score names to float values.
        """
        pass

    @abstractmethod
    def generate_plots(self, results: Dict[str, Any], output_dir: Path) -> List[Path]:
        """Generate visualization plots.
        
        Args:
            results: Output from run_evaluation()
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plot files.
        """
        pass

    def execute(self, output_dir: Path) -> Dict[str, Any]:
        """Orchestrate the full evaluation pipeline for this metric."""
        logger.info(f"Starting execution of {self.name}...")
        
        # 1. Prepare Data
        logger.info(f"[{self.name}] Preparing data...")
        data = self.prepare_data()
        
        # 2. Run Evaluation
        logger.info(f"[{self.name}] Running evaluation...")
        results = self.run_evaluation(data)
        
        # 3. Compute Scores
        logger.info(f"[{self.name}] Computing scores...")
        scores = self.compute_scores(results)
        
        # 4. Generate Plots
        logger.info(f"[{self.name}] Generating plots...")
        plot_paths = self.generate_plots(results, output_dir)
        
        # 5. Save Results
        result_pkg = {
            "metric": self.name,
            "scores": scores,
            "plot_paths": [str(p) for p in plot_paths],
            # Avoid serializing potentially large raw data if not needed, 
            # but usually helpful for debugging.
            # "raw_results": results 
        }
        
        json_path = output_dir / f"{self.name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(result_pkg, f, indent=2)
            
        logger.info(f"[{self.name}] Completed. Results saved to {json_path}")
        return result_pkg
