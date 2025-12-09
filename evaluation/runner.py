import argparse
import sys
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from .config import EvaluationConfig
from .utils.report_generator import generate_combined_report, generate_metric_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DREAMing Agent Evaluation Runner")
    parser.add_argument("--all", action="store_true", help="Run all metrics")
    parser.add_argument("--metric", nargs="+", choices=["A", "B", "D"], help="Run specific metrics")
    parser.add_argument("--output-dir", type=str, default="evaluation/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.all and not args.metric:
        parser.print_help()
        sys.exit(1)
        
    # Setup Config
    output_dir = Path(args.output_dir)
    config = EvaluationConfig(output_dir=output_dir)
    
    # Select Metrics
    metrics_to_run = []
    if args.all:
        metrics_to_run = ["A", "B", "D"]
    else:
        metrics_to_run = args.metric
        
    results_list = []
    
    # Dynamic Import to avoid heavy load if not running everything
    from .metrics.metric_a_sabotage import MetricASabotage
    from .metrics.metric_b_synthetic import MetricBSynthetic
    from .metrics.metric_d_llm_judge import MetricDLLMJudge
    
    metric_map = {
        "A": MetricASabotage,
        "B": MetricBSynthetic,
        "D": MetricDLLMJudge
    }
    
    for m_code in metrics_to_run:
        metric_class = metric_map[m_code]
        metric_instance = metric_class(config)
        
        try:
            result = metric_instance.execute(config.output_dir)
            results_list.append(result)
            
            # Generate individual report
            report_path = config.output_dir / f"report_{metric_instance.name}.md"
            generate_metric_report(metric_instance.name, result, report_path)
            
        except Exception as e:
            logger.error(f"Failed to run metric {m_code}: {e}", exc_info=True)
            
    # Combine Report
    if results_list:
        combined_path = config.output_dir / "evaluation_summary.md"
        generate_combined_report(results_list, combined_path)
        logger.info(f"Summary report generated at {combined_path}")

if __name__ == "__main__":
    main()
