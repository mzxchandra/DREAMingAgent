#!/usr/bin/env python3
"""
DREAMing Agent: Main Entry Point

Agentic Reconciliation of Biological Literature and High-Throughput Data

Usage:
    python main.py --help                          # Show help
    python main.py --synthetic                     # Run with synthetic test data
    python main.py --network path/to/network.txt  # Run with real data
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/dreaming_agent_{time}.log",
    rotation="10 MB",
    level="DEBUG"
)


def main():
    parser = argparse.ArgumentParser(
        description="DREAMing Agent: Agentic Reconciliation of Gene Regulatory Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic test data
  python main.py --synthetic

  # Run with real data files
  python main.py \\
    --network data/network_tf_gene.txt \\
    --genes data/gene_product.txt \\
    --expression data/E_coli_v4_Build_6_exps.tab \\
    --metadata data/E_coli_v4_Build_6_meta.tab \\
    --output results/

  # Run specific TFs only
  python main.py --synthetic --tfs FNR,CRP,ArcA
        """
    )
    
    # Data paths
    parser.add_argument(
        "--network", "-n",
        type=Path,
        help="Path to RegulonDB network_tf_gene.txt"
    )
    parser.add_argument(
        "--genes", "-g",
        type=Path,
        help="Path to RegulonDB gene_product.txt"
    )
    parser.add_argument(
        "--expression", "-e",
        type=Path,
        help="Path to M3D expression matrix (E_coli_v4_Build_6_exps.tab)"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        help="Path to M3D metadata (E_coli_v4_Build_6_meta.tab)"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for results (default: output/)"
    )
    
    # Options
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Run with synthetic test data (no real files needed)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of processing iterations (default: 100)"
    )
    parser.add_argument(
        "--tfs",
        type=str,
        help="Comma-separated list of specific TFs to analyze"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # LLM Options
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM reasoning (use rule-based fallbacks)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Import after argument parsing to avoid slow startup for --help
    from src.workflow import (
        run_reconciliation,
        run_with_synthetic_data,
        export_results
    )
    from src.config import AgentConfig, set_config, disable_llm
    
    # Configure LLM usage
    if args.no_llm:
        logger.info("LLM reasoning DISABLED - using rule-based fallbacks")
        disable_llm()
    else:
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            logger.warning(
                "GEMINI_API_KEY not set. LLM features will be disabled. "
                "Set the environment variable or use --no-llm flag."
            )
            disable_llm()
        else:
            logger.info(f"LLM reasoning ENABLED - using {args.llm_model}")
            config = AgentConfig(
                use_llm=True,
                llm_model=args.llm_model,
                use_llm_reconciler=True,
                use_llm_context=True
            )
            set_config(config)
    
    try:
        if args.synthetic:
            # Run with synthetic data
            logger.info("Running with synthetic test data...")
            final_state = run_with_synthetic_data(
                output_dir=str(args.output / "synthetic_data")
            )
        else:
            # Validate required arguments
            if not all([args.network, args.genes, args.expression, args.metadata]):
                parser.error(
                    "When not using --synthetic, all data paths are required: "
                    "--network, --genes, --expression, --metadata"
                )
            
            # Run with real data
            logger.info("Running with provided data files...")
            final_state = run_reconciliation(
                regulondb_network_path=args.network,
                regulondb_gene_product_path=args.genes,
                m3d_expression_path=args.expression,
                m3d_metadata_path=args.metadata,
                max_iterations=args.max_iterations
            )
        
        # Export results
        args.output.mkdir(parents=True, exist_ok=True)
        output_files = export_results(
            final_state,
            args.output,
            formats=["tsv", "json"]
        )
        
        logger.info("=" * 60)
        logger.info("RESULTS EXPORTED:")
        for fmt, path in output_files.items():
            logger.info(f"  {fmt.upper()}: {path}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


