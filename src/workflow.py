"""
LangGraph Workflow Orchestration

Assembles the complete Agentic Reconciliation System workflow
using LangGraph's StateGraph.

The workflow is cyclic: it loops through batches of TFs until
all have been analyzed and reconciled.

Flow:
    loader → batch_manager → [check queue] → context_agent → analysis_agent → reconciler → batch_manager
                                ↓ (empty)
                               END
"""

from typing import Dict, Any, Literal
from pathlib import Path
from langgraph.graph import StateGraph, END
from loguru import logger

from .state import AgentState, create_initial_state
from .nodes import (
    loader_node,
    batch_manager_node,
    check_queue_status,
    context_agent_node,
    analysis_agent_node,
    reconciler_node
)
from .nodes.loader import LoaderConfig, set_loader_config


def create_reconciliation_workflow() -> StateGraph:
    """
    Create the complete LangGraph workflow for GRN reconciliation.
    
    The workflow consists of 5 nodes connected in a cyclic pattern:
    1. Loader: Ingests RegulonDB and M3D data
    2. Batch Manager: Selects next batch of TFs to process
    3. Context Agent: Filters M3D samples based on TF context
    4. Analysis Agent: Computes CLR-corrected MI scores
    5. Reconciler: Compares literature vs data evidence
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating reconciliation workflow...")
    
    # Initialize StateGraph with AgentState schema
    workflow = StateGraph(AgentState)
    
    # =========================================================================
    # Add Nodes
    # =========================================================================
    
    # 1. LOADER: Ingests text files, builds Graph, cleans Matrix
    workflow.add_node("loader", loader_node)
    
    # 2. BATCHER: Selects the next TF to analyze (prevents OOM errors)
    workflow.add_node("batch_manager", batch_manager_node)
    
    # 3. CONTEXT: Filters M3D metadata (e.g., "Find all Anaerobic samples")
    workflow.add_node("context_agent", context_agent_node)
    
    # 4. ANALYST: The Core Math Engine (CLR/MI calculation)
    workflow.add_node("analysis_agent", analysis_agent_node)
    
    # 5. RECONCILER: The Logic Engine (Literature vs. Data comparison)
    workflow.add_node("reconciler", reconciler_node)
    
    # =========================================================================
    # Add Edges (Control Flow)
    # =========================================================================
    
    # Entry point: Start with the loader
    workflow.set_entry_point("loader")
    
    # Loader → Batch Manager
    workflow.add_edge("loader", "batch_manager")
    
    # Conditional edge: If batch is empty, END. Else, continue processing.
    workflow.add_conditional_edges(
        "batch_manager",
        check_queue_status,
        {
            "process": "context_agent",
            "done": END
        }
    )
    
    # Context → Analysis → Reconciliation (linear flow)
    workflow.add_edge("context_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "reconciler")
    
    # Cycle: After reconciliation, go back to Batch Manager for next TF
    workflow.add_edge("reconciler", "batch_manager")
    
    logger.info("Workflow created successfully")
    
    return workflow


def compile_workflow(workflow: StateGraph):
    """
    Compile the workflow for execution.
    
    Args:
        workflow: StateGraph workflow
        
    Returns:
        Compiled runnable workflow
    """
    return workflow.compile()


def run_reconciliation(
    regulondb_network_path: str | Path,
    regulondb_gene_product_path: str | Path,
    m3d_expression_path: str | Path,
    m3d_metadata_path: str | Path,
    max_iterations: int = 100,
    use_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Run the complete reconciliation pipeline.
    
    This is the main entry point for executing the workflow.
    
    Args:
        regulondb_network_path: Path to network_tf_gene.txt
        regulondb_gene_product_path: Path to gene_product.txt
        m3d_expression_path: Path to M3D expression matrix
        m3d_metadata_path: Path to M3D metadata
        max_iterations: Maximum processing iterations (safety limit)
        use_synthetic: If True, generate synthetic test data
        
    Returns:
        Final agent state with all results
    """
    logger.info("=" * 60)
    logger.info("STARTING AGENTIC RECONCILIATION PIPELINE")
    logger.info("=" * 60)
    
    # Configure loader
    config = LoaderConfig(
        regulondb_network_path=regulondb_network_path,
        regulondb_gene_product_path=regulondb_gene_product_path,
        m3d_expression_path=m3d_expression_path,
        m3d_metadata_path=m3d_metadata_path,
        use_synthetic=use_synthetic
    )
    set_loader_config(config)
    
    # Create and compile workflow
    workflow = create_reconciliation_workflow()
    app = compile_workflow(workflow)
    
    # Create initial state
    initial_state = create_initial_state()
    initial_state["max_iterations"] = max_iterations
    
    # Execute workflow
    logger.info("Executing workflow...")
    
    try:
        final_state = app.invoke(initial_state)
        
        logger.info("=" * 60)
        logger.info("RECONCILIATION COMPLETE")
        logger.info("=" * 60)
        
        # Log summary
        _log_summary(final_state)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise


def run_with_synthetic_data(output_dir: str = "test_data") -> Dict[str, Any]:
    """
    Run the workflow with synthetic test data.
    
    Useful for testing and development without real data files.
    
    Args:
        output_dir: Directory to store synthetic data
        
    Returns:
        Final agent state
    """
    logger.info("Running with synthetic test data...")
    
    # Configure with synthetic data
    config = LoaderConfig(
        regulondb_network_path="",  # Will be generated
        regulondb_gene_product_path="",
        m3d_expression_path="",
        m3d_metadata_path="",
        use_synthetic=True,
        synthetic_output_dir=output_dir
    )
    set_loader_config(config)
    
    # Create and compile workflow
    workflow = create_reconciliation_workflow()
    app = compile_workflow(workflow)
    
    # Execute
    initial_state = create_initial_state()
    initial_state["max_iterations"] = 50
    
    final_state = app.invoke(initial_state)
    
    _log_summary(final_state)
    
    return final_state


def _log_summary(state: Dict[str, Any]):
    """Log a summary of the reconciliation results."""
    reconciliation_log = state.get("reconciliation_log", [])
    novel_hypotheses = state.get("novel_hypotheses", [])
    false_positives = state.get("false_positive_candidates", [])
    
    # Count statuses
    status_counts = {}
    for entry in reconciliation_log:
        status = entry.get("reconciliation_status", "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info(f"Total edges analyzed: {len(reconciliation_log)}")
    logger.info(f"Status breakdown:")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  - {status}: {count}")
    logger.info(f"Novel hypotheses: {len(novel_hypotheses)}")
    logger.info(f"Probable false positives: {len(false_positives)}")
    
    # Log top novel discoveries
    if novel_hypotheses:
        logger.info("Top novel discoveries (by z-score):")
        sorted_novel = sorted(
            novel_hypotheses, 
            key=lambda x: x.get("m3d_z_score", 0), 
            reverse=True
        )[:5]
        for i, novel in enumerate(sorted_novel, 1):
            logger.info(
                f"  {i}. {novel['source_tf']} → {novel['target_gene']} "
                f"(z={novel['m3d_z_score']:.2f})"
            )


# ============================================================================
# Streaming / Step-by-Step Execution
# ============================================================================

def run_reconciliation_streaming(
    regulondb_network_path: str | Path,
    regulondb_gene_product_path: str | Path,
    m3d_expression_path: str | Path,
    m3d_metadata_path: str | Path,
    max_iterations: int = 100
):
    """
    Run the workflow with streaming output.
    
    Yields intermediate states for real-time progress monitoring.
    
    Args:
        regulondb_network_path: Path to network_tf_gene.txt
        regulondb_gene_product_path: Path to gene_product.txt
        m3d_expression_path: Path to M3D expression matrix
        m3d_metadata_path: Path to M3D metadata
        max_iterations: Maximum processing iterations
        
    Yields:
        Tuple of (node_name, state) for each step
    """
    # Configure loader
    config = LoaderConfig(
        regulondb_network_path=regulondb_network_path,
        regulondb_gene_product_path=regulondb_gene_product_path,
        m3d_expression_path=m3d_expression_path,
        m3d_metadata_path=m3d_metadata_path
    )
    set_loader_config(config)
    
    # Create and compile workflow
    workflow = create_reconciliation_workflow()
    app = compile_workflow(workflow)
    
    # Create initial state
    initial_state = create_initial_state()
    initial_state["max_iterations"] = max_iterations
    
    # Stream execution
    for output in app.stream(initial_state):
        for node_name, state in output.items():
            logger.info(f"Completed node: {node_name}")
            yield node_name, state


# ============================================================================
# Export Functions
# ============================================================================

def export_results(
    state: Dict[str, Any],
    output_dir: str | Path,
    formats: list = ["tsv", "json"]
) -> Dict[str, Path]:
    """
    Export reconciliation results to files.
    
    Args:
        state: Final workflow state
        output_dir: Output directory
        formats: List of output formats
        
    Returns:
        Dictionary mapping format to output file path
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    reconciliation_log = state.get("reconciliation_log", [])
    novel_hypotheses = state.get("novel_hypotheses", [])
    false_positives = state.get("false_positive_candidates", [])
    
    if "tsv" in formats:
        # Export main reconciliation log
        from .nodes.reconciler import export_reconciled_network
        tsv_content = export_reconciled_network(reconciliation_log, "tsv")
        tsv_path = output_dir / "reconciled_network.tsv"
        with open(tsv_path, 'w') as f:
            f.write(tsv_content)
        output_files["tsv"] = tsv_path
        logger.info(f"Exported TSV to {tsv_path}")
    
    if "json" in formats:
        # Export full results as JSON
        json_data = {
            "reconciliation_log": reconciliation_log,
            "novel_hypotheses": novel_hypotheses,
            "false_positive_candidates": false_positives,
            "summary": {
                "total_edges": len(reconciliation_log),
                "novel_count": len(novel_hypotheses),
                "false_positive_count": len(false_positives)
            }
        }
        json_path = output_dir / "reconciliation_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        output_files["json"] = json_path
        logger.info(f"Exported JSON to {json_path}")
    
    return output_files


