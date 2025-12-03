"""
Batch Manager Node: Controls TF processing queue.

Manages the iteration cycle by selecting batches of TFs to process,
preventing memory overflow and enabling progress tracking.
"""

from typing import Dict, Any, Literal
from loguru import logger

from ..state import AgentState


# Configurable batch size
BATCH_SIZE = 5


def batch_manager_node(state: AgentState) -> Dict[str, Any]:
    """
    Batch Manager Node: Selects next TFs to process.
    
    This node controls the processing queue by:
    1. Checking if there are remaining TFs to process
    2. Selecting the next batch of TFs
    3. Updating iteration count
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with current_batch_tfs populated
    """
    logger.info("=== BATCH MANAGER NODE: Selecting next TF batch ===")
    
    tf_queue = state.get("tf_queue", [])
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 100)
    
    # Safety check for infinite loops
    if iteration_count > max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached, stopping")
        return {
            "current_batch_tfs": [],
            "iteration_count": iteration_count,
            "status": "completed"
        }
    
    # Check if queue is empty
    if not tf_queue:
        logger.info("TF queue is empty, workflow complete")
        return {
            "current_batch_tfs": [],
            "tf_queue": [],
            "iteration_count": iteration_count,
            "status": "completed"
        }
    
    # Select next batch
    batch = tf_queue[:BATCH_SIZE]
    remaining_queue = tf_queue[BATCH_SIZE:]
    
    logger.info(
        f"Iteration {iteration_count}: Processing batch of {len(batch)} TFs "
        f"({len(remaining_queue)} remaining)"
    )
    logger.debug(f"Current batch TFs: {batch}")
    
    return {
        "current_batch_tfs": batch,
        "tf_queue": remaining_queue,
        "iteration_count": iteration_count
    }


def check_queue_status(state: AgentState) -> Literal["process", "done"]:
    """
    Conditional edge function: Determines if processing should continue.
    
    Args:
        state: Current agent state
        
    Returns:
        "process" if there are TFs to analyze, "done" otherwise
    """
    current_batch = state.get("current_batch_tfs", [])
    status = state.get("status", "processing")
    
    if status == "completed" or status == "error":
        logger.info(f"Workflow status: {status}")
        return "done"
    
    if not current_batch:
        logger.info("No TFs in current batch, workflow done")
        return "done"
    
    logger.info(f"Continuing with {len(current_batch)} TFs to process")
    return "process"


