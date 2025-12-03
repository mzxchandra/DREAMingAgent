"""
Reconciler Node: The Reasoning Engine

Compares literature assertions (RegulonDB) with statistical evidence (M3D)
to validate, contradict, or discover regulatory interactions.

This is where biology meets statistics.
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
from loguru import logger

from ..state import (
    AgentState,
    ReconciliationResult,
    ReconciliationStatus,
    EvidenceLevel,
    InteractionEffect
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Z-score thresholds for reconciliation logic
HIGH_DATA_SUPPORT = 4.0      # Strong data support
MODERATE_DATA_SUPPORT = 2.0  # Some data support
LOW_DATA_SUPPORT = 1.0       # Weak/no data support


# ============================================================================
# Main Reconciler Node
# ============================================================================

def reconciler_node(state: AgentState) -> Dict[str, Any]:
    """
    Reconciler Node: Compares Literature vs Data.
    
    This node executes the core reconciliation logic:
    
    Case A (Validation): Strong lit + High data → Confirmed-Active
    Case B (Conditional Silence): Strong lit + Low data → Context Gap
    Case C (Contradiction): Weak lit + Low data → Probable False Positive
    Case D (Discovery): No lit + High data → Novel Hypothesis
    
    Args:
        state: Current agent state with:
            - current_batch_tfs: TFs being processed
            - analysis_results: Statistical results from Analysis Agent
            - literature_graph: RegulonDB knowledge graph
            
    Returns:
        Updated state with reconciliation_log populated
    """
    logger.info("=== RECONCILER NODE: Comparing Literature vs Data ===")
    
    current_batch_tfs = state.get("current_batch_tfs", [])
    analysis_results = state.get("analysis_results", {})
    literature_graph = state.get("literature_graph", nx.DiGraph())
    current_context = state.get("current_context", "unknown")
    bnumber_to_name = state.get("bnumber_to_gene_name", {})
    
    # Get existing logs
    reconciliation_log = state.get("reconciliation_log", []).copy()
    novel_hypotheses = state.get("novel_hypotheses", []).copy()
    false_positive_candidates = state.get("false_positive_candidates", []).copy()
    
    for tf in current_batch_tfs:
        if tf not in analysis_results:
            logger.warning(f"No analysis results for TF {tf}")
            continue
        
        tf_results = analysis_results[tf]
        
        # Handle error cases
        if isinstance(tf_results, dict) and "error" in tf_results:
            logger.warning(f"TF {tf} analysis error: {tf_results['error']}")
            continue
        
        if isinstance(tf_results, dict) and tf_results.get("status") == "TF_LOW_EXPRESSION":
            logger.info(f"TF {tf} has low expression, skipping reconciliation")
            continue
        
        # Get all known targets for this TF from literature
        literature_targets = _get_literature_targets(tf, literature_graph)
        logger.info(f"TF {tf}: {len(literature_targets)} known literature targets")
        
        # Process each known target (Cases A, B, C)
        for target, lit_attrs in literature_targets.items():
            result = _reconcile_known_edge(
                tf=tf,
                target=target,
                literature_attrs=lit_attrs,
                data_results=tf_results.get(target),
                context=current_context,
                bnumber_to_name=bnumber_to_name
            )
            
            reconciliation_log.append(result.to_dict())
            
            # Track false positive candidates
            if result.status == "ProbableFalsePos":
                false_positive_candidates.append(result.to_dict())
        
        # Process novel discoveries (Case D)
        novel_edges = _find_novel_edges(
            tf=tf,
            tf_results=tf_results,
            literature_graph=literature_graph,
            context=current_context,
            bnumber_to_name=bnumber_to_name
        )
        
        for novel in novel_edges:
            reconciliation_log.append(novel.to_dict())
            novel_hypotheses.append(novel.to_dict())
        
        logger.info(
            f"TF {tf}: {len(novel_edges)} novel hypotheses discovered"
        )
    
    # Summary statistics
    total_validated = sum(1 for r in reconciliation_log if r.get("reconciliation_status") == "Validated")
    total_silent = sum(1 for r in reconciliation_log if r.get("reconciliation_status") == "ConditionSilent")
    total_false_pos = len(false_positive_candidates)
    total_novel = len(novel_hypotheses)
    
    logger.info(
        f"Reconciliation summary: {total_validated} validated, "
        f"{total_silent} condition-silent, {total_false_pos} probable false positives, "
        f"{total_novel} novel hypotheses"
    )
    
    return {
        "reconciliation_log": reconciliation_log,
        "novel_hypotheses": novel_hypotheses,
        "false_positive_candidates": false_positive_candidates
    }


# ============================================================================
# Reconciliation Logic Functions
# ============================================================================

def _get_literature_targets(
    tf: str,
    graph: nx.DiGraph
) -> Dict[str, Dict[str, Any]]:
    """
    Get all known targets for a TF from the literature graph.
    
    Args:
        tf: TF identifier
        graph: Literature graph
        
    Returns:
        Dictionary mapping target genes to edge attributes
    """
    targets = {}
    
    if not graph.has_node(tf):
        return targets
    
    for _, target, attrs in graph.out_edges(tf, data=True):
        targets[target] = attrs
    
    return targets


def _reconcile_known_edge(
    tf: str,
    target: str,
    literature_attrs: Dict[str, Any],
    data_results: Optional[Dict[str, Any]],
    context: str,
    bnumber_to_name: Dict[str, str]
) -> ReconciliationResult:
    """
    Reconcile a known literature edge with data evidence.
    
    Implements the core logic:
    - Case A: Strong lit + High data → Validated
    - Case B: Strong lit + Low data → ConditionSilent
    - Case C: Weak lit + Low data → ProbableFalsePos
    
    Args:
        tf: TF identifier
        target: Target gene identifier
        literature_attrs: Edge attributes from RegulonDB
        data_results: Statistical results from Analysis Agent (or None)
        context: Current analysis context
        bnumber_to_name: ID mapping
        
    Returns:
        ReconciliationResult object
    """
    # Extract literature information
    lit_evidence = literature_attrs.get("evidence_type", "Unknown")
    lit_effect = literature_attrs.get("effect", "?")
    tf_name = bnumber_to_name.get(tf, tf)
    gene_name = bnumber_to_name.get(target, target)
    
    # Normalize evidence level
    if lit_evidence in ["Strong", "Confirmed"]:
        evidence_level: EvidenceLevel = "Strong"
    elif lit_evidence == "Weak":
        evidence_level = "Weak"
    else:
        evidence_level = "Unknown"
    
    # Extract data evidence
    if data_results is None:
        # Target not in expression matrix or not analyzed
        data_z_score = 0.0
        data_mi = 0.0
        notes = "Target gene not found in expression data"
        status: ReconciliationStatus = "ConditionSilent"
    else:
        data_z_score = data_results.get("z_score", 0.0)
        data_mi = data_results.get("mi", 0.0)
        
        # Apply reconciliation logic
        status, notes = _determine_status(
            evidence_level=evidence_level,
            z_score=data_z_score,
            tf_name=tf_name,
            gene_name=gene_name
        )
    
    return ReconciliationResult(
        source_tf=tf,
        target_gene=target,
        literature_evidence=evidence_level,
        literature_effect=lit_effect,
        data_z_score=data_z_score,
        data_mi=data_mi,
        status=status,
        context_tags=[context] if context else [],
        notes=notes
    )


def _determine_status(
    evidence_level: EvidenceLevel,
    z_score: float,
    tf_name: str,
    gene_name: str
) -> Tuple[ReconciliationStatus, str]:
    """
    Determine reconciliation status based on evidence and data support.
    
    Args:
        evidence_level: Literature evidence strength
        z_score: CLR z-score from data
        tf_name: TF name for logging
        gene_name: Gene name for logging
        
    Returns:
        Tuple of (status, notes)
    """
    has_high_data = z_score >= HIGH_DATA_SUPPORT
    has_moderate_data = z_score >= MODERATE_DATA_SUPPORT
    has_low_data = z_score < LOW_DATA_SUPPORT
    
    # Case A: Strong literature + High data → Validated
    if evidence_level == "Strong" and has_high_data:
        return (
            "Validated",
            f"Strong literature evidence confirmed by data (z={z_score:.2f})"
        )
    
    # Case A (moderate): Strong literature + Moderate data → Validated
    if evidence_level == "Strong" and has_moderate_data:
        return (
            "Validated",
            f"Strong literature evidence with moderate data support (z={z_score:.2f})"
        )
    
    # Case B: Strong literature + Low data → Conditional Silence
    if evidence_level == "Strong" and has_low_data:
        return (
            "ConditionSilent",
            f"Physical binding exists but interaction silent in current conditions (z={z_score:.2f}). "
            "TF may be inactive in sampled experiments."
        )
    
    # Case B (intermediate): Strong literature + neither high nor low
    if evidence_level == "Strong":
        return (
            "ConditionSilent",
            f"Strong evidence but marginal data support (z={z_score:.2f}). "
            "Possible partial activation in subset of conditions."
        )
    
    # Case C: Weak literature + Low data → Probable False Positive
    if evidence_level == "Weak" and has_low_data:
        return (
            "ProbableFalsePos",
            f"Weak literature evidence unsupported by data (z={z_score:.2f}). "
            "Candidate for database pruning."
        )
    
    # Weak literature + High data → Upgrade to Validated
    if evidence_level == "Weak" and has_high_data:
        return (
            "Validated",
            f"Weak literature evidence upgraded by strong data support (z={z_score:.2f})"
        )
    
    # Weak literature + Moderate data → Needs further investigation
    if evidence_level == "Weak" and has_moderate_data:
        return (
            "Pending",
            f"Weak literature evidence with some data support (z={z_score:.2f}). "
            "Requires further validation."
        )
    
    # Unknown evidence level
    if evidence_level == "Unknown":
        if has_high_data:
            return (
                "Validated",
                f"Unknown evidence type but strong data support (z={z_score:.2f})"
            )
        else:
            return (
                "Pending",
                f"Unknown evidence type with z={z_score:.2f}. Manual review recommended."
            )
    
    # Default fallback
    return (
        "Pending",
        f"Edge requires manual review (evidence={evidence_level}, z={z_score:.2f})"
    )


def _find_novel_edges(
    tf: str,
    tf_results: Dict[str, Any],
    literature_graph: nx.DiGraph,
    context: str,
    bnumber_to_name: Dict[str, str]
) -> List[ReconciliationResult]:
    """
    Find novel regulatory relationships not in literature.
    
    Case D: No literature + High data → Novel Hypothesis
    
    Args:
        tf: TF identifier
        tf_results: Analysis results for this TF
        literature_graph: Literature graph
        context: Current analysis context
        bnumber_to_name: ID mapping
        
    Returns:
        List of novel hypothesis ReconciliationResults
    """
    novel_edges = []
    
    # Skip if results indicate an error
    if not isinstance(tf_results, dict):
        return novel_edges
    
    if "_summary" not in tf_results and "error" in tf_results:
        return novel_edges
    
    for gene, result in tf_results.items():
        # Skip metadata keys
        if gene.startswith("_"):
            continue
        
        # Skip if it's a known literature target
        if literature_graph.has_edge(tf, gene):
            continue
        
        # Skip non-dict results
        if not isinstance(result, dict):
            continue
        
        z_score = result.get("z_score", 0.0)
        mi = result.get("mi", 0.0)
        
        # Only report high-confidence novel discoveries
        if z_score >= HIGH_DATA_SUPPORT:
            tf_name = bnumber_to_name.get(tf, tf)
            gene_name = bnumber_to_name.get(gene, gene)
            
            novel = ReconciliationResult(
                source_tf=tf,
                target_gene=gene,
                literature_evidence="Unknown",
                literature_effect="?",
                data_z_score=z_score,
                data_mi=mi,
                status="NovelHypothesis",
                context_tags=[context] if context else [],
                notes=f"High data support (z={z_score:.2f}) without literature record. "
                      f"Candidate for experimental validation. "
                      f"Check HT-TFBS (ChIP-seq) for binding evidence."
            )
            novel_edges.append(novel)
    
    return novel_edges


# ============================================================================
# Output Generation Functions
# ============================================================================

def generate_reconciliation_summary(
    reconciliation_log: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a summary of reconciliation results.
    
    Args:
        reconciliation_log: Full reconciliation log
        
    Returns:
        Summary statistics dictionary
    """
    status_counts = {
        "Validated": 0,
        "ConditionSilent": 0,
        "ProbableFalsePos": 0,
        "NovelHypothesis": 0,
        "Pending": 0
    }
    
    for entry in reconciliation_log:
        status = entry.get("reconciliation_status", "Pending")
        if status in status_counts:
            status_counts[status] += 1
    
    # Calculate statistics
    total = len(reconciliation_log)
    
    summary = {
        "total_edges_analyzed": total,
        "status_counts": status_counts,
        "validation_rate": status_counts["Validated"] / total if total > 0 else 0,
        "false_positive_rate": status_counts["ProbableFalsePos"] / total if total > 0 else 0,
        "novel_discovery_count": status_counts["NovelHypothesis"]
    }
    
    return summary


def export_reconciled_network(
    reconciliation_log: List[Dict[str, Any]],
    output_format: str = "tsv"
) -> str:
    """
    Export reconciled network to string format.
    
    Args:
        reconciliation_log: Full reconciliation log
        output_format: Output format ("tsv", "graphml")
        
    Returns:
        Formatted string representation
    """
    if output_format == "tsv":
        headers = [
            "Source_TF", "Target_Gene", "RegulonDB_Evidence", 
            "RegulonDB_Effect", "M3D_Z_Score", "M3D_MI",
            "Reconciliation_Status", "Context_Tags", "Notes"
        ]
        
        lines = ["\t".join(headers)]
        
        for entry in reconciliation_log:
            row = [
                entry.get("source_tf", ""),
                entry.get("target_gene", ""),
                entry.get("regulondb_evidence", ""),
                entry.get("regulondb_effect", ""),
                f"{entry.get('m3d_z_score', 0):.4f}",
                f"{entry.get('m3d_mi', 0):.4f}",
                entry.get("reconciliation_status", ""),
                ",".join(entry.get("context_tags", [])),
                entry.get("notes", "").replace("\t", " ").replace("\n", " ")
            ]
            lines.append("\t".join(row))
        
        return "\n".join(lines)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

