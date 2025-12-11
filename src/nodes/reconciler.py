"""
Reconciler Node: The AI Reasoning Engine

Compares literature assertions (RegulonDB) with statistical evidence (M3D)
to validate, contradict, or discover regulatory interactions.

Uses LLM (ALCF/Argonne) for nuanced biological reasoning when available,
with fallback to rule-based decisions.

This is where biology meets AI.
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
from ..llm_config import create_argonne_llm
from ..llm.client import generate_structured_response
from ..llm.prompts import (
    RECONCILER_SYSTEM_PROMPT,
    format_reconciliation_prompt,
    format_batch_reconciliation_prompt
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Z-score thresholds for reconciliation logic (used in fallback mode)
HIGH_DATA_SUPPORT = 2.5      # Lowered from 4.0 to catch "long tail" signals (Discovery Threshold)
MODERATE_DATA_SUPPORT = 2.0  # Some data support
LOW_DATA_SUPPORT = 1.0       # Weak/no data support

# Whether to use LLM for reasoning (can be toggled)
USE_LLM_REASONING = True

# Use batch mode for efficiency (one LLM call per TF instead of per edge)
USE_BATCH_MODE = True


# ============================================================================
# Main Reconciler Node
# ============================================================================

def reconciler_node(state: AgentState) -> Dict[str, Any]:
    """
    Reconciler Node: Compares Literature vs Data using AI Reasoning.
    
    This node executes reconciliation logic using LLM when available:
    - Sends edge information to Gemini for biological reasoning
    - Falls back to rule-based logic if LLM unavailable
    
    Cases handled:
    - Case A (Validation): Strong lit + High data → Confirmed-Active
    - Case B (Conditional Silence): Strong lit + Low data → Context Gap
    - Case C (Contradiction): Weak lit + Low data → Probable False Positive
    - Case D (Discovery): No lit + High data → Novel Hypothesis
    
    Args:
        state: Current agent state
            
    Returns:
        Updated state with reconciliation_log populated
    """
    logger.info("=== RECONCILER NODE: AI-Powered Reconciliation ===")
    
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
        tf_name = bnumber_to_name.get(tf, tf)
        
        # Handle error cases
        if isinstance(tf_results, dict) and "error" in tf_results:
            logger.warning(f"TF {tf} analysis error: {tf_results['error']}")
            continue
        
        if isinstance(tf_results, dict) and tf_results.get("status") == "TF_LOW_EXPRESSION":
            logger.info(f"TF {tf} has low expression, skipping reconciliation")
            continue
        
        # Get all known targets for this TF from literature
        literature_targets = _get_literature_targets(tf, literature_graph)
        logger.info(f"TF {tf_name}: {len(literature_targets)} known literature targets")
        
        # Process using LLM or fallback
        if USE_LLM_REASONING and USE_BATCH_MODE:
            # Batch mode: one LLM call for all edges of this TF
            results = _reconcile_batch_with_llm(
                tf=tf,
                tf_name=tf_name,
                literature_targets=literature_targets,
                tf_results=tf_results,
                context=current_context,
                bnumber_to_name=bnumber_to_name
            )
        else:
            # Individual processing (LLM per edge or rule-based)
            results = _reconcile_individually(
                tf=tf,
                tf_name=tf_name,
                literature_targets=literature_targets,
                tf_results=tf_results,
                context=current_context,
                bnumber_to_name=bnumber_to_name,
                use_llm=USE_LLM_REASONING
            )
        
        # Add results to logs
        for result in results:
            result_dict = result.to_dict()
            reconciliation_log.append(result_dict)
            
            if result.status == "ProbableFalsePos":
                false_positive_candidates.append(result_dict)
            elif result.status == "NovelHypothesis":
                novel_hypotheses.append(result_dict)
        
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
        
        logger.info(f"TF {tf_name}: {len(novel_edges)} novel hypotheses discovered")
    
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
# LLM-Powered Reconciliation
# ============================================================================

def _reconcile_batch_with_llm(
    tf: str,
    tf_name: str,
    literature_targets: Dict[str, Dict[str, Any]],
    tf_results: Dict[str, Any],
    context: str,
    bnumber_to_name: Dict[str, str]
) -> List[ReconciliationResult]:
    """
    Reconcile all edges for a TF in a single LLM call (batch mode).
    """
    results = []
    
    # Prepare edge data for batch prompt
    edges_for_llm = []
    edge_lookup = {}  # To map back results
    
    for target, lit_attrs in literature_targets.items():
        target_name = bnumber_to_name.get(target, target)
        data_results = tf_results.get(target)
        
        z_score = data_results.get("z_score", 0.0) if data_results else 0.0
        mi = data_results.get("mi", 0.0) if data_results else 0.0
        
        edge_data = {
            "target": target_name,
            "target_id": target,
            "evidence": lit_attrs.get("evidence_type", "Unknown"),
            "effect": lit_attrs.get("effect", "?"),
            "z_score": z_score,
            "mi": mi
        }
        edges_for_llm.append(edge_data)
        edge_lookup[target_name] = (target, lit_attrs, data_results)
    
    if not edges_for_llm:
        return results
    
    # Call LLM
    prompt = format_batch_reconciliation_prompt(tf_name, edges_for_llm, context)
    llm_response = generate_structured_response(prompt, RECONCILER_SYSTEM_PROMPT)
    
    if llm_response and "edges" in llm_response:
        # Process LLM results
        logger.info(f"LLM batch reasoning for {tf_name}: {llm_response.get('summary', 'N/A')}")
        
        for edge_result in llm_response["edges"]:
            target_name = edge_result.get("target")
            if target_name not in edge_lookup:
                continue
            
            target_id, lit_attrs, data_results = edge_lookup[target_name]
            
            # Map LLM status to our status enum
            llm_status = edge_result.get("status", "NeedsReview")
            status = _map_llm_status(llm_status)
            
            z_score = data_results.get("z_score", 0.0) if data_results else 0.0
            mi = data_results.get("mi", 0.0) if data_results else 0.0
            
            result = ReconciliationResult(
                source_tf=tf,
                target_gene=target_id,
                literature_evidence=lit_attrs.get("evidence_type", "Unknown"),
                literature_effect=lit_attrs.get("effect", "?"),
                data_z_score=z_score,
                data_mi=mi,
                status=status,
                context_tags=[context] if context else [],
                notes=f"[AI] {edge_result.get('note', 'LLM reasoning applied')}"
            )
            results.append(result)
            
            # Remove from lookup to track unprocessed
            del edge_lookup[target_name]
    
    # Fallback for any edges not processed by LLM
    for target_name, (target_id, lit_attrs, data_results) in edge_lookup.items():
        result = _reconcile_known_edge_rule_based(
            tf=tf,
            target=target_id,
            literature_attrs=lit_attrs,
            data_results=data_results,
            context=context,
            bnumber_to_name=bnumber_to_name
        )
        results.append(result)
    
    return results


def _reconcile_with_llm(
    tf: str,
    tf_name: str,
    target: str,
    target_name: str,
    literature_attrs: Dict[str, Any],
    data_results: Optional[Dict[str, Any]],
    context: str
) -> Optional[ReconciliationResult]:
    """
    Use LLM to reason about a single edge.
    
    Returns None if LLM fails (caller should use fallback).
    """
    lit_evidence = literature_attrs.get("evidence_type", "Unknown")
    lit_effect = literature_attrs.get("effect", "?")
    z_score = data_results.get("z_score", 0.0) if data_results else 0.0
    mi = data_results.get("mi", 0.0) if data_results else 0.0
    
    # Format prompt
    prompt = format_reconciliation_prompt(
        tf_name=tf_name,
        target_gene=target_name,
        literature_evidence=lit_evidence,
        literature_effect=lit_effect,
        z_score=z_score,
        mi_score=mi,
        context=context
    )
    
    # Get LLM response
    llm_response = generate_structured_response(prompt, RECONCILER_SYSTEM_PROMPT)
    
    if llm_response is None:
        return None
    
    # Parse LLM decision
    llm_status = llm_response.get("status", "NeedsReview")
    confidence = llm_response.get("confidence", 0.5)
    reasoning = llm_response.get("reasoning", "")
    recommendation = llm_response.get("recommendation", "")
    
    # Map LLM status to our enum
    status = _map_llm_status(llm_status)
    
    # Build notes from LLM reasoning
    notes = f"[AI Confidence: {confidence:.0%}] {reasoning}"
    if recommendation:
        notes += f" Recommendation: {recommendation}"
    
    return ReconciliationResult(
        source_tf=tf,
        target_gene=target,
        literature_evidence=lit_evidence,
        literature_effect=lit_effect,
        data_z_score=z_score,
        data_mi=mi,
        status=status,
        context_tags=[context] if context else [],
        notes=notes
    )


def _map_llm_status(llm_status: str) -> ReconciliationStatus:
    """Map LLM status string to our enum."""
    status_map = {
        "Validated": "Validated",
        "ConditionSilent": "ConditionSilent",
        "ProbableFalsePos": "ProbableFalsePos",
        "NovelHypothesis": "NovelHypothesis",
        "NeedsReview": "Pending",
        "Pending": "Pending"
    }
    return status_map.get(llm_status, "Pending")


# ============================================================================
# Individual Processing (LLM or Rule-Based)
# ============================================================================

def _reconcile_individually(
    tf: str,
    tf_name: str,
    literature_targets: Dict[str, Dict[str, Any]],
    tf_results: Dict[str, Any],
    context: str,
    bnumber_to_name: Dict[str, str],
    use_llm: bool = True
) -> List[ReconciliationResult]:
    """Process each edge individually."""
    results = []
    
    for target, lit_attrs in literature_targets.items():
        target_name = bnumber_to_name.get(target, target)
        data_results = tf_results.get(target)
        
        result = None
        
        # Try LLM first if enabled
        if use_llm:
            result = _reconcile_with_llm(
                tf=tf,
                tf_name=tf_name,
                target=target,
                target_name=target_name,
                literature_attrs=lit_attrs,
                data_results=data_results,
                context=context
            )
        
        # Fallback to rule-based if LLM failed or disabled
        if result is None:
            result = _reconcile_known_edge_rule_based(
                tf=tf,
                target=target,
                literature_attrs=lit_attrs,
                data_results=data_results,
                context=context,
                bnumber_to_name=bnumber_to_name
            )
        
        results.append(result)
    
    return results


# ============================================================================
# Rule-Based Fallback (Original Logic)
# ============================================================================

def _get_literature_targets(
    tf: str,
    graph: nx.DiGraph
) -> Dict[str, Dict[str, Any]]:
    """Get all known targets for a TF from the literature graph."""
    targets = {}
    
    if not graph.has_node(tf):
        return targets
    
    for _, target, attrs in graph.out_edges(tf, data=True):
        targets[target] = attrs
    
    return targets


def _reconcile_known_edge_rule_based(
    tf: str,
    target: str,
    literature_attrs: Dict[str, Any],
    data_results: Optional[Dict[str, Any]],
    context: str,
    bnumber_to_name: Dict[str, str]
) -> ReconciliationResult:
    """
    Rule-based reconciliation (fallback when LLM unavailable).
    """
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
        data_z_score = 0.0
        data_mi = 0.0
        notes = "[Rule-based] Target gene not found in expression data"
        status: ReconciliationStatus = "ConditionSilent"
    else:
        data_z_score = data_results.get("z_score", 0.0)
        data_mi = data_results.get("mi", 0.0)
        
        status, notes = _determine_status_rule_based(
            evidence_level=evidence_level,
            z_score=data_z_score,
            tf_name=tf_name,
            gene_name=gene_name
        )
        notes = f"[Rule-based] {notes}"
    
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


def _determine_status_rule_based(
    evidence_level: EvidenceLevel,
    z_score: float,
    tf_name: str,
    gene_name: str
) -> Tuple[ReconciliationStatus, str]:
    """Determine reconciliation status using rules (original logic)."""
    has_high_data = z_score >= HIGH_DATA_SUPPORT
    has_moderate_data = z_score >= MODERATE_DATA_SUPPORT
    has_low_data = z_score < LOW_DATA_SUPPORT
    
    if evidence_level == "Strong" and has_high_data:
        if "FNR" in tf_name: # Debug Trace for User
             logger.info(f"DEBUG TRACE [FNR->{gene_name}]: Evidence=Strong, Z={z_score:.2f} >= {HIGH_DATA_SUPPORT} -> Validated")
        return ("Validated", f"Strong literature evidence confirmed by data (z={z_score:.2f})")
    
    if evidence_level == "Strong" and has_moderate_data:
        return ("Validated", f"Strong literature evidence with moderate data support (z={z_score:.2f})")
    
    if evidence_level == "Strong" and has_low_data:
        return ("ConditionSilent", f"Physical binding exists but interaction silent in current conditions (z={z_score:.2f})")
    
    if evidence_level == "Strong":
        return ("ConditionSilent", f"Strong evidence but marginal data support (z={z_score:.2f})")
    
    if evidence_level == "Weak" and has_low_data:
        return ("ProbableFalsePos", f"Weak literature evidence unsupported by data (z={z_score:.2f})")
    
    if evidence_level == "Weak" and has_high_data:
        return ("Validated", f"Weak literature evidence upgraded by strong data support (z={z_score:.2f})")
    
    if evidence_level == "Weak" and has_moderate_data:
        return ("Pending", f"Weak literature evidence with some data support (z={z_score:.2f})")
    
    if evidence_level == "Unknown":
        if has_high_data:
            return ("Validated", f"Unknown evidence type but strong data support (z={z_score:.2f})")
        else:
            return ("Pending", f"Unknown evidence type with z={z_score:.2f}")
    
            return ("Pending", f"Unknown evidence type with z={z_score:.2f}")
    
    # Fallback/Default
    if "FNR" in tf_name: # Debug Trace for User
         reason = "Z < Threshold" if z_score < HIGH_DATA_SUPPORT else "Review Required"
         logger.info(f"DEBUG TRACE [FNR->{gene_name}]: Evidence={evidence_level}, Z={z_score:.2f} -> Pending/Reject ({reason})")

    return ("Pending", f"Edge requires manual review (evidence={evidence_level}, z={z_score:.2f})")


def _find_novel_edges(
    tf: str,
    tf_results: Dict[str, Any],
    literature_graph: nx.DiGraph,
    context: str,
    bnumber_to_name: Dict[str, str]
) -> List[ReconciliationResult]:
    """Find novel regulatory relationships not in literature (Case D)."""
    novel_edges = []
    
    if not isinstance(tf_results, dict):
        return novel_edges
    
    if "_summary" not in tf_results and "error" in tf_results:
        return novel_edges
    
    for gene, result in tf_results.items():
        if gene.startswith("_"):
            continue
        
        if literature_graph.has_edge(tf, gene):
            continue
        
        if not isinstance(result, dict):
            continue
        
        z_score = result.get("z_score", 0.0)
        mi = result.get("mi", 0.0)
        
        if z_score >= HIGH_DATA_SUPPORT:
            tf_name = bnumber_to_name.get(tf, tf)
            gene_name = bnumber_to_name.get(gene, gene)
            
            if "FNR" in tf_name or "b1334" in tf: # Debug Trace
                 logger.info(f"DEBUG TRACE [FNR->{gene_name}]: Novelty Candidate! Z={z_score:.2f} >= {HIGH_DATA_SUPPORT}")

            novel = ReconciliationResult(
                source_tf=tf,
                target_gene=gene,
                literature_evidence="Unknown",
                literature_effect="?",
                data_z_score=z_score,
                data_mi=mi,
                status="NovelHypothesis",
                context_tags=[context] if context else [],
                notes=f"[Discovery] High data support (z={z_score:.2f}) without literature record. "
                      f"Candidate for experimental validation."
            )
            novel_edges.append(novel)
    
    return novel_edges


# ============================================================================
# Output Generation Functions
# ============================================================================

def generate_reconciliation_summary(
    reconciliation_log: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate a summary of reconciliation results."""
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
    
    total = len(reconciliation_log)
    
    return {
        "total_edges_analyzed": total,
        "status_counts": status_counts,
        "validation_rate": status_counts["Validated"] / total if total > 0 else 0,
        "false_positive_rate": status_counts["ProbableFalsePos"] / total if total > 0 else 0,
        "novel_discovery_count": status_counts["NovelHypothesis"]
    }


def export_reconciled_network(
    reconciliation_log: List[Dict[str, Any]],
    output_format: str = "tsv"
) -> str:
    """Export reconciled network to string format."""
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
