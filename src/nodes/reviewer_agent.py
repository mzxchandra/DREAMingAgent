"""
Reviewer Agent: Synthesizes statistical evidence with biological literature
to produce context-aware gene regulatory network annotations.

Hybrid LLM + rule-based decision system with graceful fallback.
Integrates with vector store for literature context on each edge.
"""

from typing import Dict, List, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from loguru import logger
import networkx as nx

from ..state import AgentState
from ..llm_config import create_argonne_llm
from ..utils.vector_store import get_vector_store
from ..llm.alcf_client import get_alcf_client


class EdgeDecisionOutput(BaseModel):
    """Single edge decision with 100-1500 char scientific explanation."""
    edge_id: str
    reconciliation_status: str
    preserve_edge: bool
    literature_support: str
    data_support: str
    context_compatible: bool
    explanation: str = Field(min_length=100, max_length=1500)
    confidence: str
    recommendation: str = ""


class SubgraphReview(BaseModel):
    """Complete TF subgraph review with cross-edge analysis."""
    tf: str
    edge_decisions: List[EdgeDecisionOutput]
    tf_level_notes: str
    comparative_analysis: Dict[str, Any] = {}

def reviewer_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Main node: Processes one TF subgraph (TF + all targets) per invocation.
    LLM primary with rule-based fallback on error.

    Integrates vector store lookup to get literature context for each edge before reconciliation.

    Args:
        state: AgentState with current_batch_tfs and analysis_results

    Returns:
        Updated state with reconciliation_log, novel_hypotheses, false_positive_candidates
    """
    logger.info("=== REVIEWER AGENT NODE: Literature-Informed Reconciliation ===")

    current_tfs = state.get("current_batch_tfs", [])

    if not current_tfs:
        logger.warning("No TFs to process in reviewer")
        return {
            "reconciliation_log": state.get("reconciliation_log", []),
            "novel_hypotheses": state.get("novel_hypotheses", []),
            "false_positive_candidates": state.get("false_positive_candidates", [])
        }

    tf_bnumber = current_tfs[0]
    edges_data = prepare_subgraph_data(state, tf_bnumber)

    if not edges_data:
        logger.warning(f"No edges found for TF {tf_bnumber}")
        return {
            "reconciliation_log": state.get("reconciliation_log", []),
            "novel_hypotheses": state.get("novel_hypotheses", []),
            "false_positive_candidates": state.get("false_positive_candidates", [])
        }

    logger.info(f"Reviewing {len(edges_data)} edges for TF {tf_bnumber}")

    from ..config import get_config
    config = get_config()
    
    use_llm = config.use_llm and config.use_llm_reconciler
    
    if use_llm:
        try:
            review_result = invoke_llm_reviewer(edges_data, tf_bnumber, state)
            method = "llm"
        except Exception as e:
            logger.warning(f"LLM failed ({e}), using rule-based fallback")
            review_result = apply_rule_based_review(edges_data, tf_bnumber, state)
            method = "rules"
    else:
        logger.info("LLM disabled, using rule-based review")
        review_result = apply_rule_based_review(edges_data, tf_bnumber, state)
        method = "rules"

    return post_process_decisions(review_result, state, tf_bnumber, method, edges_data)


def prepare_subgraph_data(state: AgentState, tf_bnumber: str) -> List[Dict[str, Any]]:
    """
    Combines literature, statistics, and context for all edges of one TF.
    Uses vector store for literature context retrieval.
    """
    edges = []

    # Get data from state
    literature_graph = state.get("literature_graph", nx.DiGraph())
    analysis_results = state.get("analysis_results", {})
    metadata_df = state.get("metadata", None)
    bnumber_to_name = state.get("bnumber_to_gene_name", {})

    # Get stats for this TF
    tf_stats = analysis_results.get(tf_bnumber, {})

    # Get literature targets for this TF
    lit_targets = set()
    if literature_graph.has_node(tf_bnumber):
        lit_targets = {target for _, target in literature_graph.out_edges(tf_bnumber)}

    # Get statistical targets (genes with high correlation)
    stat_targets = set()
    if isinstance(tf_stats, dict):
        stat_targets = {gene for gene in tf_stats.keys() if not gene.startswith("_")}

    # All targets to consider
    all_targets = lit_targets | stat_targets

    logger.info(f"TF {tf_bnumber}: {len(lit_targets)} lit targets, {len(stat_targets)} stat targets, {len(all_targets)} total")

    # Get dataset metadata for research agent
    dataset_conditions = []
    if metadata_df is not None and not metadata_df.empty:
        # Extract common conditions from metadata
        dataset_conditions = extract_dataset_conditions(metadata_df)

    # Initialize vector store for literature lookup
    try:
        vector_store = get_vector_store()
        vector_store_available = True
    except Exception as e:
        logger.warning(f"Vector store unavailable: {e}")
        vector_store_available = False

    for target_bnumber in all_targets:
        edge_id = f"{tf_bnumber}→{target_bnumber}"

        # Get literature data from RegulonDB graph
        lit_data = {"exists": False}
        if literature_graph.has_edge(tf_bnumber, target_bnumber):
            lit_attrs = literature_graph[tf_bnumber][target_bnumber]
            lit_data = {
                "exists": True,
                "effect": lit_attrs.get("effect", "?"),
                "evidence_strength": lit_attrs.get("evidence_type", "unknown").lower(),
                "conditions_required": []  # Will be populated from vector store
            }

        # Get statistical data
        stat_data = tf_stats.get(target_bnumber)
        
        if not isinstance(stat_data, dict) and target_bnumber:
             # Look for key containing bnumber
             for key in tf_stats.keys():
                 if str(key).startswith("_"): continue
                 # Check for bnumber inside key (robust check)
                 if target_bnumber in str(key):
                     stat_data = tf_stats[key]
                     break
        
        if not isinstance(stat_data, dict):
            stat_data = {
                "m3d_z_score": 0.0,
                "m3d_mi": 0.0, 
                "correlation": 0.0, 
                "status": str(stat_data)
            }

        # Get literature context from vector store
        if vector_store_available:
            context_data = get_literature_context(
                vector_store,
                tf_bnumber,
                target_bnumber,
                dataset_conditions,
                bnumber_to_name
            )
        else:
            context_data = {
                "match": False,
                "explanation": "Vector store unavailable - using RegulonDB data only",
                "required_conditions": [],
                "context_found": lit_data["exists"],
                "confidence": 0.5 if lit_data["exists"] else 0.0
            }

        edges.append({
            "edge_id": edge_id,
            "tf": tf_bnumber,
            "target": target_bnumber,
            "literature": lit_data,
            "statistics": stat_data,
            "context": context_data
        })

    return edges


def extract_dataset_conditions(metadata_df) -> List[str]:
    """Extract common experimental conditions from metadata."""
    conditions = []

    # Try common metadata column patterns
    condition_columns = ["condition", "treatment", "media", "temperature", "oxygen"]

    for col in metadata_df.columns:
        col_lower = col.lower()
        if any(cond in col_lower for cond in condition_columns):
            # Get unique values
            unique_vals = metadata_df[col].dropna().unique()
            conditions.extend([str(v) for v in unique_vals[:5]])  # Limit to 5 per column

    # Default if no conditions found
    if not conditions:
        conditions = ["standard_growth", "LB_media", "aerobic"]

    return conditions[:10]  # Limit total conditions


def get_literature_context(
    vector_store,
    tf_bnumber: str,
    target_bnumber: str,
    dataset_conditions: List[str],
    bnumber_to_name: Dict[str, str]
) -> Dict[str, Any]:
    """
    Query vector store for literature context about a TF->gene edge.

    Returns context with condition matching information.
    """
    tf_name = bnumber_to_name.get(tf_bnumber, tf_bnumber)
    target_name = bnumber_to_name.get(target_bnumber, target_bnumber)

    try:
        # Query vector store for broad TF context (since specific edge docs may be missing)
        # We generally check if the TF is known to regulate similar targets or mechanisms
        documents = vector_store.query_tf_context(
            tf_name=tf_name,
            n_results=3
        )

        if not documents:
            # Fallback to specific pair query if broad context fails (historical support)
            documents = vector_store.query_by_gene_pair(
                gene_a=tf_name,
                gene_b=target_name,
                n_results=3
            )

        if not documents:
            return {
                "match": False,
                "explanation": f"No literature found for {tf_name} context",
                "required_conditions": [],
                "context_found": False,
                "confidence": 0.0
            }

        # Extract conditions from documents
        required_conditions = []
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_conditions = doc.metadata.get('conditions', [])
                required_conditions.extend(doc_conditions)

        required_conditions = list(set(required_conditions))  # Deduplicate

        # Simple condition matching
        if not required_conditions:
            condition_match = "unknown"
            explanation = f"Literature found for {tf_name} -> {target_name} but no specific conditions documented."
        else:
            # Check overlap
            dataset_set = {c.lower() for c in dataset_conditions}
            required_set = {c.lower() for c in required_conditions}
            overlap = dataset_set & required_set

            if len(overlap) >= len(required_set) * 0.7:  # 70% match
                condition_match = "match"
                explanation = f"Dataset conditions match literature requirements. Overlapping conditions: {', '.join(overlap)}"
            elif len(overlap) > 0:
                condition_match = "partial"
                explanation = f"Partial match: dataset has {', '.join(overlap)} but literature requires {', '.join(required_set - dataset_set)}"
            else:
                condition_match = "mismatch"
                explanation = f"Condition mismatch: literature requires {', '.join(required_conditions)} but dataset has {', '.join(dataset_conditions[:3])}"

        return {
            "match": condition_match in ["match", "partial"],
            "explanation": explanation,
            "required_conditions": required_conditions,
            "context_found": True,
            "confidence": min(documents[0].similarity_score, 1.0) if documents else 0.0
        }

    except Exception as e:
        logger.warning(f"Vector store query failed for {tf_name}->{target_name}: {e}")
        return {
            "match": False,
            "explanation": f"Literature lookup error: {str(e)}",
            "required_conditions": [],
            "context_found": False,
            "confidence": 0.0
        }


def format_edge_data_for_llm(edges: List[Dict]) -> str:
    """Format edge data for LLM prompt."""
    formatted = []
    for i, edge in enumerate(edges, 1):
        lit = edge["literature"]
        stats = edge["statistics"]
        ctx = edge["context"]

        formatted.append(f"""
### Edge {i}: {edge['edge_id']}

**Literature:**
- Exists: {lit.get('exists', False)}
- Effect: {lit.get('effect', 'N/A')}
- Evidence: {lit.get('evidence_strength', 'none')}
- Required Conditions: {', '.join(lit.get('conditions_required', ['None']))}

**Statistical Analysis:**
- CLR Z-score: {stats.get('z_score', stats.get('m3d_z_score', 0.0)):.4f}
- Mutual Information: {stats.get('mi', stats.get('m3d_mi', 0.0)):.4f}
- Correlation: {stats.get('correlation', 0.0):.4f}

**Context:**
- Conditions Match: {ctx.get('match', False)}
- Explanation: {ctx.get('explanation', 'No context info')}
""")

    return "\n".join(formatted)


def format_tf_expression(tf_expr: Dict) -> str:
    """Format TF expression profile for prompt."""
    return f"""
- Is Expressed: {tf_expr.get('is_expressed', 'Unknown')}
- Mean Expression: {tf_expr.get('mean_expression', 'N/A')}
- Percentile: {tf_expr.get('percentile', 'N/A')}"""


def invoke_llm_reviewer(edges: List[Dict], tf_bnumber: str, state: AgentState) -> SubgraphReview:
    """Invokes Argonne LLM with structured JSON output via Pydantic parsing."""
    llm = create_argonne_llm()
    parser = JsonOutputParser(pydantic_object=SubgraphReview)

    tf_expr = state.get("analysis__tf_expression", {}).get(tf_bnumber, {})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Reviewer Agent in a gene regulatory network reconciliation system.

Your role: Synthesize statistical evidence with biological literature to decide edge classifications.

Core Principles:
1. Biology is context-dependent: No correlation ≠ No interaction
2. Literature and data complement each other
3. Conflicts often reveal conditional regulation
4. Preserve edges with annotations rather than delete
5. Every decision needs scientific justification

Decision Rules (4-Category System):
- Validated: Strong lit + Strong data (CLR>4.0)
- ConditionSilent: Strong lit + Weak data (CLR<2.0) - true interaction but inactive in dataset conditions
- ProbableFalsePos: Weak lit + No data (CLR<1.0) - likely experimental artifact in literature
- NovelHypothesis: No/weak lit + Strong data (CLR>4.0) - new discovery

Output valid JSON only."""),

        ("human", """Review this transcription factor subgraph:

**TF:** {tf}
**TF Expression:** {tf_expression}

**Edges to Review ({num_edges} total):**
{edge_data}

For EACH edge:
1. Classify using the 4-category system (Validated, ConditionSilent, ProbableFalsePos, NovelHypothesis)
2. Provide 3-5 sentence scientific explanation with specific CLR scores
3. Identify cross-edge patterns

{format_instructions}""")
    ])

    formatted_prompt = prompt.format_messages(
        tf=tf_bnumber,
        tf_expression=format_tf_expression(tf_expr),
        num_edges=len(edges),
        edge_data=format_edge_data_for_llm(edges),
        format_instructions=parser.get_format_instructions()
    )

    response = llm.invoke(formatted_prompt)
    parsed = parser.parse(response.content)
    return SubgraphReview(**parsed)


def apply_rule_based_review(edges: List[Dict], tf_bnumber: str, state: AgentState) -> SubgraphReview:
    """Deterministic decision tree fallback when LLM unavailable."""
    decisions = []

    for edge in edges:
        lit = edge["literature"]
        stats = edge["statistics"]
        ctx = edge["context"]

        status, explanation = apply_decision_tree(lit, stats, ctx)

        decisions.append(EdgeDecisionOutput(
            edge_id=edge["edge_id"],
            reconciliation_status=status,
            preserve_edge=(status != "Unsupported"),
            literature_support=lit.get("evidence_strength", "none"),
            data_support=classify_data_strength(stats.get("clr_zscore", 0.0)),
            context_compatible=ctx.get("match", False),
            explanation=explanation,
            confidence="medium",
            recommendation=""
        ))

    return SubgraphReview(
        tf=tf_bnumber,
        edge_decisions=decisions,
        tf_level_notes="Rule-based fallback used (LLM unavailable)",
        comparative_analysis={}
    )


def apply_decision_tree(lit: Dict, stats: Dict, ctx: Dict) -> Tuple[str, str]:
    """
    4-category decision tree using CLR thresholds (>4.0 strong, <2.0 weak).
    Returns (status, explanation) with 100+ character scientific justification.

    Categories:
    - Validated: Strong literature + High data support
    - ConditionSilent: Strong literature + Low data (context mismatch explains silence)
    - ProbableFalsePos: Weak literature + No data support
    - NovelHypothesis: No literature + High data support
    """
    # Correctly map keys from AnalysisAgent
    clr = stats.get("z_score", stats.get("m3d_z_score", stats.get("clr_zscore", 0.0)))
    mi = stats.get("mi", stats.get("m3d_mi", 0.0))
    lit_exists = lit.get("exists", False)
    lit_strong = lit.get("evidence_strength") in ["strong", "confirmed"]
    lit_weak = lit.get("evidence_strength") == "weak"
    lit_effect = lit.get("effect", "unknown")
    ctx_explanation = ctx.get("explanation", "No context information available")

    # Thresholds (matching reconciler agent)
    HIGH_DATA = 4.0
    MODERATE_DATA = 2.0
    LOW_DATA = 1.0

    if lit_exists and lit_strong:
        # Case A: Strong literature evidence
        if clr >= HIGH_DATA:
            return "Validated", (
                f"This edge shows strong agreement between literature and experimental data. "
                f"Literature reports {lit_effect} with {lit.get('evidence_strength', 'unknown')} evidence, "
                f"and the statistical analysis confirms this with CLR z-score of {clr:.2f} (strong signal) "
                f"and mutual information of {mi:.3f}. This is a confirmed active regulatory relationship."
            )
        elif clr >= MODERATE_DATA:
            return "Validated", (
                f"Strong literature evidence for {lit_effect} regulation with moderate data support "
                f"(CLR={clr:.2f}, MI={mi:.3f}). While not the strongest correlation, "
                f"the combination of solid literature and moderate statistical signal supports this edge."
            )
        else:
            # Case B: Condition-Silent
            return "ConditionSilent", (
                f"Literature supports this edge ({lit_effect}, {lit.get('evidence_strength', 'unknown')} evidence) "
                f"but statistical signal is weak (CLR={clr:.2f}, MI={mi:.3f}). "
                f"Context: {ctx_explanation}. "
                f"This interaction likely requires specific conditions not present in this dataset, "
                f"or involves post-transcriptional regulation not captured in expression data."
            )

    elif lit_exists and lit_weak:
        # Weak literature evidence
        if clr >= HIGH_DATA:
            return "Validated", (
                f"Weak literature evidence upgraded by strong statistical support (CLR={clr:.2f}, MI={mi:.3f}). "
                f"The high-throughput data provides robust evidence for this {lit_effect} regulatory relationship, "
                f"validating the original weak literature claim."
            )
        elif clr < LOW_DATA:
            # Case C: Probable False Positive
            return "ProbableFalsePos", (
                f"Weak literature evidence combined with no statistical support (CLR={clr:.2f}, MI={mi:.3f}). "
                f"This edge likely represents a false positive from the original literature study, "
                f"possibly due to experimental artifacts or non-physiological conditions. "
                f"Consider removing from the network or flagging for re-validation."
            )
        else:
            return "ConditionSilent", (
                f"Weak literature evidence with marginal data support (CLR={clr:.2f}, MI={mi:.3f}). "
                f"The relationship may exist but be condition-dependent or indirect."
            )

    else:
        # No literature evidence
        if clr >= HIGH_DATA:
            # Case D: Novel Hypothesis
            return "NovelHypothesis", (
                f"Strong statistical evidence (CLR={clr:.2f}, MI={mi:.3f}) indicates a regulatory relationship "
                f"with no supporting literature documentation. This represents a potential novel discovery "
                f"that has not been characterized in existing databases. The high correlation suggests direct "
                f"or strong indirect regulation worth experimental validation via ChIP-seq or reporter assays."
            )
        else:
            # Not enough evidence from either source - treat as ConditionSilent
            return "ConditionSilent", (
                f"No literature evidence and weak statistical signal (CLR={clr:.2f}, MI={mi:.3f}). "
                f"Insufficient evidence for classification."
            )


def classify_data_strength(clr: float) -> str:
    """CLR thresholds matching 4-category system."""
    if clr >= 4.0:
        return "strong"
    elif clr >= 2.0:
        return "moderate"
    elif clr >= 1.0:
        return "weak"
    else:
        return "none"


def post_process_decisions(
    review: SubgraphReview,
    state: AgentState,
    tf_bnumber: str,
    method: str,
    original_edges: List[Dict] = []
) -> Dict[str, Any]:
    """
    Aggregates decisions, identifies zombies/novel hypotheses, updates state.

    Uses 4-category system: Validated, ConditionSilent, ProbableFalsePos, NovelHypothesis
    """
    edge_decisions = [d.model_dump() for d in review.edge_decisions]

    # Get existing logs
    reconciliation_log = state.get("reconciliation_log", []).copy()
    false_positive_candidates = state.get("false_positive_candidates", []).copy()
    novel_hypotheses = state.get("novel_hypotheses", []).copy()
    bnumber_to_name = state.get("bnumber_to_gene_name", {})

    # Map original edges by edge_id for robust lookup
    stats_lookup = {e['edge_id']: e['statistics'] for e in original_edges}
    bnumber_lookup = {e['edge_id']: e['target'] for e in original_edges}
    
    # Process each edge decision
    for decision in edge_decisions:
        # Convert to standard reconciliation log format
        tf_name = bnumber_to_name.get(tf_bnumber, tf_bnumber)
        
        # Robust edge_id parsing
        eid = decision["edge_id"]
        if "→" in eid:
            target = eid.split("→")[1]
        elif "->" in eid:
            target = eid.split("->")[1]
        else:
            # Fallback: Check if it matches "Edge N" pattern
            if eid.lower().startswith("edge"):
                try:
                    # Extract number
                    parts = eid.replace(":", "").split()
                    for p in parts:
                         if p.isdigit():
                             idx = int(p) - 1 # 1-based to 0-based
                             if 0 <= idx < len(original_edges):
                                 target = original_edges[idx]["target"]
                                 logger.info(f"Resolved '{eid}' to target {target} using index {idx}")
                                 break
                except Exception as e:
                    logger.warning(f"Failed to parse index from {eid}: {e}")

            # If still not found, try space separation fallback
            if 'target' not in locals():
                parts = eid.split()
                if len(parts) >= 2:
                    target = parts[1] # "TF Target"
                else:
                    target = "unknown_target"
                    logger.warning(f"Could not parse target from edge_id: {eid}")
        
        target = target.strip() # Remove any whitespace

        # Resolve target bnumber from edge_id map (safer than splitting)
        target_bnumber = bnumber_lookup.get(decision["edge_id"])
        if not target_bnumber:
             # Fallback if LLM hallucinated an ID, try splitting but be careful
             parts = decision["edge_id"].split("_")
             target_bnumber = parts[1] if len(parts) > 1 else decision["edge_id"]
        
        target_name = bnumber_to_name.get(target_bnumber, target_bnumber)
        
        # Retrieve stats using unique edge_id
        edge_stats = stats_lookup.get(decision["edge_id"], {})
        
        # Handle key mismatch (AnalysisAgent uses 'z_score', Reviewer defaults used 'm3d_z_score')
        z_score = edge_stats.get("z_score", edge_stats.get("m3d_z_score", 0.0))
        mi_score = edge_stats.get("mi", edge_stats.get("m3d_mi", 0.0))

        log_entry = {
            "source_tf": tf_bnumber,
            "target_gene": target_bnumber,
            "source_tf_name": tf_name,
            "target_gene_name": target_name,
            "regulondb_evidence": decision.get("literature_support", "unknown"),
            "regulondb_effect": "?",  # Not tracked in reviewer output
            "m3d_z_score": z_score,
            "m3d_mi": mi_score,
            "reconciliation_status": decision["reconciliation_status"],
            "context_tags": [],  # Could extract from context
            "notes": f"[{method.upper()}] {decision['explanation']}"
        }

        reconciliation_log.append(log_entry)

        # Add to category-specific lists
        if decision["reconciliation_status"] == "ProbableFalsePos":
            false_positive_candidates.append(log_entry)
        elif decision["reconciliation_status"] == "NovelHypothesis":
            novel_hypotheses.append(log_entry)

    summary = generate_summary_message(review, method)
    logger.info(summary)

    return {
        "reconciliation_log": reconciliation_log,
        "false_positive_candidates": false_positive_candidates,
        "novel_hypotheses": novel_hypotheses
    }


def generate_summary_message(review: SubgraphReview, method: str) -> str:
    """Generates human-readable summary with status breakdown."""
    status_counts = {}
    for d in review.edge_decisions:
        status = d.reconciliation_status
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = f"""Reviewer Agent: Completed review of TF '{review.tf}' using {method.upper()} method

Edges Reviewed: {len(review.edge_decisions)}

Status Breakdown:
"""
    for status, count in sorted(status_counts.items()):
        summary += f"  - {status}: {count}\n"

    summary += f"\nTF-Level Notes:\n{review.tf_level_notes}\n"

    if review.comparative_analysis:
        summary += f"\nComparative Analysis:\n{review.comparative_analysis}\n"

    return summary
