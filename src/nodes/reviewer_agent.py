"""
Reviewer Agent: Synthesizes statistical evidence with biological literature
to produce context-aware gene regulatory network annotations.

Hybrid LLM + rule-based decision system with graceful fallback.
"""

from typing import Dict, List, Any, Tuple
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.state import AgentState
from src.llm_config import create_argonne_llm


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
    """
    messages = state["messages"]
    current_tfs = state.get("batch__current_tfs", [])

    if not current_tfs:
        return {
            "messages": messages + [AIMessage(content="Reviewer: No TFs to process")],
            "reviewer__edge_decisions": []
        }

    tf_bnumber = current_tfs[0]
    edges_data = prepare_subgraph_data(state, tf_bnumber)

    if not edges_data:
        return {
            "messages": messages + [AIMessage(content=f"Reviewer: No edges found for TF {tf_bnumber}")],
            "reviewer__edge_decisions": []
        }

    try:
        review_result = invoke_llm_reviewer(edges_data, tf_bnumber, state)
        method = "llm"
    except Exception as e:
        print(f"LLM failed ({e}), using rule-based fallback")
        review_result = apply_rule_based_review(edges_data, tf_bnumber, state)
        method = "rules"

    return post_process_decisions(review_result, state, tf_bnumber, method, messages)


def prepare_subgraph_data(state: AgentState, tf_bnumber: str) -> List[Dict[str, Any]]:
    """Combines literature, statistics, and context for all edges of one TF."""
    edges = []

    lit_edges = state.get("research__literature_edges", {}).get(tf_bnumber, {})
    stat_results = state.get("analysis__statistical_results", {}).get(tf_bnumber, {})

    all_targets = set(lit_edges.keys()) | set(stat_results.keys())

    for target_bnumber in all_targets:
        edge_id = f"{tf_bnumber}→{target_bnumber}"

        lit_data = lit_edges.get(target_bnumber, {"exists": False})
        stat_data = stat_results.get(target_bnumber, {
            "clr_zscore": 0.0,
            "mi": 0.0,
            "correlation": 0.0
        })
        context_data = state.get("research__annotations", {}).get(edge_id, {
            "match": False,
            "explanation": "No context information available"
        })

        edges.append({
            "edge_id": edge_id,
            "tf": tf_bnumber,
            "target": target_bnumber,
            "literature": lit_data,
            "statistics": stat_data,
            "context": context_data
        })

    return edges


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
- CLR Z-score: {stats.get('clr_zscore', 0.0):.4f}
- Mutual Information: {stats.get('mi', 0.0):.4f}
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

Decision Rules:
- Validated: Strong lit + Strong data (CLR>3) + Context match
- Active: Any lit + Strong data (CLR>3) + Context mismatch
- Condition-Silent: Strong lit + Weak data (CLR<2) + Context MISMATCH
- Novel Hypothesis: No/weak lit + Strong data (CLR>3)
- Probable False Positive: Weak lit + No data (CLR<1) + Context MATCHES
- Unsupported: No lit + No data

Output valid JSON only."""),

        ("human", """Review this transcription factor subgraph:

**TF:** {tf}
**TF Expression:** {tf_expression}

**Edges to Review ({num_edges} total):**
{edge_data}

For EACH edge:
1. Classify using the 7-category system
2. Provide 3-5 sentence scientific explanation with specific CLR scores and context details
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
    7-category decision tree using CLR thresholds (>3.0 strong, <2.0 weak).
    Returns (status, explanation) with 100+ character scientific justification.
    """
    clr = stats.get("clr_zscore", 0.0)
    mi = stats.get("mi", 0.0)
    lit_exists = lit.get("exists", False)
    lit_strong = lit.get("evidence_strength") == "strong"
    lit_effect = lit.get("effect", "unknown")
    ctx_match = ctx.get("match", False)
    ctx_explanation = ctx.get("explanation", "No context information available")

    if lit_exists:
        if clr > 3.0:
            if ctx_match:
                return "Validated", (
                    f"This edge shows strong agreement between literature and experimental data. "
                    f"Literature reports {lit_effect} with {lit.get('evidence_strength', 'unknown')} evidence, "
                    f"and the statistical analysis confirms this with CLR z-score of {clr:.2f} (strong signal) "
                    f"and mutual information of {mi:.3f}. Context conditions match between literature and dataset."
                )
            else:
                return "Active", (
                    f"Strong statistical signal (CLR={clr:.2f}, MI={mi:.3f}) supports this regulatory relationship "
                    f"despite context mismatch with literature. Literature reports {lit_effect}, "
                    f"but experimental conditions differ. Context note: {ctx_explanation}. "
                    f"The edge is likely active but may exhibit different behavior than literature expectations."
                )
        elif clr < 2.0:
            if not ctx_match:
                return "Condition-Silent", (
                    f"Literature supports this edge ({lit_effect}, {lit.get('evidence_strength', 'unknown')} evidence) "
                    f"but statistical signal is weak (CLR={clr:.2f}, MI={mi:.3f}). "
                    f"The low correlation is explained by context mismatch: {ctx_explanation}. "
                    f"This interaction likely requires specific conditions not present in this dataset."
                )
            elif lit_strong:
                return "Condition-Silent", (
                    f"Strong literature evidence for {lit_effect} regulation, but weak statistical signal "
                    f"(CLR={clr:.2f}, MI={mi:.3f}) despite matching contexts. "
                    f"This suggests the interaction may require additional specific conditions, "
                    f"post-transcriptional regulation, or indirect effects not captured in expression data alone."
                )
            else:
                return "Probable False Positive", (
                    f"Weak literature evidence combined with no statistical support (CLR={clr:.2f}, MI={mi:.3f}) "
                    f"despite context match. Context: {ctx_explanation}. "
                    f"This edge likely represents a false positive from the original literature study. "
                    f"Consider removing from the network or flagging for experimental validation."
                )
        else:
            return "Condition-Silent", (
                f"Literature supports this regulatory relationship ({lit_effect}), "
                f"and moderate statistical signal (CLR={clr:.2f}, MI={mi:.3f}) suggests partial activity. "
                f"This borderline case may indicate condition-dependent regulation or weak regulatory effects. "
                f"Context match status: {ctx_match}."
            )
    else:
        if clr > 3.0:
            return "Novel Hypothesis", (
                f"Strong statistical evidence (CLR={clr:.2f}, MI={mi:.3f}) indicates a regulatory relationship "
                f"with no supporting literature documentation. This represents a potential novel discovery "
                f"that has not been characterized in existing databases. The high correlation suggests direct "
                f"or strong indirect regulation worth experimental validation via ChIP-seq or reporter assays."
            )
        else:
            return "Unsupported", (
                f"No literature evidence exists for this edge, and statistical analysis shows weak signal "
                f"(CLR={clr:.2f}, MI={mi:.3f}). Without support from either source, "
                f"this edge lacks sufficient evidence for inclusion in the regulatory network. "
                f"May represent noise or very weak indirect effects below detection threshold."
            )


def classify_data_strength(clr: float) -> str:
    """CLR thresholds from architecture."""
    if clr > 3.0:
        return "strong"
    elif clr > 1.0:
        return "weak"
    else:
        return "none"


def post_process_decisions(
    review: SubgraphReview,
    state: AgentState,
    tf_bnumber: str,
    method: str,
    messages: List
) -> Dict[str, Any]:
    """Aggregates decisions, identifies zombies/novel hypotheses, updates state."""
    edge_decisions = [d.model_dump() for d in review.edge_decisions]

    zombie_candidates = [
        d for d in edge_decisions
        if d["reconciliation_status"] == "Probable False Positive"
    ]

    novel_hypotheses = [
        d for d in edge_decisions
        if d["reconciliation_status"] == "Novel Hypothesis"
    ]

    summary = generate_summary_message(review, method)

    return {
        "messages": messages + [AIMessage(content=summary)],
        "reviewer__edge_decisions": edge_decisions,
        "reviewer__tf_summaries": {tf_bnumber: review.tf_level_notes},
        "reviewer__comparative_analysis": {tf_bnumber: review.comparative_analysis},
        "reviewer__reconciliation_log": state.get("reviewer__reconciliation_log", []) + edge_decisions,
        "reviewer__zombie_candidates": state.get("reviewer__zombie_candidates", []) + zombie_candidates,
        "reviewer__novel_hypotheses": state.get("reviewer__novel_hypotheses", []) + novel_hypotheses
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
