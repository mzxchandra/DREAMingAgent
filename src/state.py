"""
State Definition for DREAMing Agent Multi-Agent Workflow

Design Pattern: NAMESPACED FLAT STATE
- Fields use {agent_name}__{field_name} convention to show ownership
- Global fields (no prefix) for workflow control
- Maintains LangGraph compatibility while improving clarity

Data Flow:
  Loader → Batch Manager → Research Agent → Analysis Agent → Reviewer Agent → (loop)
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage
import operator
import pandas as pd
import networkx as nx


EvidenceLevel = Literal["strong", "weak", "confirmed", "none"]
InteractionEffect = Literal["activation", "repression", "dual", "unknown"]
ReconciliationStatus = Literal[
    "Validated",
    "Active",
    "Condition-Silent",
    "Inactive-Context-Mismatch",
    "Novel Hypothesis",
    "Probable False Positive",
    "Unsupported",
]
DataSignificance = Literal["strong", "weak", "none"]

@dataclass
class EdgeAnalysis:
    """Statistical analysis result for a single TF→Gene edge."""
    source_tf: str
    target_gene: str
    mutual_information: float
    clr_zscore: float
    correlation: float
    pvalue: Optional[float] = None
    significance: DataSignificance = "none"
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tf": self.source_tf,
            "target_gene": self.target_gene,
            "mi": self.mutual_information,
            "clr_zscore": self.clr_zscore,
            "correlation": self.correlation,
            "pvalue": self.pvalue,
            "significance": self.significance,
            "sample_count": self.sample_count
        }


@dataclass
class EdgeDecision:
    """Reconciliation decision for a single TF→Gene edge."""
    edge_id: str
    source_tf: str
    target_gene: str
    reconciliation_status: ReconciliationStatus
    preserve_edge: bool
    literature_support: EvidenceLevel
    literature_effect: InteractionEffect
    data_support: DataSignificance
    context_compatible: bool
    explanation: str
    confidence: Literal["high", "medium", "low"] = "medium"
    recommendation: Optional[str] = None
    context_tags: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_tf": self.source_tf,
            "target_gene": self.target_gene,
            "reconciliation_status": self.reconciliation_status,
            "preserve_edge": self.preserve_edge,
            "literature_support": self.literature_support,
            "literature_effect": self.literature_effect,
            "data_support": self.data_support,
            "context_compatible": self.context_compatible,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "context_tags": self.context_tags,
            "flags": self.flags
        }


@dataclass
class ContextAnnotation:
    """Context matching result for a single edge."""
    edge_id: str
    literature_conditions: List[str]
    dataset_conditions: List[str]
    match: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "literature_conditions": self.literature_conditions,
            "dataset_conditions": self.dataset_conditions,
            "match": self.match,
            "explanation": self.explanation
        }


class AgentState(TypedDict):
    """Shared state for the LangGraph workflow using namespaced flat design."""

    messages: Annotated[List[BaseMessage], operator.add]
    status: Literal["initializing", "loading", "processing", "completed", "error"]
    iteration_count: int
    max_iterations: int
    errors: List[str]

    loader__literature_graph: nx.DiGraph
    loader__gene_name_to_bnumber: Dict[str, str]
    loader__bnumber_to_gene_name: Dict[str, str]
    loader__expression_matrix: pd.DataFrame
    loader__metadata: pd.DataFrame

    batch__tf_queue: List[str]
    batch__current_tfs: List[str]
    batch__complete: bool

    research__filtered_samples: List[str]
    research__current_context_description: str
    research__dataset_conditions: List[str]
    research__literature_edges: Dict[str, Dict[str, Dict[str, Any]]]
    research__annotations: Dict[str, Dict[str, Any]]

    analysis__statistical_results: Dict[str, Dict[str, Dict[str, Any]]]
    analysis__tf_expression: Dict[str, Dict[str, Any]]

    reviewer__edge_decisions: List[Dict[str, Any]]
    reviewer__tf_summaries: Dict[str, str]
    reviewer__comparative_analysis: Dict[str, Dict[str, Any]]
    reviewer__reconciliation_log: List[Dict[str, Any]]
    reviewer__zombie_candidates: List[Dict[str, Any]]
    reviewer__novel_hypotheses: List[Dict[str, Any]]
    reviewer__edge_replication_history: Dict[str, List[Dict[str, Any]]]


def create_initial_state() -> AgentState:
    """Create an empty initial state for the workflow."""
    return AgentState(
        messages=[],
        status="initializing",
        iteration_count=0,
        max_iterations=100,
        errors=[],
        loader__literature_graph=nx.DiGraph(),
        loader__gene_name_to_bnumber={},
        loader__bnumber_to_gene_name={},
        loader__expression_matrix=pd.DataFrame(),
        loader__metadata=pd.DataFrame(),
        batch__tf_queue=[],
        batch__current_tfs=[],
        batch__complete=False,
        research__filtered_samples=[],
        research__current_context_description="",
        research__dataset_conditions=[],
        research__literature_edges={},
        research__annotations={},
        analysis__statistical_results={},
        analysis__tf_expression={},
        reviewer__edge_decisions=[],
        reviewer__tf_summaries={},
        reviewer__comparative_analysis={},
        reviewer__reconciliation_log=[],
        reviewer__zombie_candidates=[],
        reviewer__novel_hypotheses=[],
        reviewer__edge_replication_history={}
    )


def get_current_tf(state: AgentState) -> Optional[str]:
    """Get the first TF from current batch."""
    current_tfs = state.get("batch__current_tfs", [])
    return current_tfs[0] if current_tfs else None


def get_literature_edge(state: AgentState, tf_bnumber: str, target_bnumber: str) -> Optional[Dict[str, Any]]:
    """Lookup a literature edge from the research agent's indexed data."""
    lit_edges = state.get("research__literature_edges", {})
    return lit_edges.get(tf_bnumber, {}).get(target_bnumber, None)


def get_statistical_result(state: AgentState, tf_bnumber: str, target_bnumber: str) -> Optional[Dict[str, Any]]:
    """Lookup statistical analysis result for an edge."""
    stats = state.get("analysis__statistical_results", {})
    return stats.get(tf_bnumber, {}).get(target_bnumber, None)


def get_research_annotation(state: AgentState, edge_id: str) -> Optional[Dict[str, Any]]:
    """Lookup research annotation for an edge."""
    annotations = state.get("research__annotations", {})
    return annotations.get(edge_id, None)


def validate_state(state: AgentState) -> List[str]:
    """Validate state integrity and return list of issues."""
    errors = []

    if state["status"] not in ["initializing", "loading"]:
        if state["loader__literature_graph"].number_of_nodes() == 0:
            errors.append("Literature graph is empty but status is not initializing")
        if state["loader__expression_matrix"].empty:
            errors.append("Expression matrix is empty but status is not initializing")

    if state["batch__current_tfs"] and not state["batch__tf_queue"]:
        if not state["batch__complete"]:
            errors.append("Current batch set but queue empty and not complete")

    current_tfs = state.get("batch__current_tfs", [])
    analysis_tfs = set(state.get("analysis__statistical_results", {}).keys())
    if current_tfs and not analysis_tfs.intersection(current_tfs):
        errors.append(f"Analysis results don't include current TFs: {current_tfs}")

    return errors


__all__ = [
    "EvidenceLevel",
    "InteractionEffect",
    "ReconciliationStatus",
    "DataSignificance",
    "EdgeAnalysis",
    "EdgeDecision",
    "ContextAnnotation",
    "AgentState",
    "create_initial_state",
    "get_current_tf",
    "get_literature_edge",
    "get_statistical_result",
    "get_research_annotation",
    "validate_state",
]
