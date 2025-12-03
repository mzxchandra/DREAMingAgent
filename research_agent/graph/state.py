"""LangGraph state definition for Research Agent."""

from typing import List, Optional, TypedDict

from ..models import (
    DatasetMetadata,
    StatisticalSummary,
    SupportingDocument
)


class ResearchAgentState(TypedDict, total=False):
    """
    State dictionary for Research Agent LangGraph workflow.

    This state is passed between nodes and updated throughout execution.
    """

    # Input fields
    gene_a: str
    gene_b: str
    dataset_metadata: DatasetMetadata
    statistical_summary: Optional[StatisticalSummary]

    # Query formulation outputs
    search_query: str
    query_embedding: List[float]

    # Vector retrieval outputs
    retrieved_documents: List[SupportingDocument]
    retrieval_count: int

    # Context extraction outputs
    context_found: bool
    regulatory_relationship: str  # 'activation', 'repression', 'no_interaction', 'unknown'
    required_conditions: List[str]
    literature_summary: str

    # Condition matching outputs
    dataset_conditions: List[str]
    condition_match: str  # 'match', 'mismatch', 'partial', 'unknown'
    condition_analysis: str

    # Explanation generation outputs
    explanation: str
    confidence: float

    # Output formatting
    final_output: dict

    # Reasoning trace for transparency
    reasoning_trace: List[str]

    # Error handling
    error: Optional[str]
