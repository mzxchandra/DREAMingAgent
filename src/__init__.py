"""

A LangGraph-based multi-agent system for reconciling curated literature knowledge (RegulonDB)
with high-throughput expression data (M3D) to produce context-aware gene regulatory networks.
"""

from .state import (
    AgentState,
    create_initial_state,
    EdgeAnalysis,
    EdgeDecision,
    ContextAnnotation,
    EvidenceLevel,
    InteractionEffect,
    ReconciliationStatus,
    DataSignificance,
)

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "create_initial_state",
    "EdgeAnalysis",
    "EdgeDecision",
    "ContextAnnotation",
    "EvidenceLevel",
    "InteractionEffect",
    "ReconciliationStatus",
    "DataSignificance",
]
DREAMing Agent: Agentic Reconciliation of Biological Literature and High-Throughput Data

This system reconciles gene regulatory network knowledge from RegulonDB (literature prior)
with high-throughput expression data from M3D (data landscape) using LangGraph.
"""

__version__ = "0.1.0"
__author__ = "DREAMing Agent Team"
