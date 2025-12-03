"""
DREAMing Agent - Agentic Reconciliation of Biological Literature and High-Throughput Data
=========================================================================================

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
