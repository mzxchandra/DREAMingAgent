"""
AgentState: The shared state schema for the LangGraph workflow.

This TypedDict serves as the "short-term memory" passed between nodes,
containing both static knowledge bases and dynamic processing state.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import pandas as pd
import networkx as nx


# ============================================================================
# Evidence Classification
# ============================================================================

EvidenceLevel = Literal["Strong", "Weak", "Confirmed", "Unknown"]
InteractionEffect = Literal["+", "-", "+-", "?"]
ReconciliationStatus = Literal[
    "Validated",           # Strong lit + High data signal
    "ConditionSilent",     # Strong lit + Low data signal (context gap)
    "ProbableFalsePos",    # Weak lit + Low data signal
    "NovelHypothesis",     # No lit + High data signal
    "Pending"              # Not yet processed
]


# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class EdgeAnalysis:
    """Statistical analysis result for a single TF->Gene edge."""
    source_tf: str
    target_gene: str
    mutual_information: float
    z_score: float
    p_value: Optional[float] = None
    significance: Literal["High", "Moderate", "Low", "Insignificant"] = "Insignificant"
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tf": self.source_tf,
            "target_gene": self.target_gene,
            "mi": self.mutual_information,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "significance": self.significance,
            "sample_count": self.sample_count
        }


@dataclass 
class ReconciliationResult:
    """Result of reconciling literature assertion with data evidence."""
    source_tf: str
    target_gene: str
    literature_evidence: EvidenceLevel
    literature_effect: InteractionEffect
    data_z_score: float
    data_mi: float
    status: ReconciliationStatus
    context_tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_tf": self.source_tf,
            "target_gene": self.target_gene,
            "regulondb_evidence": self.literature_evidence,
            "regulondb_effect": self.literature_effect,
            "m3d_z_score": self.data_z_score,
            "m3d_mi": self.data_mi,
            "reconciliation_status": self.status,
            "context_tags": self.context_tags,
            "notes": self.notes
        }


# ============================================================================
# Main Agent State
# ============================================================================

class AgentState(TypedDict):
    """
    The shared state schema passed between LangGraph nodes.
    
    This TypedDict contains:
    - STATIC KNOWLEDGE: Loaded once from RegulonDB and M3D
    - DYNAMIC STATE: Updated per iteration/batch
    - OUTPUT: Accumulated results
    """
    
    # --- STATIC KNOWLEDGE (Loaded by Loader Node) ---
    
    # The Literature Prior (RegulonDB) as a NetworkX DiGraph
    # Nodes: Genes/TFs (using b-numbers)
    # Edges: Attributes {'evidence': 'Strong', 'effect': '+', 'tf_name': 'FNR', 'gene_name': 'narG'}
    literature_graph: nx.DiGraph
    
    # Gene name to b-number mapping (Rosetta Stone)
    # e.g., {"fnr": "b1334", "ompF": "b0929", ...}
    gene_name_to_bnumber: Dict[str, str]
    
    # Reverse mapping: b-number to gene name
    bnumber_to_gene_name: Dict[str, str]
    
    # The Data Landscape (M3D) - Expression Matrix
    # Rows: Genes (indexed by b-number), Columns: Experiments
    expression_matrix: pd.DataFrame
    
    # M3D Experimental Metadata
    # Rows: Experiments, Columns: Condition attributes
    metadata: pd.DataFrame
    
    # --- DYNAMIC STATE (Per Iteration) ---
    
    # Queue of TFs to process (as b-numbers)
    tf_queue: List[str]
    
    # Current batch of TFs being processed
    current_batch_tfs: List[str]
    
    # The subset of sample IDs relevant to the current TF context
    # Populated by the Context Agent
    active_sample_indices: List[str]
    
    # Current context description (e.g., "Anaerobic conditions")
    current_context: str
    
    # The statistical results from the Analysis Agent
    # Structure: {tf_bnumber: {target_bnumber: EdgeAnalysis.to_dict(), ...}, ...}
    analysis_results: Dict[str, Dict[str, Dict[str, Any]]]
    
    # --- OUTPUT (Accumulated Results) ---
    
    # The final reconciliation log
    reconciliation_log: List[Dict[str, Any]]
    
    # Registry of novel hypotheses (high data signal, no literature)
    novel_hypotheses: List[Dict[str, Any]]
    
    # Registry of probable false positives (weak lit, no data support)
    false_positive_candidates: List[Dict[str, Any]]
    
    # Iteration tracking
    iteration_count: int
    max_iterations: int
    
    # Error log
    errors: List[str]
    
    # Processing status
    status: Literal["initializing", "processing", "completed", "error"]


def create_initial_state() -> AgentState:
    """Create an empty initial state for the workflow."""
    return AgentState(
        literature_graph=nx.DiGraph(),
        gene_name_to_bnumber={},
        bnumber_to_gene_name={},
        expression_matrix=pd.DataFrame(),
        metadata=pd.DataFrame(),
        tf_queue=[],
        current_batch_tfs=[],
        active_sample_indices=[],
        current_context="",
        analysis_results={},
        reconciliation_log=[],
        novel_hypotheses=[],
        false_positive_candidates=[],
        iteration_count=0,
        max_iterations=100,
        errors=[],
        status="initializing"
    )


