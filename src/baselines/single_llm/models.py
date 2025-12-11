from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class ReconciliationLogEntry(BaseModel):
    """
    Represents a single reconciled edge.
    Matches the schema expected by evaluation.metrics.metric_a_sabotage.
    """
    source_tf_name: str = Field(description="The Transcription Factor (Regulator)")
    target_gene_name: str = Field(description="The Target Gene")
    reconciliation_status: Literal[
        "Validated", 
        "NovelHypothesis", 
        "ConditionSilent", 
        "ProbableFalsePos", 
        # "FalsePositive" is mapped to ProbableFalsePos usually, but keeping compatible types
        "FalsePositive",
        "Unknown"
    ] = Field(description="The final verdict on the edge")
    confidence_score: float = Field(description="Confidence in the verdict (0-1)")
    reasoning: str = Field(description="Scientific explanation for the verdict")
    
    # z-score is hidden from LLM input but required for report
    m3d_z_score: float = Field(default=0.0, description="Placeholder for Z-score")

class SingleLLMResult(BaseModel):
    """Container for the full batch results."""
    reconciliation_log: List[ReconciliationLogEntry]
