"""Pydantic models for Research Agent input/output."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata about the experimental dataset."""

    conditions: List[str] = Field(
        description="Experimental conditions (e.g., ['aerobic', '37C', 'rich_media'])"
    )
    treatment: str = Field(description="Treatment applied (e.g., 'heat_shock', 'none')")
    growth_phase: str = Field(description="Growth phase (e.g., 'exponential', 'stationary')")
    time_point: Optional[str] = Field(default=None, description="Time point (e.g., '0h', '2h')")
    media: Optional[str] = Field(default=None, description="Growth media type")
    strain: Optional[str] = Field(default=None, description="Organism strain")


class StatisticalSummary(BaseModel):
    """Statistical summary from Analysis Agent."""

    correlation: Optional[float] = Field(default=None, description="Correlation coefficient")
    mutual_information: Optional[float] = Field(default=None, description="Mutual information score")
    p_value: Optional[float] = Field(default=None, description="Statistical p-value")


class ResearchAgentInput(BaseModel):
    """Input to the Research Agent."""

    gene_a: str = Field(description="First gene identifier")
    gene_b: str = Field(description="Second gene identifier (potential target)")
    dataset_metadata: DatasetMetadata = Field(description="Dataset experimental metadata")
    statistical_summary: Optional[StatisticalSummary] = Field(
        default=None,
        description="Optional statistical summary from Analysis Agent"
    )


class SupportingDocument(BaseModel):
    """A supporting document from literature retrieval."""

    doc_id: str = Field(description="Document identifier")
    source: str = Field(description="Source (e.g., 'RegulonDB', 'PubMed:PMID')")
    text: str = Field(description="Document text")
    similarity_score: float = Field(description="Similarity score from retrieval")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ResearchAgentOutput(BaseModel):
    """Output from the Research Agent."""

    gene_pair: str = Field(description="Gene pair (e.g., 'lexA -> recA')")
    context_found: bool = Field(description="Whether relevant literature exists")
    regulatory_relationship: str = Field(
        description="Type of relationship: 'activation', 'repression', 'no_interaction', 'unknown'"
    )
    required_conditions: List[str] = Field(description="Conditions needed for interaction")
    dataset_conditions: List[str] = Field(description="Actual dataset conditions")
    condition_match: str = Field(
        description="Condition match status: 'match', 'mismatch', 'partial', 'unknown'"
    )
    explanation: str = Field(description="Natural language explanation of findings")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0) based on retrieval quality"
    )
    supporting_documents: List[SupportingDocument] = Field(
        description="Retrieved source documents"
    )
    reasoning_trace: List[str] = Field(description="Step-by-step reasoning for transparency")


class LiteratureDocument(BaseModel):
    """A literature document to be stored in the vector database."""

    doc_id: str = Field(description="Unique document identifier")
    gene_a: str = Field(description="Regulator gene")
    gene_b: str = Field(description="Target gene")
    interaction_type: str = Field(
        description="Interaction type: 'activation', 'repression', 'binding', etc."
    )
    conditions: List[str] = Field(description="Required/observed conditions for interaction")
    evidence: str = Field(description="Evidence type (e.g., 'ChIP-seq', 'RNA-seq', 'literature')")
    source: str = Field(description="Source identifier (e.g., 'RegulonDB', 'PubMed:12345678')")
    text: str = Field(description="Full text description for embedding")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
