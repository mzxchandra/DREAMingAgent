"""
Research Agent Node: Literature-aware hypothesis generation for gene regulatory networks.

This module provides a complete Research Agent that retrieves and contextualizes
literature evidence for gene pairs. It integrates into the main workflow as a
single node while internally running a 6-step LangGraph workflow.

Architecture:
    - External Interface: research_agent_node(state: AgentState) -> Dict[str, Any]
    - Internal Workflow: 6-node LangGraph (query_formulation, vector_retrieval,
      context_extraction, condition_matching, explanation_generation, output_formatting)
"""

import re
from typing import Dict, Any, List, Optional, Tuple, TypedDict
import pandas as pd
import networkx as nx
from pydantic import BaseModel, Field
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("langgraph not installed. Run: pip install langgraph>=0.0.20")

from src.state import AgentState
from src.alcf_config import get_alcf_config
from src.llm.alcf_client import get_alcf_client
from src.utils.vector_store import get_vector_store, SupportingDocument


# ============================================================================
# Pydantic Models (inline)
# ============================================================================

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


# ============================================================================
# Internal LangGraph State
# ============================================================================

class ResearchAgentState(TypedDict, total=False):
    """
    State dictionary for Research Agent internal LangGraph workflow.

    This state is passed between the 6 internal nodes and updated throughout execution.
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


# ============================================================================
# LLM Prompts (inline)
# ============================================================================

QUERY_FORMULATION_PROMPT = """You are a biological research expert analyzing gene regulatory relationships.

Given the following information:
- Gene A (potential regulator): {gene_a}
- Gene B (potential target): {gene_b}
- Dataset conditions: {dataset_conditions}

Create an optimized search query to find relevant literature about the regulatory relationship between {gene_a} and {gene_b}.

Your query should:
1. Focus on regulatory interactions, transcriptional regulation, or gene expression control
2. Include both gene names
3. Be concise but comprehensive (1-2 sentences)
4. Consider the biological context

Return ONLY the search query text, nothing else."""


CONTEXT_EXTRACTION_PROMPT = """You are analyzing scientific literature to extract regulatory relationships between genes.

Gene pair: {gene_a} -> {gene_b}

Retrieved documents:
{documents}

Based on these documents, extract:
1. Whether relevant context exists (yes/no)
2. The regulatory relationship type: 'activation', 'repression', 'no_interaction', or 'unknown'
3. Required conditions for the interaction (list)
4. A brief summary of the literature findings

Respond in the following format:
CONTEXT_FOUND: yes/no
REGULATORY_RELATIONSHIP: activation/repression/no_interaction/unknown
REQUIRED_CONDITIONS: condition1, condition2, condition3
SUMMARY: Brief summary of findings (2-3 sentences)"""


CONDITION_MATCHING_PROMPT = """You are comparing experimental conditions with literature-reported conditions for gene regulation.

Gene pair: {gene_a} -> {gene_b}
Regulatory relationship: {regulatory_relationship}

Required conditions from literature:
{required_conditions}

Actual dataset conditions:
{dataset_conditions}

Statistical evidence:
{statistical_summary}

Analyze whether the dataset conditions match the required conditions for this regulatory interaction.

Consider:
1. Direct matches (e.g., both mention "anaerobic")
2. Semantic matches (e.g., "oxygen_depletion" matches "anaerobic")
3. Contradictions (e.g., dataset has "aerobic" but literature requires "anaerobic")
4. Missing information (e.g., literature mentions "heat_shock" but dataset doesn't specify temperature stress)

Respond in the following format:
CONDITION_MATCH: match/mismatch/partial/unknown
ANALYSIS: Detailed explanation of condition comparison (3-4 sentences)"""


EXPLANATION_GENERATION_PROMPT = """You are generating a scientific explanation for gene regulatory analysis results.

Gene pair: {gene_a} -> {gene_b}

Context found: {context_found}
Regulatory relationship: {regulatory_relationship}
Required conditions: {required_conditions}
Dataset conditions: {dataset_conditions}
Condition match: {condition_match}

Literature summary:
{literature_summary}

Condition analysis:
{condition_analysis}

Statistical evidence:
{statistical_summary}

Generate a clear, scientifically accurate explanation that:
1. States whether the regulatory relationship is expected in this dataset
2. Explains why or why not based on condition matching
3. Mentions statistical evidence if relevant
4. Uses precise biological terminology
5. Is 3-5 sentences long

Return ONLY the explanation text."""


CONFIDENCE_SCORING_PROMPT = """You are assigning a confidence score to a gene regulatory prediction.

Gene pair: {gene_a} -> {gene_b}

Evidence quality:
- Number of supporting documents: {num_documents}
- Average similarity score: {avg_similarity:.3f}
- Context found: {context_found}
- Condition match: {condition_match}
- Statistical support: {statistical_summary}

Assign a confidence score between 0.0 and 1.0 based on:
1. Quality and quantity of literature evidence (0-0.4 points)
2. Condition matching quality (0-0.3 points)
3. Statistical support strength (0-0.3 points)

Guidelines:
- High literature support + condition match + strong stats: 0.8-1.0
- Good literature + condition mismatch: 0.4-0.6
- Weak literature or no context found: 0.0-0.3
- Contradictory evidence: 0.0-0.2

Return ONLY a single number between 0.0 and 1.0 (e.g., "0.75")"""


def format_documents_for_prompt(documents: List[SupportingDocument]) -> str:
    """Format retrieved documents for inclusion in prompts."""
    if not documents:
        return "No documents retrieved."

    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(
            f"Document {i} (similarity: {doc.similarity_score:.3f}, source: {doc.source}):\n"
            f"{doc.text}\n"
        )

    return "\n".join(formatted)


def format_conditions(conditions: List[str]) -> str:
    """Format condition list for prompts."""
    if not conditions:
        return "No specific conditions specified"

    return ", ".join(conditions)


def format_statistical_summary(summary: Optional[StatisticalSummary]) -> str:
    """Format statistical summary for prompts."""
    if summary is None:
        return "No statistical analysis available"

    parts = []
    if summary.correlation is not None:
        parts.append(f"Correlation: {summary.correlation:.3f}")
    if summary.mutual_information is not None:
        parts.append(f"Mutual Information: {summary.mutual_information:.3f}")
    if summary.p_value is not None:
        parts.append(f"P-value: {summary.p_value:.4f}")

    return ", ".join(parts) if parts else "No statistical metrics available"


# ============================================================================
# Internal Workflow Nodes (6 nodes)
# ============================================================================

def query_formulation(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 1: Formulate optimized search query and generate embedding.

    Args:
        state: Current workflow state

    Returns:
        State updates with search_query, query_embedding, and reasoning trace
    """
    gene_a = state["gene_a"]
    gene_b = state["gene_b"]
    dataset_conditions = state["dataset_metadata"].conditions

    # Add reasoning trace entry
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(f"Formulating query for {gene_a} -> {gene_b}")

    # Generate search query using LLM
    client = get_alcf_client()
    prompt = QUERY_FORMULATION_PROMPT.format(
        gene_a=gene_a,
        gene_b=gene_b,
        dataset_conditions=format_conditions(dataset_conditions)
    )

    search_query = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  # Lower temperature for focused query
    ).strip()

    reasoning_trace.append(f"Generated search query: {search_query}")

    # Generate query embedding
    query_embedding = client.embed(search_query)

    reasoning_trace.append(f"Generated query embedding (dim={len(query_embedding)})")

    return {
        "search_query": search_query,
        "query_embedding": query_embedding,
        "reasoning_trace": reasoning_trace
    }


def vector_retrieval(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 2: Retrieve relevant documents from vector store.

    Args:
        state: Current workflow state

    Returns:
        State updates with retrieved_documents, retrieval_count, and reasoning trace
    """
    gene_a = state["gene_a"]
    gene_b = state["gene_b"]
    search_query = state["search_query"]

    reasoning_trace = state["reasoning_trace"]
    reasoning_trace.append(f"Retrieving documents for query: {search_query}")

    # Retrieve documents
    vector_store = get_vector_store()
    retrieved_documents = vector_store.query_by_gene_pair(
        gene_a=gene_a,
        gene_b=gene_b
    )

    retrieval_count = len(retrieved_documents)
    reasoning_trace.append(
        f"Retrieved {retrieval_count} documents "
        f"(avg similarity: {sum(d.similarity_score for d in retrieved_documents) / max(retrieval_count, 1):.3f})"
    )

    return {
        "retrieved_documents": retrieved_documents,
        "retrieval_count": retrieval_count,
        "reasoning_trace": reasoning_trace
    }


def context_extraction(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 3: Extract regulatory context from retrieved documents.

    Args:
        state: Current workflow state

    Returns:
        State updates with context info and reasoning trace
    """
    gene_a = state["gene_a"]
    gene_b = state["gene_b"]
    retrieved_documents = state["retrieved_documents"]

    reasoning_trace = state["reasoning_trace"]
    reasoning_trace.append("Extracting regulatory context from documents")

    # Handle no documents case
    if not retrieved_documents:
        reasoning_trace.append("No documents found - marking as unknown")
        return {
            "context_found": False,
            "regulatory_relationship": "unknown",
            "required_conditions": [],
            "literature_summary": "No literature evidence found for this gene pair.",
            "reasoning_trace": reasoning_trace
        }

    # Format documents for prompt
    documents_text = format_documents_for_prompt(retrieved_documents)

    # Extract context using LLM
    client = get_alcf_client()
    prompt = CONTEXT_EXTRACTION_PROMPT.format(
        gene_a=gene_a,
        gene_b=gene_b,
        documents=documents_text
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    # Parse response
    context_found = "yes" in response.split("CONTEXT_FOUND:")[1].split("\n")[0].lower()
    regulatory_relationship = response.split("REGULATORY_RELATIONSHIP:")[1].split("\n")[0].strip()
    required_conditions_str = response.split("REQUIRED_CONDITIONS:")[1].split("\n")[0].strip()
    required_conditions = [c.strip() for c in required_conditions_str.split(",") if c.strip()]
    literature_summary = response.split("SUMMARY:")[1].strip()

    reasoning_trace.append(
        f"Context extraction: found={context_found}, "
        f"relationship={regulatory_relationship}, "
        f"conditions={len(required_conditions)}"
    )

    return {
        "context_found": context_found,
        "regulatory_relationship": regulatory_relationship,
        "required_conditions": required_conditions,
        "literature_summary": literature_summary,
        "reasoning_trace": reasoning_trace
    }


def condition_matching(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 4: Compare dataset conditions with required conditions.

    Args:
        state: Current workflow state

    Returns:
        State updates with condition matching results and reasoning trace
    """
    gene_a = state["gene_a"]
    gene_b = state["gene_b"]
    regulatory_relationship = state["regulatory_relationship"]
    required_conditions = state["required_conditions"]
    dataset_metadata = state["dataset_metadata"]
    statistical_summary = state.get("statistical_summary")

    reasoning_trace = state["reasoning_trace"]
    reasoning_trace.append("Matching dataset conditions with literature requirements")

    # Extract dataset conditions
    dataset_conditions = dataset_metadata.conditions.copy()
    if dataset_metadata.treatment:
        dataset_conditions.append(dataset_metadata.treatment)
    if dataset_metadata.growth_phase:
        dataset_conditions.append(dataset_metadata.growth_phase)

    # Use LLM to compare conditions
    client = get_alcf_client()
    prompt = CONDITION_MATCHING_PROMPT.format(
        gene_a=gene_a,
        gene_b=gene_b,
        regulatory_relationship=regulatory_relationship,
        required_conditions=format_conditions(required_conditions),
        dataset_conditions=format_conditions(dataset_conditions),
        statistical_summary=format_statistical_summary(statistical_summary)
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    # Parse response
    condition_match = response.split("CONDITION_MATCH:")[1].split("\n")[0].strip()
    condition_analysis = response.split("ANALYSIS:")[1].strip()

    reasoning_trace.append(f"Condition matching result: {condition_match}")

    return {
        "dataset_conditions": dataset_conditions,
        "condition_match": condition_match,
        "condition_analysis": condition_analysis,
        "reasoning_trace": reasoning_trace
    }


def explanation_generation(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 5: Generate natural language explanation.

    Args:
        state: Current workflow state

    Returns:
        State updates with explanation and reasoning trace
    """
    gene_a = state["gene_a"]
    gene_b = state["gene_b"]
    context_found = state["context_found"]
    regulatory_relationship = state["regulatory_relationship"]
    required_conditions = state["required_conditions"]
    dataset_conditions = state["dataset_conditions"]
    condition_match = state["condition_match"]
    literature_summary = state["literature_summary"]
    condition_analysis = state["condition_analysis"]
    statistical_summary = state.get("statistical_summary")

    reasoning_trace = state["reasoning_trace"]
    reasoning_trace.append("Generating natural language explanation")

    # Generate explanation using LLM
    client = get_alcf_client()
    prompt = EXPLANATION_GENERATION_PROMPT.format(
        gene_a=gene_a,
        gene_b=gene_b,
        context_found=context_found,
        regulatory_relationship=regulatory_relationship,
        required_conditions=format_conditions(required_conditions),
        dataset_conditions=format_conditions(dataset_conditions),
        condition_match=condition_match,
        literature_summary=literature_summary,
        condition_analysis=condition_analysis,
        statistical_summary=format_statistical_summary(statistical_summary)
    )

    explanation = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    ).strip()

    # Calculate confidence score
    retrieved_documents = state["retrieved_documents"]
    num_documents = len(retrieved_documents)
    avg_similarity = (
        sum(d.similarity_score for d in retrieved_documents) / max(num_documents, 1)
        if retrieved_documents else 0.0
    )

    confidence_prompt = CONFIDENCE_SCORING_PROMPT.format(
        gene_a=gene_a,
        gene_b=gene_b,
        num_documents=num_documents,
        avg_similarity=avg_similarity,
        context_found=context_found,
        condition_match=condition_match,
        statistical_summary=format_statistical_summary(statistical_summary)
    )

    confidence_response = client.chat(
        messages=[{"role": "user", "content": confidence_prompt}],
        temperature=0.1
    ).strip()

    # Extract numeric confidence
    try:
        confidence = float(re.search(r"0\.\d+|1\.0|0", confidence_response).group())
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except (AttributeError, ValueError):
        confidence = 0.5  # Default if parsing fails

    reasoning_trace.append(f"Generated explanation with confidence={confidence:.3f}")

    return {
        "explanation": explanation,
        "confidence": confidence,
        "reasoning_trace": reasoning_trace
    }


def output_formatting(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Node 6: Format final output.

    Args:
        state: Current workflow state

    Returns:
        State updates with final_output and reasoning trace
    """
    reasoning_trace = state["reasoning_trace"]
    reasoning_trace.append("Formatting final output")

    # Create ResearchAgentOutput
    output = ResearchAgentOutput(
        gene_pair=f"{state['gene_a']} -> {state['gene_b']}",
        context_found=state["context_found"],
        regulatory_relationship=state["regulatory_relationship"],
        required_conditions=state["required_conditions"],
        dataset_conditions=state["dataset_conditions"],
        condition_match=state["condition_match"],
        explanation=state["explanation"],
        confidence=state["confidence"],
        supporting_documents=state["retrieved_documents"],
        reasoning_trace=reasoning_trace
    )

    reasoning_trace.append("Workflow completed successfully")

    return {
        "final_output": output.model_dump(),
        "reasoning_trace": reasoning_trace
    }


# ============================================================================
# Internal LangGraph Workflow
# ============================================================================

# Global internal graph instance (cached)
_internal_graph = None


def _get_or_create_internal_graph():
    """Get or create the internal 6-node LangGraph workflow."""
    global _internal_graph

    if _internal_graph is None:
        # Create state graph
        workflow = StateGraph(ResearchAgentState)

        # Add nodes
        workflow.add_node("query_formulation", query_formulation)
        workflow.add_node("vector_retrieval", vector_retrieval)
        workflow.add_node("context_extraction", context_extraction)
        workflow.add_node("condition_matching", condition_matching)
        workflow.add_node("explanation_generation", explanation_generation)
        workflow.add_node("output_formatting", output_formatting)

        # Define edges (sequential flow)
        workflow.set_entry_point("query_formulation")
        workflow.add_edge("query_formulation", "vector_retrieval")
        workflow.add_edge("vector_retrieval", "context_extraction")
        workflow.add_edge("context_extraction", "condition_matching")
        workflow.add_edge("condition_matching", "explanation_generation")
        workflow.add_edge("explanation_generation", "output_formatting")
        workflow.add_edge("output_formatting", END)

        # Compile graph
        _internal_graph = workflow.compile()

    return _internal_graph


def _run_internal_workflow(research_input: ResearchAgentInput) -> ResearchAgentOutput:
    """Execute internal 6-node LangGraph workflow."""
    graph = _get_or_create_internal_graph()

    # Initialize internal state
    internal_state: ResearchAgentState = {
        "gene_a": research_input.gene_a,
        "gene_b": research_input.gene_b,
        "dataset_metadata": research_input.dataset_metadata,
        "statistical_summary": research_input.statistical_summary,
        "reasoning_trace": []
    }

    # Run workflow
    final_state = graph.invoke(internal_state)

    # Convert to output
    return ResearchAgentOutput(**final_state["final_output"])


# ============================================================================
# State Mapping Functions
# ============================================================================

def _extract_gene_pairs_from_state(state: AgentState) -> List[Tuple[str, str]]:
    """Extract (tf, target) pairs from current batch."""
    pairs = []
    current_tfs = state.get("current_batch_tfs", [])
    lit_graph = state.get("literature_graph", nx.DiGraph())

    for tf in current_tfs:
        # Get all targets from literature graph
        targets = list(lit_graph.successors(tf))
        pairs.extend((tf, target) for target in targets)

    return pairs


def _construct_dataset_metadata(state: AgentState) -> DatasetMetadata:
    """Build DatasetMetadata from AgentState."""
    metadata_df = state.get("metadata", pd.DataFrame())
    current_context = state.get("current_context", "unknown")

    # Extract conditions from context string
    conditions = _parse_conditions_from_context(current_context)

    # Extract treatment and growth_phase from metadata
    treatment = _extract_treatment(metadata_df)
    growth_phase = _extract_growth_phase(metadata_df)

    return DatasetMetadata(
        conditions=conditions,
        treatment=treatment,
        growth_phase=growth_phase,
        media=_extract_media(metadata_df),
        strain=_extract_strain(metadata_df)
    )


def _parse_conditions_from_context(context: str) -> List[str]:
    """Parse conditions from context string (e.g., 'Anaerobic conditions' -> ['anaerobic'])."""
    if not context or context == "unknown":
        return []

    # Simple parsing - split by common separators and lowercase
    conditions = context.lower().replace(" conditions", "").replace(" condition", "")
    conditions = [c.strip() for c in conditions.split(",") if c.strip()]

    return conditions


def _extract_treatment(metadata_df: pd.DataFrame) -> str:
    """Extract treatment from metadata DataFrame."""
    if metadata_df.empty:
        return "none"

    # Look for common treatment columns
    treatment_cols = ["treatment", "Treatment", "perturbation", "Perturbation"]
    for col in treatment_cols:
        if col in metadata_df.columns:
            # Get most common treatment
            treatment = metadata_df[col].mode()
            if not treatment.empty:
                return str(treatment[0])

    return "none"


def _extract_growth_phase(metadata_df: pd.DataFrame) -> str:
    """Extract growth phase from metadata DataFrame."""
    if metadata_df.empty:
        return "exponential"

    # Look for growth phase columns
    phase_cols = ["growth_phase", "Growth_Phase", "phase", "Phase"]
    for col in phase_cols:
        if col in metadata_df.columns:
            # Get most common phase
            phase = metadata_df[col].mode()
            if not phase.empty:
                return str(phase[0])

    return "exponential"


def _extract_media(metadata_df: pd.DataFrame) -> Optional[str]:
    """Extract media from metadata DataFrame."""
    if metadata_df.empty:
        return None

    # Look for media columns
    media_cols = ["media", "Media", "medium", "Medium"]
    for col in media_cols:
        if col in metadata_df.columns:
            media = metadata_df[col].mode()
            if not media.empty:
                return str(media[0])

    return None


def _extract_strain(metadata_df: pd.DataFrame) -> Optional[str]:
    """Extract strain from metadata DataFrame."""
    if metadata_df.empty:
        return None

    # Look for strain columns
    strain_cols = ["strain", "Strain", "organism", "Organism"]
    for col in strain_cols:
        if col in metadata_df.columns:
            strain = metadata_df[col].mode()
            if not strain.empty:
                return str(strain[0])

    return None


def _get_statistical_summary(state: AgentState, gene_a: str, gene_b: str) -> Optional[StatisticalSummary]:
    """Extract statistical summary for gene pair from AgentState."""
    analysis_results = state.get("analysis_results", {})

    # Navigate to gene pair results
    if gene_a not in analysis_results:
        return None

    if gene_b not in analysis_results[gene_a]:
        return None

    stats = analysis_results[gene_a][gene_b]

    # Extract relevant fields
    return StatisticalSummary(
        correlation=stats.get("correlation"),
        mutual_information=stats.get("mi"),
        p_value=stats.get("p_value")
    )


def _format_results_for_agent_state(
    results: List[Tuple[str, str, ResearchAgentOutput]]
) -> Dict[str, Any]:
    """Format results as AgentState updates."""
    literature_edges = {}
    annotations = {}
    reasoning_traces = {}
    errors = []

    for gene_a, gene_b, output in results:
        # Build literature_edges structure
        if gene_a not in literature_edges:
            literature_edges[gene_a] = {}

        literature_edges[gene_a][gene_b] = {
            "exists": output.context_found,
            "effect": output.regulatory_relationship,
            "evidence_strength": "strong" if output.confidence > 0.7 else "weak",
            "conditions_required": output.required_conditions
        }

        # Build annotations structure
        edge_id = f"{gene_a}→{gene_b}"
        annotations[edge_id] = {
            "match": output.condition_match == "match",
            "explanation": output.explanation,
            "confidence": output.confidence,
            "supporting_docs": [doc.doc_id for doc in output.supporting_documents]
        }

        # Store reasoning trace
        reasoning_traces[f"{gene_a}->{gene_b}"] = output.reasoning_trace

    return {
        "research__literature_edges": literature_edges,
        "research__annotations": annotations,
        "research__reasoning_traces": reasoning_traces,
        "research__errors": errors
    }


def _error_response(message: str) -> Dict[str, Any]:
    """Return error state update."""
    return {
        "research__errors": [message],
        "research__literature_edges": {},
        "research__annotations": {}
    }


def _empty_response(message: str) -> Dict[str, Any]:
    """Return empty state update with info message."""
    return {
        "research__literature_edges": {},
        "research__annotations": {},
        "research__reasoning_traces": {"info": message}
    }


def _error_output(error: Exception, gene_a: str, gene_b: str) -> ResearchAgentOutput:
    """Create error ResearchAgentOutput."""
    return ResearchAgentOutput(
        gene_pair=f"{gene_a} -> {gene_b}",
        context_found=False,
        regulatory_relationship="unknown",
        required_conditions=[],
        dataset_conditions=[],
        condition_match="unknown",
        explanation=f"Analysis failed: {str(error)}",
        confidence=0.0,
        supporting_documents=[],
        reasoning_trace=[f"Error: {str(error)}"]
    )


def _validate_state_inputs(state: AgentState) -> None:
    """Validate required state fields exist."""
    required = ["current_batch_tfs", "literature_graph", "metadata",
                "gene_name_to_bnumber", "bnumber_to_gene_name"]
    missing = [k for k in required if k not in state or not state[k]]
    if missing:
        raise ValueError(f"Missing required state fields: {missing}")


def _ensure_vector_store_initialized() -> None:
    """Check if vector store has documents."""
    vs = get_vector_store()
    if vs.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Load literature with:\n"
            "  from src.utils.vector_store import get_vector_store\n"
            "  vs = get_vector_store()\n"
            "  vs.load_from_json('path/to/literature.json')"
        )


# ============================================================================
# Main Node Function (External Interface)
# ============================================================================

def research_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Research Agent Node: Retrieves and contextualizes literature evidence.

    This node integrates a 6-step LangGraph workflow to analyze gene pairs:
    1. Query Formulation: Create optimized search query
    2. Vector Retrieval: Retrieve literature from ChromaDB
    3. Context Extraction: Extract regulatory relationships
    4. Condition Matching: Compare dataset vs literature conditions
    5. Explanation Generation: Generate natural language explanation
    6. Output Formatting: Structure final output

    Args:
        state: AgentState with current_batch_tfs, literature_graph, metadata

    Returns:
        Dict with state updates:
        - research__literature_edges: Literature relationships
        - research__annotations: Edge-level annotations
        - research__reasoning_traces: Transparency logs
        - research__errors: Error messages (if any)

    Raises:
        No exceptions raised - errors are captured in state updates
    """
    logger.info("=== RESEARCH AGENT NODE: Retrieving literature evidence ===")

    # Phase 1: Validation & Setup
    try:
        _validate_state_inputs(state)
        _ensure_vector_store_initialized()
    except Exception as e:
        logger.error(f"Research Agent setup failed: {e}")
        return _error_response(f"Setup failed: {e}")

    # Phase 2: Extract Gene Pairs
    gene_pairs = _extract_gene_pairs_from_state(state)
    if not gene_pairs:
        logger.warning("No gene pairs to analyze")
        return _empty_response("No gene pairs to analyze")

    logger.info(f"Processing {len(gene_pairs)} gene pairs")

    # Phase 3: Build Dataset Context
    dataset_metadata = _construct_dataset_metadata(state)

    # Phase 4: Process Each Gene Pair
    all_results = []
    for gene_a, gene_b in gene_pairs:
        try:
            # Create input
            research_input = ResearchAgentInput(
                gene_a=gene_a,
                gene_b=gene_b,
                dataset_metadata=dataset_metadata,
                statistical_summary=_get_statistical_summary(state, gene_a, gene_b)
            )

            # Run internal 6-node workflow
            logger.debug(f"Analyzing {gene_a} -> {gene_b}")
            output = _run_internal_workflow(research_input)
            all_results.append((gene_a, gene_b, output))

        except Exception as e:
            # Capture per-pair errors
            logger.warning(f"Error analyzing {gene_a} -> {gene_b}: {e}")
            all_results.append((gene_a, gene_b, _error_output(e, gene_a, gene_b)))

    # Phase 5: Format Results
    logger.info(f"Research Agent completed: {len(all_results)} gene pairs processed")
    return _format_results_for_agent_state(all_results)


# ============================================================================
# Standalone Function (Alternative Interface)
# ============================================================================

def analyze_gene_pair(
    gene_a: str,
    gene_b: str,
    dataset_metadata: DatasetMetadata,
    statistical_summary: Optional[StatisticalSummary] = None
) -> ResearchAgentOutput:
    """
    Standalone function for analyzing a single gene pair.

    This provides an interface for using the Research Agent outside
    of the main LangGraph workflow.

    Args:
        gene_a: Regulator gene identifier
        gene_b: Target gene identifier
        dataset_metadata: Experimental dataset metadata
        statistical_summary: Optional statistical results

    Returns:
        ResearchAgentOutput with analysis results
    """
    research_input = ResearchAgentInput(
        gene_a=gene_a,
        gene_b=gene_b,
        dataset_metadata=dataset_metadata,
        statistical_summary=statistical_summary
    )

    return _run_internal_workflow(research_input)
