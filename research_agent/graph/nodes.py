"""LangGraph node functions for Research Agent workflow."""

import re
from typing import Dict, Any

from .state import ResearchAgentState
from .prompts import (
    QUERY_FORMULATION_PROMPT,
    CONTEXT_EXTRACTION_PROMPT,
    CONDITION_MATCHING_PROMPT,
    EXPLANATION_GENERATION_PROMPT,
    CONFIDENCE_SCORING_PROMPT,
    format_documents_for_prompt,
    format_conditions,
    format_statistical_summary
)
from ..alcf_client import get_alcf_client
from ..vector_store import get_vector_store
from ..models import ResearchAgentOutput


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
    dataset_conditions = dataset_metadata.conditions
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
