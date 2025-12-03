"""LLM prompts for Research Agent nodes."""


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


def format_documents_for_prompt(documents: list) -> str:
    """
    Format retrieved documents for inclusion in prompts.

    Args:
        documents: List of SupportingDocument instances

    Returns:
        Formatted string with document information
    """
    if not documents:
        return "No documents retrieved."

    formatted = []
    for i, doc in enumerate(documents, 1):
        formatted.append(
            f"Document {i} (similarity: {doc.similarity_score:.3f}, source: {doc.source}):\n"
            f"{doc.text}\n"
        )

    return "\n".join(formatted)


def format_conditions(conditions: list) -> str:
    """
    Format condition list for prompts.

    Args:
        conditions: List of condition strings

    Returns:
        Formatted string
    """
    if not conditions:
        return "No specific conditions specified"

    return ", ".join(conditions)


def format_statistical_summary(summary) -> str:
    """
    Format statistical summary for prompts.

    Args:
        summary: StatisticalSummary instance or None

    Returns:
        Formatted string
    """
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
