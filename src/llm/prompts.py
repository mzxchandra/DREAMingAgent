"""
LLM Prompts for Agentic Reasoning

Contains system prompts and prompt templates for the Context Agent
and Reconciler nodes.
"""

# ============================================================================
# System Prompts
# ============================================================================

RECONCILER_SYSTEM_PROMPT = """You are an expert systems biologist specializing in E. coli gene regulatory networks. Your task is to reconcile literature-based knowledge (from RegulonDB) with high-throughput expression data (from M3D).

You understand:
- Transcription factors (TFs) and their target genes
- Evidence types: "Strong" (DNA footprinting, ChIP-seq) vs "Weak" (expression inference, predictions)
- Mutual Information (MI) and CLR z-scores as measures of regulatory relationship strength
- Context-dependent regulation (a TF may only regulate a gene under specific conditions)

Your job is to reason about discrepancies between literature assertions and data observations, providing nuanced biological interpretations rather than simple rule-based decisions.

Always consider:
1. Could the TF be inactive in the experimental conditions sampled?
2. Could there be indirect regulation through another TF?
3. Could the literature evidence be outdated or based on weak methodology?
4. Could this be a genuine novel discovery?

Respond with structured JSON containing your reasoning and decision."""


CONTEXT_AGENT_SYSTEM_PROMPT = """You are an expert in E. coli transcriptional regulation. Given a transcription factor (TF) name, you must determine what experimental conditions would activate this TF and make its regulatory effects observable.

You have deep knowledge of:
- Metabolic regulation (CRP, FNR, ArcA, etc.)
- Stress responses (RpoH for heat shock, SoxRS for oxidative stress, etc.)
- Nutrient sensing (NtrC for nitrogen, PhoB for phosphate, etc.)
- DNA damage response (LexA, RecA)

Given a TF, respond with:
1. What biological signals activate this TF
2. What experimental conditions (keywords) would capture its activity
3. What conditions should be EXCLUDED (where the TF would be inactive)

Respond with structured JSON."""


# ============================================================================
# Prompt Templates
# ============================================================================

def format_reconciliation_prompt(
    tf_name: str,
    target_gene: str,
    literature_evidence: str,
    literature_effect: str,
    z_score: float,
    mi_score: float,
    context: str,
    additional_context: str = ""
) -> str:
    """
    Format a prompt for the Reconciler LLM to reason about a single edge.
    
    Args:
        tf_name: Transcription factor name
        target_gene: Target gene name
        literature_evidence: Evidence level (Strong/Weak/Unknown)
        literature_effect: Regulation type (+/-/+-/?)
        z_score: CLR z-score from M3D data
        mi_score: Mutual information score
        context: Experimental context description
        additional_context: Any additional relevant information
        
    Returns:
        Formatted prompt string
    """
    effect_desc = {
        "+": "activator (positive regulation)",
        "-": "repressor (negative regulation)", 
        "+-": "dual regulator (context-dependent)",
        "?": "unknown regulation type"
    }.get(literature_effect, "unknown")
    
    z_interpretation = (
        "very strong statistical support" if z_score >= 4.0 else
        "moderate statistical support" if z_score >= 2.0 else
        "weak statistical support" if z_score >= 1.0 else
        "no statistical support"
    )
    
    prompt = f"""
## Regulatory Edge Analysis

**Transcription Factor:** {tf_name}
**Target Gene:** {target_gene}
**Experimental Context:** {context}

### Literature Evidence (RegulonDB)
- **Evidence Level:** {literature_evidence}
- **Regulation Type:** {effect_desc}

### Data Evidence (M3D Expression Compendium)
- **CLR Z-score:** {z_score:.3f} ({z_interpretation})
- **Mutual Information:** {mi_score:.4f}

{f"### Additional Context{chr(10)}{additional_context}" if additional_context else ""}

---

**Your Task:**
Analyze this regulatory relationship and determine its status. Consider:
1. Does the data support the literature assertion?
2. If not, what is the most likely biological explanation?
3. What is your confidence in this assessment?

**Respond with JSON in this exact format:**
```json
{{
    "status": "Validated|ConditionSilent|ProbableFalsePos|NovelHypothesis|NeedsReview",
    "confidence": 0.0-1.0,
    "reasoning": "Your biological reasoning (2-3 sentences)",
    "recommendation": "What should be done next (1 sentence)"
}}
```
"""
    return prompt


def format_context_prompt(tf_name: str, tf_description: str = "") -> str:
    """
    Format a prompt for the Context Agent to determine relevant conditions.
    
    Args:
        tf_name: Transcription factor name
        tf_description: Optional description of TF function
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""
## Transcription Factor Context Analysis

**TF Name:** {tf_name}
{f"**Known Function:** {tf_description}" if tf_description else ""}

**Your Task:**
Determine what experimental conditions in a microarray compendium would be relevant for analyzing this transcription factor's regulatory activity.

**Respond with JSON in this exact format:**
```json
{{
    "tf_name": "{tf_name}",
    "biological_role": "Brief description of the TF's biological function",
    "activating_signals": ["signal1", "signal2"],
    "relevant_conditions": ["keyword1", "keyword2", "keyword3"],
    "exclude_conditions": ["keyword1", "keyword2"],
    "reasoning": "Why these conditions are relevant (1-2 sentences)"
}}
```

Be specific with condition keywords that would appear in experimental metadata (e.g., "anaerobic", "heat_shock", "glucose", "stationary_phase").
"""
    return prompt


def format_batch_reconciliation_prompt(
    tf_name: str,
    edges: list,
    context: str
) -> str:
    """
    Format a prompt for batch reconciliation of multiple edges.
    
    Args:
        tf_name: Transcription factor name
        edges: List of edge dictionaries with target, evidence, z_score, etc.
        context: Experimental context
        
    Returns:
        Formatted prompt string
    """
    edge_summaries = []
    for i, edge in enumerate(edges, 1):
        edge_summaries.append(
            f"{i}. {edge['target']}: Lit={edge['evidence']}, Effect={edge['effect']}, "
            f"z={edge['z_score']:.2f}"
        )
    
    edges_text = "\n".join(edge_summaries)
    
    prompt = f"""
## Batch Regulatory Analysis for {tf_name}

**Context:** {context}

### Edges to Analyze:
{edges_text}

**Your Task:**
For each edge, provide a brief assessment. Focus on:
- Which edges are confidently validated by data?
- Which edges might be condition-specific (silent in this context)?
- Are there any suspicious entries (potential false positives)?

**Respond with JSON:**
```json
{{
    "tf": "{tf_name}",
    "summary": "Overall assessment (1-2 sentences)",
    "edges": [
        {{"target": "gene1", "status": "Validated|ConditionSilent|ProbableFalsePos", "note": "brief reason"}}
    ]
}}
```
"""
    return prompt

