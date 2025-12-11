# Single LLM System Design: "The Unified Scientist"

## Core Philosophy
Consolidate the multi-agent pipeline into a single **Reasoning Engine** that mimics a human scientist's workflow.
**Critical Context:** This system serves as a **Baseline**. We expect it to struggle with raw data analysis compared to the specialized Multi-Agent system. The goal is to prove *why* the agents are necessary by showing that a single LLM with raw data is insufficient.

## Constraints
*   **Input:** Raw M3D Vectors, RegulonDB Graph, LitSense Excerpts.
*   **Forbidden:** Pre-calculated statistics (Correlation, Z-Scores) not present in the source files.

## Input Schema
The LLM receives a context packet for each candidate pair:

```json
{
  "transcription_factor": "AraC",
  "candidate_gene": "araB",
  "data_evidence": {
    "m3d_vectors": {
      "tf_expression": [0.1, 0.5, 2.3, ...],  // Raw data (length ~466)
      "gene_expression": [0.2, 0.4, 2.1, ...]
    },
    "metadata": {
      "num_samples": 466,
      "conditions": ["Acid", "Base", "Starvation", ...]
    }
  },
  "prior_knowledge": {
    "regulondb_status": "Confirmed Interaction",
    "regulondb_type": "Activator"
  },
  "literature_context": {
    "litsense_snippets": [
      "AraC binds to araI1/I2 sites in the presence of arabinose...",
      "Repression loops form in absence of arabinose."
    ]
  }
}
```

## The "Raw Data" Challenge & Solution
**Problem:** LLMs cannot calculate Pearson correlation on 466-dimensional vectors via token prediction.
**Solution:** The "Unified Scientist" prompt will interpret the **Visual/Structural** patterns of the vectors (if truncated) or, ideally, be equipped with a **Code Interpreter Tool** to "check the data".

### System Prompt Strategy
"You are a Computational Biologist. You have three sources of truth:
1.  **The Map (RegulonDB):** What we *expect* to see.
2.  **The Notebook (LitSense):** *Why* we expect it.
3.  **The Raw Experiment (M3D):** What actually happened.

**Your Task:**
Look at the `m3d_vectors`. Do they move together?
*   *Note:* Since you cannot compute `rho` mentally, look for **Condition-Specific** patterns. E.g., 'Target is high ONLY when TF is high'.
*   Compare this to the `litsense_snippets`. Does the data match the conditions described?
*   If RegulonDB says 'Yes' but Vectors stay flat: Mark as **Condition Silent**.
*   If RegulonDB says 'No' but Vectors match perfectly: Mark as **Novel Hypothesis**.

**Output Decision:**
{
  "verdict": "Validated" | "Novel" | "Silent" | "FalsePos",
  "confidence": 0-100,
  "reasoning": "RegulonDB expects interaction, and while overall correlation is noisy, the vector shows clear co-activation in the first 20 samples (Acidic context), matching the literature..."
}
"

## Implementation Plan
1.  **Unified Loader:** Modify `loader.py` to fetch raw rows from `expression_matrix` instead of computing `analysis_results`.
2.  **Prompt Engineering:** Design the few-shot examples to teach the LLM how to "read" vector lists (or truncated representative samples).
3.  **Execution:** Run on the same AraC test set.
