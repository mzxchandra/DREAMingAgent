from langchain_core.prompts import PromptTemplate

JUDGE_PROMPT_TEMPLATE = """You are an expert biological judge evaluating gene regulatory network explanations.
Your task is to score the quality of a reconciliation explanation provided by an AI agent.

Evaluate the following explanation on a 1-5 scale for these dimensions:

1. **Biological Accuracy** (1-5): Does the explanation align with known E. coli biology (e.g., FNR is anaerobic)? 
   - 1: Completely incorrect or nonsense.
   - 5: Highly accurate, cites specific mechanisms correctly.
   
2. **Statistical Reasoning** (1-5): Does it correctly interpret the CLR z-score and correlation?
   - 1: Ignores data or misinterprets high/low scores.
   - 5: Properly contextualizes data vs literature.
   
3. **Clarity** (1-5): Is the explanation clear, concise, and easy to understand?
   
4. **Overall Quality** (1-5): General assessment of utility.

---
**Input Data:**
Edge: {edge_id}
Recommended Status: {status}
Literature Evidence: {literature_evidence}
CLR Z-Score: {statistical_score}

**Agent Explanation:**
"{explanation}"
---

Provide your evaluation in valid JSON format ONLY:
{{
  "biological_accuracy": int,
  "statistical_reasoning": int,
  "clarity": int,
  "overall_quality": int,
  "reasoning": "string (explain your scores)"
}}
"""

judge_prompt = PromptTemplate(
    template=JUDGE_PROMPT_TEMPLATE,
    input_variables=["edge_id", "status", "literature_evidence", "statistical_score", "explanation"]
)
