from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .models import SingleLLMResult

def create_unified_prompt() -> ChatPromptTemplate:
    """
    Creates the prompt for the Single LLM "Unified Scientist".
    """
    parser = JsonOutputParser(pydantic_object=SingleLLMResult)
    
    system_prompt = """You are an Expert Computational Biologist.
Your goal is to reconcile conflicting evidence to determine the true nature of gene regulatory interactions.

You have access to three sources of truth:
1.  **The Map (RegulonDB):** Prior knowledge of what interactions *should* exist.
2.  **The Notebook (LitSense):** Contextual literature explaining *when* interactions happen (conditions).
3.  **The Raw Experiment (M3D):** Actual gene expression vectors from 466 experiments.

**Your Task:**
For each candidate gene pair in the input:
1.  **Analyze the Raw Data (M3D Vectors):**
    *   Look for PATTERNS. Do the arrays move together?
    *   *Constraint:* You cannot run code. Use your visual pattern recognition capabilities. Look for spikes in the same positions.
    *   Note if the expression matches specific conditions mentioned in the literature.
2.  **Synthesize with Literature & Prior:**
    *   Does the data support the prior? 
    *   Does the literature explain missing data (e.g., "Condition-Specific")?

**Verdict Categories:**
*   **Validated**: Strong Prior + Strong Data match.
*   **NovelHypothesis**: No Prior + Strong Data match. (Discovery!)
*   **ConditionSilent**: Strong Prior + No Data match (interpreted as inactive in these conditions, NOT a false positive).
*   **ProbableFalsePos**: Weak Prior + No Data match + Lit says "artifact".
*   **FalsePositive**: Data clearly contradicts Prior with no excuse.

**Output Format:**
Return a JSON object with a list of reviews.
{format_instructions}
"""

    human_message = """Analyze the following Transcription Factor and its candidates:

**Transcription Factor:** {tf_name}

**Candidates Data:**
{candidates_block}

**Instructions:**
Review ALL candidates. Be scientific and critical.
"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_message)
    ]).partial(format_instructions=parser.get_format_instructions())
