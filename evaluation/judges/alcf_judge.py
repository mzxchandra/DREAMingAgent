from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.llm_config import create_argonne_llm
from .prompts import judge_prompt

class JudgeScores(BaseModel):
    biological_accuracy: int = Field(description="Score 1-5 for biological correctness")
    statistical_reasoning: int = Field(description="Score 1-5 for statistical interpretation")
    clarity: int = Field(description="Score 1-5 for clarity")
    overall_quality: int = Field(description="Score 1-5 overall")
    reasoning: str = Field(description="Justification for the scores")

class ALCFJudge:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = create_argonne_llm(
            model_name=model_name,
            temperature=0.1 # Low temp for consistent judging
        )
        self.parser = JsonOutputParser(pydantic_object=JudgeScores)
        self.chain = judge_prompt | self.llm | self.parser

    def evaluate_explanation(
        self, 
        edge_id: str, 
        status: str, 
        lit_evidence: str, 
        z_score: float, 
        explanation: str
    ) -> Dict[str, Any]:
        """Run evaluation for a single explanation."""
        try:
            result = self.chain.invoke({
                "edge_id": edge_id,
                "status": status,
                "literature_evidence": lit_evidence,
                "statistical_score": f"{z_score:.2f}",
                "explanation": explanation
            })
            return result
        except Exception as e:
            # Fallback for parsing errors
            return {
                "biological_accuracy": 1, 
                "statistical_reasoning": 1, 
                "clarity": 1, 
                "overall_quality": 1, 
                "reasoning": f"Judge Error: {str(e)}"
            }
