"""LangGraph workflow construction for Research Agent."""

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError(
        "langgraph is not installed. Please run: pip install langgraph>=0.0.20"
    )

from .state import ResearchAgentState
from .nodes import (
    query_formulation,
    vector_retrieval,
    context_extraction,
    condition_matching,
    explanation_generation,
    output_formatting
)
from ..models import ResearchAgentInput, ResearchAgentOutput


def create_research_agent_graph():
    """
    Create the Research Agent LangGraph workflow.

    The workflow consists of 6 sequential nodes:
    1. query_formulation - Create optimized search query
    2. vector_retrieval - Retrieve relevant documents
    3. context_extraction - Extract regulatory information
    4. condition_matching - Compare conditions
    5. explanation_generation - Generate natural language explanation
    6. output_formatting - Format final output

    Returns:
        Compiled LangGraph
    """
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
    return workflow.compile()


class ResearchAgent:
    """
    High-level Research Agent interface.

    This class provides a simple interface to run the Research Agent
    workflow on input data.
    """

    def __init__(self):
        """Initialize Research Agent with compiled graph."""
        self.graph = create_research_agent_graph()

    def analyze(self, agent_input: ResearchAgentInput) -> ResearchAgentOutput:
        """
        Analyze a gene pair using the Research Agent workflow.

        Args:
            agent_input: ResearchAgentInput with gene pair and metadata

        Returns:
            ResearchAgentOutput with analysis results

        Raises:
            Exception: If workflow execution fails
        """
        # Initialize state from input
        initial_state: ResearchAgentState = {
            "gene_a": agent_input.gene_a,
            "gene_b": agent_input.gene_b,
            "dataset_metadata": agent_input.dataset_metadata,
            "statistical_summary": agent_input.statistical_summary,
            "reasoning_trace": []
        }

        try:
            # Run workflow
            final_state = self.graph.invoke(initial_state)

            # Extract and validate output
            output_dict = final_state["final_output"]
            return ResearchAgentOutput(**output_dict)

        except Exception as e:
            # Return error output
            return ResearchAgentOutput(
                gene_pair=f"{agent_input.gene_a} -> {agent_input.gene_b}",
                context_found=False,
                regulatory_relationship="unknown",
                required_conditions=[],
                dataset_conditions=agent_input.dataset_metadata.conditions,
                condition_match="unknown",
                explanation=f"Analysis failed: {str(e)}",
                confidence=0.0,
                supporting_documents=[],
                reasoning_trace=[f"Error: {str(e)}"]
            )

    def analyze_batch(
        self,
        inputs: list[ResearchAgentInput]
    ) -> list[ResearchAgentOutput]:
        """
        Analyze multiple gene pairs.

        Args:
            inputs: List of ResearchAgentInput instances

        Returns:
            List of ResearchAgentOutput instances
        """
        return [self.analyze(inp) for inp in inputs]


# Global agent instance
_research_agent = None


def get_research_agent() -> ResearchAgent:
    """
    Get or create global Research Agent instance.

    Returns:
        ResearchAgent instance
    """
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent
