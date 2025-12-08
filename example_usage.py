"""Example usage of the Research Agent."""

import json
from pathlib import Path

from research_agent.models import (
    ResearchAgentInput,
    DatasetMetadata,
    StatisticalSummary
)
from research_agent.graph.graph import get_research_agent
from research_agent.vector_store import get_vector_store


def setup_vector_store():
    """Load sample literature into the vector store."""
    print("Setting up vector store...")

    # Get vector store instance
    vector_store = get_vector_store()

    # Load sample literature data
    sample_data_path = Path(__file__).parent / "research_agent" / "data" / "sample_literature.json"

    # Only load if collection is empty
    if vector_store.count() == 0:
        print(f"Loading literature from {sample_data_path}")
        vector_store.load_from_json(str(sample_data_path))
        print(f"Loaded {vector_store.count()} documents")
    else:
        print(f"Vector store already contains {vector_store.count()} documents")

    return vector_store


def example_lexA_recA_dna_damage():
    """
    Example: lexA -> recA under DNA damage conditions.

    Expected outcome: Should find literature about SOS response and
    note that DNA damage conditions are required for this interaction.
    """
    print("\n" + "="*80)
    print("Example 1: lexA -> recA under DNA damage conditions")
    print("="*80)

    # Create input
    agent_input = ResearchAgentInput(
        gene_a="lexA",
        gene_b="recA",
        dataset_metadata=DatasetMetadata(
            conditions=["DNA_damage", "UV_exposure"],
            treatment="UV_radiation",
            growth_phase="exponential"
        ),
        statistical_summary=StatisticalSummary(
            correlation=0.85,
            mutual_information=1.2,
            p_value=0.001
        )
    )

    # Run analysis
    agent = get_research_agent()
    output = agent.analyze(agent_input)

    # Display results
    print(f"\nGene pair: {output.gene_pair}")
    print(f"Context found: {output.context_found}")
    print(f"Regulatory relationship: {output.regulatory_relationship}")
    print(f"Required conditions: {', '.join(output.required_conditions)}")
    print(f"Dataset conditions: {', '.join(output.dataset_conditions)}")
    print(f"Condition match: {output.condition_match}")
    print(f"Confidence: {output.confidence:.3f}")
    print(f"\nExplanation:\n{output.explanation}")
    print(f"\nSupporting documents: {len(output.supporting_documents)}")

    return output


def example_lexA_recA_normal_growth():
    """
    Example: lexA -> recA under normal growth conditions.

    Expected outcome: Should find literature about repression under
    normal conditions, and note condition mismatch if dataset has DNA damage.
    """
    print("\n" + "="*80)
    print("Example 2: lexA -> recA under normal growth conditions")
    print("="*80)

    # Create input
    agent_input = ResearchAgentInput(
        gene_a="lexA",
        gene_b="recA",
        dataset_metadata=DatasetMetadata(
            conditions=["normal_growth", "unstressed"],
            treatment="none",
            growth_phase="exponential"
        ),
        statistical_summary=StatisticalSummary(
            correlation=0.15,
            p_value=0.45
        )
    )

    # Run analysis
    agent = get_research_agent()
    output = agent.analyze(agent_input)

    # Display results
    print(f"\nGene pair: {output.gene_pair}")
    print(f"Context found: {output.context_found}")
    print(f"Regulatory relationship: {output.regulatory_relationship}")
    print(f"Condition match: {output.condition_match}")
    print(f"Confidence: {output.confidence:.3f}")
    print(f"\nExplanation:\n{output.explanation}")

    return output


def example_crp_lacZ_glucose_depleted():
    """
    Example: crp -> lacZ under glucose depletion.

    Expected outcome: Should find activation under catabolite repression conditions.
    """
    print("\n" + "="*80)
    print("Example 3: crp -> lacZ under glucose depletion")
    print("="*80)

    # Create input
    agent_input = ResearchAgentInput(
        gene_a="crp",
        gene_b="lacZ",
        dataset_metadata=DatasetMetadata(
            conditions=["glucose_depletion", "lactose_present"],
            treatment="lactose_shift",
            growth_phase="exponential"
        ),
        statistical_summary=StatisticalSummary(
            correlation=0.78,
            mutual_information=1.5,
            p_value=0.002
        )
    )

    # Run analysis
    agent = get_research_agent()
    output = agent.analyze(agent_input)

    # Display results
    print(f"\nGene pair: {output.gene_pair}")
    print(f"Regulatory relationship: {output.regulatory_relationship}")
    print(f"Condition match: {output.condition_match}")
    print(f"Confidence: {output.confidence:.3f}")
    print(f"\nExplanation:\n{output.explanation}")

    return output


def main():
    """Run all examples."""
    print("Research Agent Example Usage")
    print("="*80)

    # Setup
    setup_vector_store()

    # Run examples
    example_lexA_recA_dna_damage()
    example_lexA_recA_normal_growth()
    example_crp_lacZ_glucose_depleted()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
