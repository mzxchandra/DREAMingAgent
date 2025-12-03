"""
Integration Tests for Reviewer Agent LLM Integration

Tests the LLM-powered review functionality using mocked LLM responses
to verify structured output parsing and prompt engineering.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from langchain_core.messages import AIMessage
from src.state import create_initial_state
from src.nodes.reviewer_agent import (
    invoke_llm_reviewer,
    reviewer_agent_node,
    SubgraphReview,
    EdgeDecisionOutput,
    format_edge_data_for_llm,
    format_tf_expression,
)


# ============================================================================
# Mock LLM Responses
# ============================================================================

MOCK_LLM_RESPONSE_VALID = """{
    "tf": "b1334",
    "edge_decisions": [
        {
            "edge_id": "b1334→b0929",
            "reconciliation_status": "Condition-Silent",
            "preserve_edge": true,
            "literature_support": "strong",
            "data_support": "none",
            "context_compatible": false,
            "explanation": "Strong literature evidence from RegulonDB indicates b1334 (FNR) activates b0929 (ompF) under anaerobic conditions. However, CLR z-score of 0.3 shows minimal correlation in this dataset. The context annotation reveals a mismatch: literature requires glucose depletion, but the M3D dataset has high glucose conditions. This explains the absence of statistical signal despite robust biological evidence.",
            "confidence": "high",
            "recommendation": "Preserve edge with annotation indicating condition-specific regulation. Consider testing under glucose-limited conditions."
        },
        {
            "edge_id": "b1334→b2415",
            "reconciliation_status": "Novel Hypothesis",
            "preserve_edge": true,
            "literature_support": "none",
            "data_support": "strong",
            "context_compatible": false,
            "explanation": "No literature evidence for b1334 regulating b2415, yet CLR z-score of 4.2 and MI of 0.85 indicate strong statistical dependency. This could represent a novel regulatory interaction not yet characterized in RegulonDB. The high correlation (0.78) and significant p-value (0.001) strengthen this hypothesis.",
            "confidence": "medium",
            "recommendation": "Flag as potential discovery. Validate with ChIP-seq or reporter assays to confirm direct regulation."
        }
    ],
    "tf_level_notes": "FNR (b1334) shows context-dependent regulation. One edge exhibits condition-silent behavior explained by glucose conditions, while another represents a potential novel target. The TF is well-expressed (percentile 65), suggesting functional activity in this dataset.",
    "comparative_analysis": {
        "total_edges": 2,
        "validated_count": 0,
        "condition_silent_count": 1,
        "novel_hypothesis_count": 1,
        "pattern": "Context-dependent regulator with potential undiscovered targets"
    }
}"""

MOCK_LLM_RESPONSE_MULTIPLE_EDGES = """{
    "tf": "b1334",
    "edge_decisions": [
        {
            "edge_id": "b1334→b0929",
            "reconciliation_status": "Validated",
            "preserve_edge": true,
            "literature_support": "strong",
            "data_support": "strong",
            "context_compatible": true,
            "explanation": "Excellent agreement between literature and data. RegulonDB reports strong activation, and CLR z-score of 4.5 confirms this in the M3D dataset under matching aerobic conditions.",
            "confidence": "high",
            "recommendation": "High-confidence edge. Include in final network."
        },
        {
            "edge_id": "b1334→b1522",
            "reconciliation_status": "Active",
            "preserve_edge": true,
            "literature_support": "strong",
            "data_support": "strong",
            "context_compatible": false,
            "explanation": "Literature indicates repression, and CLR of 3.8 shows strong signal despite context mismatch. Edge is active but may behave differently than literature expectations.",
            "confidence": "medium",
            "recommendation": "Preserve with note about context difference."
        },
        {
            "edge_id": "b1334→b2415",
            "reconciliation_status": "Probable False Positive",
            "preserve_edge": false,
            "literature_support": "weak",
            "data_support": "none",
            "context_compatible": true,
            "explanation": "Weak literature evidence from low-throughput study, but CLR of 0.3 shows no statistical support. Context matches, suggesting this is likely a false positive from the original study.",
            "confidence": "medium",
            "recommendation": "Remove from network. Flag as zombie candidate."
        }
    ],
    "tf_level_notes": "Three edges with diverse reconciliation outcomes. Two validated/active edges support FNR's role, while one appears to be a false positive.",
    "comparative_analysis": {
        "total_edges": 3,
        "validated_count": 1,
        "active_count": 1,
        "false_positive_count": 1
    }
}"""

MOCK_LLM_RESPONSE_MALFORMED = """{
    "tf": "b1334",
    "edge_decisions": [
        {
            "edge_id": "b1334→b0929",
            "reconciliation_status": "Validated",
            "preserve_edge": true,
            "literature_support": "strong",
            "data_support": "strong",
            "context_compatible": true,
            "explanation": "Too short",
            "confidence": "high",
            "recommendation": "Keep"
        }
    ]
}"""


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_edges():
    """Sample edge data for LLM input."""
    return [
        {
            "edge_id": "b1334→b0929",
            "tf": "b1334",
            "target": "b0929",
            "literature": {
                "exists": True,
                "effect": "activation",
                "evidence_strength": "strong",
                "conditions_required": ["glucose_depletion"]
            },
            "statistics": {
                "clr_zscore": 0.3,
                "mi": 0.05,
                "correlation": 0.02
            },
            "context": {
                "match": False,
                "explanation": "Context mismatch: literature requires glucose depletion, dataset has high glucose"
            }
        },
        {
            "edge_id": "b1334→b2415",
            "tf": "b1334",
            "target": "b2415",
            "literature": {
                "exists": False,
                "effect": "unknown",
                "evidence_strength": "none",
                "conditions_required": []
            },
            "statistics": {
                "clr_zscore": 4.2,
                "mi": 0.85,
                "correlation": 0.78,
                "pvalue": 0.001
            },
            "context": {
                "match": False,
                "explanation": "No context info"
            }
        }
    ]


@pytest.fixture
def sample_state_with_tf():
    """Create state with TF data."""
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["analysis__tf_expression"] = {
        "b1334": {
            "is_expressed": True,
            "mean_expression": 5.2,
            "percentile": 65
        }
    }
    return state


# ============================================================================
# Test LLM Response Mocking
# ============================================================================

@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_invoke_llm_reviewer_with_mock(mock_create_llm, sample_edges, sample_state_with_tf):
    """Test LLM invocation with mocked response."""
    # Setup mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE_VALID
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    # Invoke
    result = invoke_llm_reviewer(sample_edges, "b1334", sample_state_with_tf)

    # Verify result structure
    assert isinstance(result, SubgraphReview)
    assert result.tf == "b1334"
    assert len(result.edge_decisions) == 2

    # Verify first edge decision
    edge1 = result.edge_decisions[0]
    assert edge1.edge_id == "b1334→b0929"
    assert edge1.reconciliation_status == "Condition-Silent"
    assert edge1.preserve_edge is True
    assert edge1.literature_support == "strong"
    assert edge1.data_support == "none"
    assert edge1.context_compatible is False
    assert len(edge1.explanation) >= 100  # Pydantic min_length validation

    # Verify second edge decision
    edge2 = result.edge_decisions[1]
    assert edge2.edge_id == "b1334→b2415"
    assert edge2.reconciliation_status == "Novel Hypothesis"
    assert edge2.data_support == "strong"

    # Verify TF-level notes
    assert "FNR" in result.tf_level_notes
    assert "context-dependent" in result.tf_level_notes.lower()

    # Verify comparative analysis
    assert result.comparative_analysis["total_edges"] == 2
    assert result.comparative_analysis["condition_silent_count"] == 1
    assert result.comparative_analysis["novel_hypothesis_count"] == 1


@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_invoke_llm_reviewer_multiple_statuses(mock_create_llm, sample_state_with_tf):
    """Test LLM with multiple edge statuses."""
    # Setup mock
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE_MULTIPLE_EDGES
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    # Create edges
    edges = [
        {
            "edge_id": "b1334→b0929",
            "tf": "b1334",
            "target": "b0929",
            "literature": {"exists": True, "evidence_strength": "strong"},
            "statistics": {"clr_zscore": 4.5, "mi": 0.8},
            "context": {"match": True}
        },
        {
            "edge_id": "b1334→b1522",
            "tf": "b1334",
            "target": "b1522",
            "literature": {"exists": True, "evidence_strength": "strong"},
            "statistics": {"clr_zscore": 3.8, "mi": 0.7},
            "context": {"match": False}
        },
        {
            "edge_id": "b1334→b2415",
            "tf": "b1334",
            "target": "b2415",
            "literature": {"exists": True, "evidence_strength": "weak"},
            "statistics": {"clr_zscore": 0.3, "mi": 0.02},
            "context": {"match": True}
        }
    ]

    result = invoke_llm_reviewer(edges, "b1334", sample_state_with_tf)

    # Verify all three statuses
    statuses = [d.reconciliation_status for d in result.edge_decisions]
    assert "Validated" in statuses
    assert "Active" in statuses
    assert "Probable False Positive" in statuses

    # Verify preserve_edge logic
    decisions_by_id = {d.edge_id: d for d in result.edge_decisions}
    assert decisions_by_id["b1334→b0929"].preserve_edge is True
    assert decisions_by_id["b1334→b1522"].preserve_edge is True
    assert decisions_by_id["b1334→b2415"].preserve_edge is False


@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_invoke_llm_reviewer_pydantic_validation_fails(mock_create_llm, sample_edges, sample_state_with_tf):
    """Test that malformed LLM output raises validation error."""
    # Setup mock with invalid response (explanation too short)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE_MALFORMED
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    # Should raise ValidationError due to min_length constraint on explanation
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        invoke_llm_reviewer(sample_edges, "b1334", sample_state_with_tf)


# ============================================================================
# Test End-to-End with Mock LLM
# ============================================================================

@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_reviewer_agent_node_with_mock_llm_success(mock_create_llm):
    """Test full node execution with successful LLM response."""
    # Setup mock
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE_VALID
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    # Create state
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {
                "exists": True,
                "effect": "activation",
                "evidence_strength": "strong",
                "conditions_required": ["glucose_depletion"]
            }
        }
    }
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {"clr_zscore": 0.3, "mi": 0.05},
            "b2415": {"clr_zscore": 4.2, "mi": 0.85}
        }
    }
    state["research__annotations"] = {
        "b1334→b0929": {"match": False, "explanation": "Context mismatch"}
    }
    state["analysis__tf_expression"] = {
        "b1334": {"is_expressed": True, "mean_expression": 5.2, "percentile": 65}
    }

    # Execute
    result = reviewer_agent_node(state)

    # Verify LLM method was used (check message)
    assert "LLM" in result["messages"][-1].content or "llm" in result["messages"][-1].content

    # Verify decisions
    assert len(result["reviewer__edge_decisions"]) == 2

    # Verify special cases identified
    assert len(result["reviewer__novel_hypotheses"]) == 1
    assert result["reviewer__novel_hypotheses"][0]["edge_id"] == "b1334→b2415"


@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_reviewer_agent_node_llm_fails_uses_fallback(mock_create_llm):
    """Test that node falls back to rules when LLM fails."""
    # Setup mock to raise exception
    mock_create_llm.side_effect = Exception("LLM authentication failed")

    # Create state
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {"exists": True, "evidence_strength": "strong"}
        }
    }
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {"clr_zscore": 0.5, "mi": 0.1}
        }
    }
    state["research__annotations"] = {
        "b1334→b0929": {"match": False, "explanation": "Mismatch"}
    }

    # Execute - should not raise, should use fallback
    result = reviewer_agent_node(state)

    # Verify fallback was used (check message)
    assert "RULES" in result["messages"][-1].content or "rules" in result["messages"][-1].content

    # Should still produce results
    assert len(result["reviewer__edge_decisions"]) > 0


# ============================================================================
# Test Prompt Construction
# ============================================================================

def test_format_edge_data_for_llm_includes_all_info(sample_edges):
    """Test that LLM prompt formatting includes all necessary data."""
    formatted = format_edge_data_for_llm(sample_edges)

    # Check all edge IDs present
    assert "b1334→b0929" in formatted
    assert "b1334→b2415" in formatted

    # Check literature info
    assert "activation" in formatted
    assert "glucose_depletion" in formatted

    # Check statistics
    assert "0.3" in formatted or "0.30" in formatted  # CLR for first edge
    assert "4.2" in formatted or "4.20" in formatted  # CLR for second edge
    assert "0.85" in formatted  # MI for second edge

    # Check context
    assert "Context mismatch" in formatted

    # Check structure
    assert "### Edge 1:" in formatted
    assert "### Edge 2:" in formatted
    assert "Literature:" in formatted
    assert "Statistical Analysis:" in formatted
    assert "Context:" in formatted


def test_format_tf_expression_includes_all_fields():
    """Test TF expression formatting."""
    tf_expr = {
        "is_expressed": True,
        "mean_expression": 5.2,
        "percentile": 65
    }

    formatted = format_tf_expression(tf_expr)

    assert "True" in formatted
    assert "5.2" in formatted
    assert "65" in formatted


def test_format_tf_expression_handles_missing_data():
    """Test TF expression formatting with missing data."""
    tf_expr = {}

    formatted = format_tf_expression(tf_expr)

    # Should have default values
    assert "Unknown" in formatted or "N/A" in formatted


# ============================================================================
# Test Structured Output Models
# ============================================================================

def test_edge_decision_output_model_validation():
    """Test Pydantic model validation for EdgeDecisionOutput."""
    # Valid decision
    decision = EdgeDecisionOutput(
        edge_id="b1334→b0929",
        reconciliation_status="Validated",
        preserve_edge=True,
        literature_support="strong",
        data_support="strong",
        context_compatible=True,
        explanation="This is a valid explanation that is longer than 100 characters and provides sufficient detail about the biological reasoning behind the decision.",
        confidence="high",
        recommendation="Keep this edge"
    )

    assert decision.edge_id == "b1334→b0929"
    assert decision.preserve_edge is True


def test_edge_decision_output_explanation_too_short_fails():
    """Test that explanation < 100 chars fails validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EdgeDecisionOutput(
            edge_id="b1334→b0929",
            reconciliation_status="Validated",
            preserve_edge=True,
            literature_support="strong",
            data_support="strong",
            context_compatible=True,
            explanation="Too short",  # < 100 chars
            confidence="high",
            recommendation=""
        )


def test_edge_decision_output_explanation_too_long_fails():
    """Test that explanation > 1500 chars fails validation."""
    from pydantic import ValidationError

    long_explanation = "A" * 1501  # 1501 characters

    with pytest.raises(ValidationError):
        EdgeDecisionOutput(
            edge_id="b1334→b0929",
            reconciliation_status="Validated",
            preserve_edge=True,
            literature_support="strong",
            data_support="strong",
            context_compatible=True,
            explanation=long_explanation,
            confidence="high",
            recommendation=""
        )


def test_subgraph_review_model_validation():
    """Test SubgraphReview model with nested decisions."""
    review = SubgraphReview(
        tf="b1334",
        edge_decisions=[
            EdgeDecisionOutput(
                edge_id="b1334→b0929",
                reconciliation_status="Validated",
                preserve_edge=True,
                literature_support="strong",
                data_support="strong",
                context_compatible=True,
                explanation="Valid explanation with sufficient length to pass the minimum character requirement for Pydantic validation.",
                confidence="high",
                recommendation="Keep"
            )
        ],
        tf_level_notes="TF-level summary",
        comparative_analysis={"total_edges": 1}
    )

    assert review.tf == "b1334"
    assert len(review.edge_decisions) == 1
    assert isinstance(review.edge_decisions[0], EdgeDecisionOutput)


# ============================================================================
# Test LLM Prompt Construction (No Mock)
# ============================================================================

@patch('src.nodes.reviewer_agent.create_argonne_llm')
def test_llm_prompt_includes_system_message(mock_create_llm, sample_edges, sample_state_with_tf):
    """Test that LLM is called with proper system prompt."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = MOCK_LLM_RESPONSE_VALID
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    invoke_llm_reviewer(sample_edges, "b1334", sample_state_with_tf)

    # Verify invoke was called
    assert mock_llm.invoke.called

    # Get the prompt that was passed
    call_args = mock_llm.invoke.call_args
    messages = call_args[0][0]

    # Check system message content
    system_content = str(messages[0].content)
    assert "Reviewer Agent" in system_content
    assert "reconciliation" in system_content.lower()
    assert "Validated" in system_content  # Decision categories

    # Check human message content
    human_content = str(messages[1].content)
    assert "b1334" in human_content
    assert "Review this transcription factor" in human_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
