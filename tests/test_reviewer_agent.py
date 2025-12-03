"""
Unit Tests for Reviewer Agent

Tests the decision logic, data preparation, and state management
of the Reviewer Agent node.
"""

import pytest
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from src.state import AgentState, create_initial_state
from src.nodes.reviewer_agent import (
    reviewer_agent_node,
    prepare_subgraph_data,
    apply_decision_tree,
    classify_data_strength,
    apply_rule_based_review,
    format_edge_data_for_llm,
    post_process_decisions,
    SubgraphReview,
    EdgeDecisionOutput,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def empty_state() -> AgentState:
    """Create an empty initial state."""
    return create_initial_state()


@pytest.fixture
def sample_state_single_tf() -> AgentState:
    """Create state with one TF and sample data for testing."""
    state = create_initial_state()

    # Set current TF
    state["batch__current_tfs"] = ["b1334"]

    # Literature edges for b1334
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {
                "exists": True,
                "effect": "activation",
                "evidence_strength": "strong",
                "conditions_required": ["glucose_depletion"]
            },
            "b1522": {
                "exists": True,
                "effect": "repression",
                "evidence_strength": "weak",
                "conditions_required": []
            },
            "b2415": {
                "exists": False,
                "effect": "unknown",
                "evidence_strength": "none",
                "conditions_required": []
            }
        }
    }

    # Statistical results
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {
                "clr_zscore": 0.3,
                "mi": 0.05,
                "correlation": 0.02,
                "pvalue": 0.85
            },
            "b1522": {
                "clr_zscore": 0.8,
                "mi": 0.12,
                "correlation": -0.05,
                "pvalue": 0.62
            },
            "b2415": {
                "clr_zscore": 4.2,
                "mi": 0.85,
                "correlation": 0.78,
                "pvalue": 0.001
            },
            "b3891": {
                "clr_zscore": 3.5,
                "mi": 0.65,
                "correlation": 0.62,
                "pvalue": 0.005
            }
        }
    }

    # Context annotations
    state["research__annotations"] = {
        "b1334→b0929": {
            "literature_conditions": ["glucose_depletion"],
            "dataset_conditions": ["high_glucose"],
            "match": False,
            "explanation": "Context mismatch: literature requires glucose depletion, dataset has high glucose"
        },
        "b1334→b1522": {
            "literature_conditions": [],
            "dataset_conditions": ["high_glucose"],
            "match": True,
            "explanation": "No specific conditions required"
        },
        "b1334→b3891": {
            "literature_conditions": ["aerobic"],
            "dataset_conditions": ["aerobic"],
            "match": True,
            "explanation": "Conditions match: both aerobic"
        }
    }

    # TF expression
    state["analysis__tf_expression"] = {
        "b1334": {
            "is_expressed": True,
            "mean_expression": 5.2,
            "percentile": 65
        }
    }

    return state


# ============================================================================
# Test Decision Tree Logic
# ============================================================================

def test_decision_tree_validated():
    """Test Validated: Strong lit + Strong data + Context match"""
    lit = {
        "exists": True,
        "evidence_strength": "strong",
        "effect": "activation"
    }
    stats = {"clr_zscore": 4.5, "mi": 0.8}
    ctx = {"match": True}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Validated"
    assert "4.5" in explanation or "4.50" in explanation
    assert "match" in explanation.lower()


def test_decision_tree_active():
    """Test Active: Strong lit + Strong data + Context mismatch"""
    lit = {
        "exists": True,
        "evidence_strength": "strong",
        "effect": "repression"
    }
    stats = {"clr_zscore": 3.8, "mi": 0.7}
    ctx = {"match": False, "explanation": "Different conditions"}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Active"
    assert "mismatch" in explanation.lower()


def test_decision_tree_condition_silent_no_context_match():
    """Test Condition-Silent: Lit exists + Weak data + Context mismatch"""
    lit = {
        "exists": True,
        "evidence_strength": "strong",
        "effect": "activation"
    }
    stats = {"clr_zscore": 0.5, "mi": 0.1}
    ctx = {
        "match": False,
        "explanation": "Context mismatch: literature requires anaerobic, dataset is aerobic"
    }

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Condition-Silent"
    assert "context mismatch" in explanation.lower()


def test_decision_tree_condition_silent_strong_lit():
    """Test Condition-Silent: Strong lit + Weak data + Context match"""
    lit = {
        "exists": True,
        "evidence_strength": "strong",
        "effect": "activation"
    }
    stats = {"clr_zscore": 1.2, "mi": 0.15}
    ctx = {"match": True}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Condition-Silent"
    assert "strong lit" in explanation.lower() or "specific conditions" in explanation.lower()


def test_decision_tree_probable_false_positive():
    """Test Probable False Positive: Weak lit + No data + Context match"""
    lit = {
        "exists": True,
        "evidence_strength": "weak",
        "effect": "unknown"
    }
    stats = {"clr_zscore": 0.3, "mi": 0.02}
    ctx = {"match": True}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Probable False Positive"
    assert "false positive" in explanation.lower()


def test_decision_tree_novel_hypothesis():
    """Test Novel Hypothesis: No lit + Strong data"""
    lit = {
        "exists": False,
        "evidence_strength": "none"
    }
    stats = {"clr_zscore": 5.2, "mi": 0.9}
    ctx = {"match": False}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Novel Hypothesis"
    assert "no supporting literature" in explanation.lower() or "strong statistical" in explanation.lower()


def test_decision_tree_unsupported():
    """Test Unsupported: No lit + No data"""
    lit = {
        "exists": False,
        "evidence_strength": "none"
    }
    stats = {"clr_zscore": 0.2, "mi": 0.01}
    ctx = {"match": False}

    status, explanation = apply_decision_tree(lit, stats, ctx)

    assert status == "Unsupported"
    assert "no evidence" in explanation.lower() or "no lit" in explanation.lower()


# ============================================================================
# Test Data Strength Classification
# ============================================================================

def test_classify_data_strength_strong():
    """Test strong data classification (CLR > 3.0)"""
    assert classify_data_strength(4.5) == "strong"
    assert classify_data_strength(3.1) == "strong"
    assert classify_data_strength(10.0) == "strong"


def test_classify_data_strength_weak():
    """Test weak data classification (1.0 < CLR <= 3.0)"""
    assert classify_data_strength(2.5) == "weak"
    assert classify_data_strength(1.5) == "weak"
    assert classify_data_strength(3.0) == "weak"


def test_classify_data_strength_none():
    """Test no data classification (CLR <= 1.0)"""
    assert classify_data_strength(0.5) == "none"
    assert classify_data_strength(0.0) == "none"
    assert classify_data_strength(1.0) == "none"


# ============================================================================
# Test Data Preparation
# ============================================================================

def test_prepare_subgraph_data(sample_state_single_tf):
    """Test subgraph data preparation combines all sources"""
    edges = prepare_subgraph_data(sample_state_single_tf, "b1334")

    # Should have 4 edges (3 from literature + 1 data-only)
    assert len(edges) == 4

    # Check edge IDs
    edge_ids = {edge["edge_id"] for edge in edges}
    assert "b1334→b0929" in edge_ids
    assert "b1334→b1522" in edge_ids
    assert "b1334→b2415" in edge_ids
    assert "b1334→b3891" in edge_ids

    # Check data structure
    for edge in edges:
        assert "edge_id" in edge
        assert "tf" in edge
        assert "target" in edge
        assert "literature" in edge
        assert "statistics" in edge
        assert "context" in edge


def test_prepare_subgraph_data_handles_missing_data(empty_state):
    """Test data preparation with missing upstream data"""
    empty_state["batch__current_tfs"] = ["b9999"]

    edges = prepare_subgraph_data(empty_state, "b9999")

    assert len(edges) == 0


def test_format_edge_data_for_llm():
    """Test LLM prompt formatting"""
    edges = [
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
                "explanation": "Context mismatch"
            }
        }
    ]

    formatted = format_edge_data_for_llm(edges)

    assert "b1334→b0929" in formatted
    assert "activation" in formatted
    assert "0.3" in formatted or "0.30" in formatted
    assert "glucose_depletion" in formatted
    assert "Context mismatch" in formatted


# ============================================================================
# Test Rule-Based Review (Fallback)
# ============================================================================

def test_apply_rule_based_review(sample_state_single_tf):
    """Test complete rule-based review for a TF"""
    edges = prepare_subgraph_data(sample_state_single_tf, "b1334")

    review = apply_rule_based_review(edges, "b1334", sample_state_single_tf)

    assert isinstance(review, SubgraphReview)
    assert review.tf == "b1334"
    assert len(review.edge_decisions) == 4
    assert "Rule-based fallback" in review.tf_level_notes

    # Check that all decisions are EdgeDecisionOutput instances
    for decision in review.edge_decisions:
        assert isinstance(decision, EdgeDecisionOutput)
        assert hasattr(decision, "edge_id")
        assert hasattr(decision, "reconciliation_status")
        assert hasattr(decision, "explanation")


def test_rule_based_review_categorizes_correctly(sample_state_single_tf):
    """Test that rule-based review produces correct categories"""
    edges = prepare_subgraph_data(sample_state_single_tf, "b1334")
    review = apply_rule_based_review(edges, "b1334", sample_state_single_tf)

    # Find specific edges and check their status
    decisions_by_edge = {d.edge_id: d for d in review.edge_decisions}

    # b1334→b0929: Strong lit + Weak data + Context mismatch → Condition-Silent
    assert decisions_by_edge["b1334→b0929"].reconciliation_status == "Condition-Silent"

    # b1334→b2415: No lit + Strong data → Novel Hypothesis
    assert decisions_by_edge["b1334→b2415"].reconciliation_status == "Novel Hypothesis"

    # b1334→b3891: No lit record (not in research__literature_edges) + Strong data → Novel Hypothesis
    assert decisions_by_edge["b1334→b3891"].reconciliation_status == "Novel Hypothesis"


# ============================================================================
# Test Post-Processing
# ============================================================================

def test_post_process_decisions_updates_state():
    """Test post-processing correctly updates state"""
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    messages = []

    # Create a sample review result
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
                explanation="Strong evidence from both literature and experimental data with matching context conditions. "
                           "Literature reports strong activation, and statistical analysis confirms with high CLR score. "
                           "This edge shows excellent agreement across all evidence sources.",
                confidence="high",
                recommendation="Keep this edge in the final network."
            ),
            EdgeDecisionOutput(
                edge_id="b1334→b2415",
                reconciliation_status="Novel Hypothesis",
                preserve_edge=True,
                literature_support="none",
                data_support="strong",
                context_compatible=False,
                explanation="Strong statistical evidence indicates regulatory relationship but no literature support exists. "
                           "This represents a potential new discovery worth experimental validation. "
                           "The high correlation suggests direct or strong indirect regulation.",
                confidence="medium",
                recommendation="Validate experimentally."
            ),
            EdgeDecisionOutput(
                edge_id="b1334→b1522",
                reconciliation_status="Probable False Positive",
                preserve_edge=False,
                literature_support="weak",
                data_support="none",
                context_compatible=True,
                explanation="Weak literature evidence combined with no statistical support despite matching contexts. "
                           "This edge likely represents a false positive from the original literature study. "
                           "Consider removing from network or flagging for experimental validation.",
                confidence="medium",
                recommendation="Remove from network."
            )
        ],
        tf_level_notes="Review completed successfully",
        comparative_analysis={}
    )

    result = post_process_decisions(review, state, "b1334", "llm", messages)

    # Check messages updated
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)

    # Check edge decisions
    assert len(result["reviewer__edge_decisions"]) == 3

    # Check TF summaries
    assert "b1334" in result["reviewer__tf_summaries"]

    # Check reconciliation log accumulated
    assert len(result["reviewer__reconciliation_log"]) == 3

    # Check zombie candidates identified
    assert len(result["reviewer__zombie_candidates"]) == 1
    assert result["reviewer__zombie_candidates"][0]["edge_id"] == "b1334→b1522"

    # Check novel hypotheses identified
    assert len(result["reviewer__novel_hypotheses"]) == 1
    assert result["reviewer__novel_hypotheses"][0]["edge_id"] == "b1334→b2415"


# ============================================================================
# Test Main Node Function
# ============================================================================

def test_reviewer_agent_node_no_tfs(empty_state):
    """Test reviewer node handles empty TF list gracefully"""
    result = reviewer_agent_node(empty_state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "No TFs to process" in result["messages"][0].content


def test_reviewer_agent_node_processes_tf(sample_state_single_tf):
    """Test reviewer node processes a TF successfully"""
    result = reviewer_agent_node(sample_state_single_tf)

    # Check messages
    assert "messages" in result
    assert len(result["messages"]) == 1

    # Check edge decisions created
    assert "reviewer__edge_decisions" in result
    assert len(result["reviewer__edge_decisions"]) > 0

    # Check TF summaries
    assert "reviewer__tf_summaries" in result
    assert "b1334" in result["reviewer__tf_summaries"]

    # Check reconciliation log
    assert "reviewer__reconciliation_log" in result

    # Check that decisions have required fields
    for decision in result["reviewer__edge_decisions"]:
        assert "edge_id" in decision
        assert "reconciliation_status" in decision
        assert "explanation" in decision
        assert "preserve_edge" in decision


def test_reviewer_agent_node_uses_fallback_on_missing_llm():
    """Test that node uses rule-based fallback when LLM unavailable"""
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {
                "exists": True,
                "evidence_strength": "strong"
            }
        }
    }
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {
                "clr_zscore": 4.0,
                "mi": 0.7
            }
        }
    }
    state["research__annotations"] = {
        "b1334→b0929": {
            "match": True,
            "explanation": "Matches"
        }
    }

    # Should use fallback since LLM may not be available in test environment
    result = reviewer_agent_node(state)

    # Should still produce results
    assert "reviewer__edge_decisions" in result
    assert len(result["reviewer__edge_decisions"]) > 0


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_decision_tree_boundary_clr_exactly_3():
    """Test CLR exactly at threshold (3.0)"""
    lit = {"exists": True, "evidence_strength": "strong"}
    stats = {"clr_zscore": 3.0, "mi": 0.5}
    ctx = {"match": True}

    status, _ = apply_decision_tree(lit, stats, ctx)

    # CLR > 3.0 check should be False for exactly 3.0, so not Validated
    assert status in ["Condition-Silent", "Active"]


def test_decision_tree_boundary_clr_just_over_3():
    """Test CLR just over threshold"""
    lit = {"exists": True, "evidence_strength": "strong"}
    stats = {"clr_zscore": 3.01, "mi": 0.5}
    ctx = {"match": True}

    status, _ = apply_decision_tree(lit, stats, ctx)

    assert status == "Validated"


def test_prepare_subgraph_data_with_partial_annotations():
    """Test data preparation when some edges lack context annotations"""
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {"exists": True, "evidence_strength": "strong"}
        }
    }
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {"clr_zscore": 4.0, "mi": 0.7}
        }
    }
    # No annotations provided
    state["research__annotations"] = {}

    edges = prepare_subgraph_data(state, "b1334")

    assert len(edges) == 1
    # Should have default context data
    assert edges[0]["context"]["match"] is False


def test_reviewer_node_preserves_existing_logs():
    """Test that reviewer node accumulates logs across batches"""
    state = create_initial_state()
    state["batch__current_tfs"] = ["b1334"]
    state["research__literature_edges"] = {
        "b1334": {
            "b0929": {"exists": True, "evidence_strength": "strong"}
        }
    }
    state["analysis__statistical_results"] = {
        "b1334": {
            "b0929": {"clr_zscore": 4.0, "mi": 0.7}
        }
    }
    state["research__annotations"] = {
        "b1334→b0929": {"match": True, "explanation": "Test"}
    }

    # Pre-existing logs
    state["reviewer__reconciliation_log"] = [
        {"edge_id": "old_edge", "status": "old_status"}
    ]
    state["reviewer__zombie_candidates"] = [
        {"edge_id": "old_zombie"}
    ]

    result = reviewer_agent_node(state)

    # Should have accumulated (old + new)
    assert len(result["reviewer__reconciliation_log"]) > 1
    assert result["reviewer__reconciliation_log"][0]["edge_id"] == "old_edge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
