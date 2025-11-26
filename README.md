# GRN Inference via Multi-Agent LangGraph

1. Iterative loop architecture

Top-level flow (one “round”):

ResearchAgent  →  AnalysisAgent  →  ReviewerAgent  →  EvaluatorAgent
      ↑                                                           │
      └────────────────────── loop if not done ───────────────────┘

	•	ResearchAgent: propose new/updated hypotheses (possibly using RAG)
	•	AnalysisAgent: add stats / support scores using tools
	•	ReviewerAgent: update the global weighted adjacency matrix
	•	EvaluatorAgent: compute metrics vs. DREAM4 gold standard, decide done or continue

⸻

2. Shared state model (the contract everyone codes against)

# src/state/grn_state.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Hypothesis(BaseModel):
    """One candidate regulatory edge with optional stats."""
    source_gene: str
    target_gene: str
    interaction_type: str  # "activation" | "repression" | "unknown"
    rationale: Optional[str] = None  # natural-language explanation

    # Filled/updated by AnalysisAgent
    stats: Optional[Dict[str, float]] = None  # e.g. {"partial_corr": 0.78, "p_value": 0.002}
    support_score: Optional[float] = None     # e.g. 0–1 aggregated confidence


class GRNState(BaseModel):
    """Global state passed around the LangGraph."""
    # --- static data (loaded at start) ---
    genes: List[str]                      # ["G1", "G2", ..., "G100"]
    expression_path: str                  # e.g. "data/dream4_100_01_expression.npy"
    metadata_path: Optional[str] = None   # optional path to perturbation metadata
    gold_adj_path: Optional[str] = None   # path to goldStandardAdjacencyMatrix for evaluator

    # --- dynamic data (agents read/write) ---
    candidate_hypotheses: List[Hypothesis] = []
    validated_hypotheses: List[Hypothesis] = []

    # adjacency_weights: "G3->G7" → 0.87
    adjacency_weights: Dict[str, float] = {}

    # evaluation metrics (AUROC, AUPR, F1, etc.)
    eval_metrics: Dict[str, float] = {}

    # logging / debugging
    logs: List[str] = []

    # loop control
    iteration: int = 0
    max_iterations: int = 5
    done: bool = False

🔑 Rule of thumb: nobody mutates raw numpy arrays in state. You pass paths and each node loads from disk as needed (or uses a shared loader utility) to keep state lean and JSON-serializable.

⸻

3. LangGraph skeleton with stubbed nodes

# src/graph/grn_graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state.grn_state import GRNState


# ---------- Node stubs (to be implemented by each teammate) ----------

def research_agent_node(state: GRNState) -> GRNState:
    """
    Owner: Research Agent person.
    Responsibilities:
      - Read genes + expression_path (+ RAG docs if you add those).
      - Read existing adjacency_weights / validated_hypotheses if you want to
        bias new hypotheses.
      - Write/extend state.candidate_hypotheses with Hypothesis objects
        (at minimum: source_gene, target_gene, interaction_type, rationale, initial guess).
    """
    # TODO: implement
    state.logs.append(f"[iter {state.iteration}] ResearchAgent: proposed X hypotheses")
    return state


def analysis_agent_node(state: GRNState) -> GRNState:
    """
    Owner: Analysis Agent person.
    Responsibilities:
      - Take state.candidate_hypotheses as input.
      - Load expression data from state.expression_path (and metadata if needed).
      - For each Hypothesis, run statistical / causal tests via MCP tools.
      - Fill Hypothesis.stats and Hypothesis.support_score.
      - Write results to state.validated_hypotheses (can overwrite or extend).
    """
    # TODO: implement
    state.logs.append(f"[iter {state.iteration}] AnalysisAgent: validated Y hypotheses")
    return state


def reviewer_agent_node(state: GRNState) -> GRNState:
    """
    Owner: Reviewer Agent person.
    Responsibilities:
      - Read state.validated_hypotheses.
      - Apply thresholds, sparsity constraints, motif logic, etc.
      - Update state.adjacency_weights as {\"Gi->Gj\": weight}.
      - Optionally prune validated_hypotheses.
    """
    # TODO: implement
    state.logs.append(f"[iter {state.iteration}] ReviewerAgent: adjacency updated with Z edges")
    return state


def evaluator_agent_node(state: GRNState) -> GRNState:
    """
    Owner: Evaluator Agent person.
    Responsibilities:
      - Load gold-standard adjacency from state.gold_adj_path.
      - Compare state.adjacency_weights to gold standard.
      - Compute AUROC, AUPR, F1, etc. and store in state.eval_metrics.
      - Decide whether to stop:
          - set state.done = True if convergence or max_iterations reached,
          - else keep state.done = False.
      - Increment iteration counter.
    """
    # TODO: implement metrics & stopping logic
    state.iteration += 1

    # Example stopping rule:
    if state.iteration >= state.max_iterations:
        state.done = True

    state.logs.append(f"[iter {state.iteration}] EvaluatorAgent: metrics={state.eval_metrics}, done={state.done}")
    return state


# ---------- Graph construction ----------

def _continue_or_end(state: GRNState):
    """Small router function used by add_conditional_edges."""
    if state.done:
        return END
    else:
        return "research_agent"


def build_grn_graph():
    graph = StateGraph(GRNState)

    # register nodes
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("reviewer_agent", reviewer_agent_node)
    graph.add_node("evaluator_agent", evaluator_agent_node)

    # set entry point → first step in the loop
    graph.set_entry_point("research_agent")

    # linear flow within one iteration:
    graph.add_edge("research_agent", "analysis_agent")
    graph.add_edge("analysis_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", "evaluator_agent")

    # conditional edge after evaluator: either loop or end
    graph.add_conditional_edges(
        "evaluator_agent",
        _continue_or_end,
        {
            "research_agent": "research_agent",  # keep going
            END: END,                            # stop
        },
    )

    return graph.compile()

Usage example (e.g., main.py):

from src.graph.grn_graph import build_grn_graph
from src.state.grn_state import GRNState
from src.data.dream4_loader import load_dream4  # you'll implement

def main():
    genes, expression_path, metadata_path, gold_adj_path = load_dream4("dream4_010_01")

    initial_state = GRNState(
        genes=genes,
        expression_path=expression_path,
        metadata_path=metadata_path,
        gold_adj_path=gold_adj_path,
        max_iterations=5,
    )

    app = build_grn_graph()
    final_state = app.invoke(initial_state)

    print("Final metrics:", final_state.eval_metrics)
    print("Num edges inferred:", len(final_state.adjacency_weights))

if __name__ == "__main__":
    main()


⸻

4. API outline / who owns what (for async work)

Here’s the division of labor in API terms:

Teammate 1 – Research Agent

Implements research_agent_node(state: GRNState) -> GRNState:
	•	Reads:
	•	state.genes
	•	state.expression_path (optional; you might compute summary stats)
	•	optionally state.adjacency_weights / state.validated_hypotheses to avoid re-proposing edges
	•	Writes:
	•	state.candidate_hypotheses (list of Hypothesis objects with minimal fields filled)
	•	Extra files:
	•	src/agents/research_agent.py for prompts + RAG utilities

Promise: candidate_hypotheses must be a list of valid Hypothesis objects.

⸻

Teammate 2 – Analysis Agent

Implements analysis_agent_node(state: GRNState) -> GRNState:
	•	Reads:
	•	state.candidate_hypotheses
	•	state.expression_path (+ metadata_path if needed)
	•	Uses tools:
	•	MCP-exposed functions: run_regression, compute_partial_corr, etc.
	•	Writes:
	•	For each Hypothesis in the list: fill stats and support_score
	•	Fill state.validated_hypotheses (probably a copy of candidate list with stats filled)

Promise: Each hypothesis in validated_hypotheses has .support_score ∈ [0,1].

⸻

Teammate 3 – Reviewer + Evaluator + Infra

Implements:
	1.	reviewer_agent_node(state: GRNState) -> GRNState:
	•	Reads: state.validated_hypotheses
	•	Applies thresholds, sparsity, etc.
	•	Writes: state.adjacency_weights ("Gi->Gj" → float)
	2.	evaluator_agent_node(state: GRNState) -> GRNState:
	•	Reads: state.adjacency_weights, state.gold_adj_path, state.iteration, state.max_iterations
	•	Loads gold adjacency; computes metrics; writes to state.eval_metrics
	•	Sets state.done and increments state.iteration
	3.	Owns:
	•	GRNState / Hypothesis definitions
	•	build_grn_graph() and wiring (already sketched)
	•	load_dream4() util in src/data/dream4_loader.py

Promise: When state.done=True, metrics are meaningful and adjacency_weights is “final”.

