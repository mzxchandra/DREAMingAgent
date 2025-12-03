# DREAMing Agent: Biological Network Reconciliation System

Multi-agent system for reconciling curated literature knowledge (RegulonDB) with high-throughput expression data (M3D) using LangGraph and LLM-powered decision making.

---

## System Architecture

```
Loader → Batch Manager → Research Agent → Analysis Agent → Reviewer Agent → (loop)
```

**Current Implementation Status:**
- ✅ **State Schema** ([src/state.py](src/state.py)) - Namespaced flat state design
- ✅ **Reviewer Agent** ([src/nodes/reviewer_agent.py](src/nodes/reviewer_agent.py)) - LLM + rule-based reconciliation
- ⏳ **Other Agents** - To be implemented

---

## Reviewer Agent

The Reviewer Agent synthesizes statistical evidence with biological literature to classify gene regulatory edges into 7 categories:

1. **Validated** - Strong literature + strong data + context match
2. **Active** - Strong data despite context mismatch
3. **Condition-Silent** - Literature supported but weak data (context explains)
4. **Inactive-Context-Mismatch** - No activity due to wrong conditions
5. **Novel Hypothesis** - Strong data, no literature (potential discovery)
6. **Probable False Positive** - Weak literature, no data (zombie science)
7. **Unsupported** - No evidence from either source

### Inputs

The Reviewer Agent reads from AgentState:

```python
# From Batch Manager
batch__current_tfs: List[str]  # TFs to process (e.g., ["b1334"])

# From Research Agent
research__literature_edges: Dict[str, Dict[str, Dict]]
# Structure: {tf_bnumber: {target_bnumber: {exists, effect, evidence_strength, ...}}}

research__annotations: Dict[str, Dict]
# Structure: {edge_id: {match, explanation, literature_conditions, dataset_conditions}}

# From Analysis Agent
analysis__statistical_results: Dict[str, Dict[str, Dict]]
# Structure: {tf_bnumber: {target_bnumber: {clr_zscore, mi, correlation, pvalue}}}

analysis__tf_expression: Dict[str, Dict]
# Structure: {tf_bnumber: {is_expressed, mean_expression, percentile}}
```

### Outputs

The Reviewer Agent updates AgentState:

```python
# Current batch results
reviewer__edge_decisions: List[Dict]           # All decisions for current TF
reviewer__tf_summaries: Dict[str, str]         # TF-level analysis notes
reviewer__comparative_analysis: Dict           # Cross-edge patterns

# Accumulated across batches
reviewer__reconciliation_log: List[Dict]       # All decisions (append)
reviewer__zombie_candidates: List[Dict]        # Probable false positives
reviewer__novel_hypotheses: List[Dict]         # Strong data, weak lit
```

### Example Usage

```python
from src.state import create_initial_state
from src.nodes.reviewer_agent import reviewer_agent_node

# Create state with sample data
state = create_initial_state()
state["batch__current_tfs"] = ["b1334"]
state["research__literature_edges"] = {
    "b1334": {
        "b0929": {
            "exists": True,
            "effect": "activation",
            "evidence_strength": "strong"
        }
    }
}
state["analysis__statistical_results"] = {
    "b1334": {
        "b0929": {"clr_zscore": 4.2, "mi": 0.8}
    }
}

# Run reviewer
result = reviewer_agent_node(state)
print(result["reviewer__edge_decisions"])
```

---

## Testing

### Run All Tests

```bash
# Activate virtual environment
source ../aienv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_reviewer_agent.py -v          # Rule-based tests (23 tests)
pytest tests/test_reviewer_agent_llm.py -v      # LLM integration tests (13 tests)
```

### Test Coverage

**[tests/test_reviewer_agent.py](tests/test_reviewer_agent.py)** (23 tests)
- Decision tree logic for all 7 categories
- Data strength classification (CLR thresholds)
- Subgraph data preparation
- Rule-based fallback
- Post-processing and state updates
- Edge cases and boundary conditions

**[tests/test_reviewer_agent_llm.py](tests/test_reviewer_agent_llm.py)** (13 tests)
- LLM invocation with mocked responses
- Pydantic structured output validation
- Prompt construction
- Graceful fallback when LLM fails
- Explanation length validation (100-1500 chars)

---

## Project Structure

```
DREAMingAgent/
├── README.md
├── requirements.txt
├── src/
│   ├── state.py                    # AgentState schema
│   ├── llm_config.py               # Argonne LLM initialization
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── reviewer_agent.py       # Reviewer Agent implementation
│   └── utils/
│       └── inference_auth_token.py # Globus authentication
└── tests/
    ├── __init__.py
    ├── test_reviewer_agent.py      # Rule-based tests
    └── test_reviewer_agent_llm.py  # LLM integration tests
```

---

## Dependencies

Core:
- `langgraph>=0.2.0` - Workflow orchestration
- `langchain>=0.3.0`, `langchain-openai>=0.1.0` - LLM integration
- `pydantic>=2.0.0` - Structured output validation

Data & Analysis:
- `pandas>=2.0.0` - Data processing
- `networkx>=3.0` - Graph structures
- `scikit-learn>=1.3.0` - Statistical analysis

Auth & Testing:
- `globus-sdk>=3.0.0` - Argonne API authentication
- `pytest>=7.0.0`, `pytest-mock>=3.12.0` - Testing

---

## Key Features

### Hybrid Decision System
- **Primary**: Argonne-hosted LLM with structured JSON output (Pydantic validation)
- **Fallback**: Deterministic rule-based decision tree
- **Temperature**: 0.1 for reproducibility
- **Graceful degradation**: Never fails completely

### Subgraph-Level Processing
Processes one TF + all its targets together for:
- Cross-edge comparative analysis
- TF-level pattern detection
- Zombie science identification

### Scientific Explanations
All decisions include 100-1500 character scientific justifications with:
- Specific CLR scores and MI values
- Context mismatch explanations
- Regulatory mechanism hypotheses

---

## Next Steps

1. Implement remaining agents (Loader, Batch Manager, Research, Analysis)
2. Create workflow.py for LangGraph orchestration
3. Add end-to-end integration tests
4. Test with real RegulonDB + M3D data
