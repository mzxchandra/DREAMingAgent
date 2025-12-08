# DREAMing Agent

**Agentic Reconciliation of Biological Literature and High-Throughput Data**

An autonomous AI-powered system that reconciles curated literature knowledge (RegulonDB) with high-throughput expression data (M3D) to validate, contradict, or discover gene regulatory relationships in *E. coli*.

---

## Overview

The transcriptional regulatory network of *E. coli* K-12 is the most well-characterized biological network, yet knowledge is fragmented between:

- **Literature Prior** (RegulonDB): Curated assertions from decades of molecular biology
- **Data Landscape** (M3D): High-throughput expression measurements across 1000+ conditions

This system bridges these worlds using a **LangGraph cyclic workflow** with **LLM-powered reasoning** (ALCF/Argonne) for nuanced biological interpretation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LangGraph Workflow                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌──────────┐    ┌─────────────┐    ┌─────────────────┐            │
│   │  Loader  │───▶│Batch Manager│───▶│Research Agent   │            │
│   │(ingest)  │    │  (queue)    │    │ (🤖 RAG/Vector) │            │
│   └──────────┘    └─────────────┘    └────────┬────────┘            │
│                          ▲                     │                     │
│                          │                     ▼                     │
│                          │            ┌────────────────┐             │
│                          │            │Analysis Agent  │             │
│                          │            │   (CLR/MI)     │             │
│                          │            └────────┬───────┘             │
│                          │                     │                     │
│                          │                     ▼                     │
│                          │            ┌────────────────┐             │
│                          └────────────│  Reviewer      │             │
│                                       │  (🤖 LLM)      │             │
│                                       └────────────────┘             │
│                                               │                      │
│                                               ▼                      │
│                                            [END]                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node | LLM | Description |
|------|-----|-------------|
| **Loader** | ❌ | Ingests RegulonDB network, gene mappings, M3D expression matrix & metadata |
| **Batch Manager** | ❌ | Manages TF processing queue, prevents memory overflow |
| **Research Agent** | ✅ | Retrieves literature context from vector store using RAG (BAAI/bge-small-en-v1.5 embeddings), performs condition matching between dataset and literature |
| **Analysis Agent** | ❌ | Computes Mutual Information & CLR z-scores (deterministic statistics) |
| **Reviewer** | ✅ | Integrates RegulonDB, Research Agent context, and statistical evidence to classify edges into 4 categories using ALCF LLMs |

```python
# From Batch Manager
batch__current_tfs: List[str]  # TFs to process (e.g., ["b1334"])

# From Research Agent
research__literature_edges: Dict[str, Dict[str, Dict]]
# Structure: {tf_bnumber: {target_bnumber: {exists, effect, evidence_strength, ...}}}

```
DREAMingAgent/
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── src/
│   ├── workflow.py             # LangGraph StateGraph orchestration
│   ├── state.py                # AgentState schema & data classes
│   ├── config.py               # System configuration (LLM toggle, thresholds)
│   ├── nodes/
│   │   ├── loader.py           # Data ingestion (RegulonDB + M3D)
│   │   ├── batch_manager.py    # TF queue management
│   │   ├── context_agent.py    # AI-powered sample filtering
│   │   ├── analysis_agent.py   # CLR/MI statistical engine
│   │   └── reconciler.py       # AI-powered reconciliation logic
│   ├── llm/
│   │   ├── client.py           # Google Gemini API wrapper
│   │   └── prompts.py          # System prompts for AI reasoning
│   └── utils/
│       ├── parsers.py          # File parsers & synthetic data generator
│       └── statistics.py       # MI, CLR, correlation functions
├── examples/
│   └── demo_analysis_agent.py  # Standalone CLR/MI demonstration
└── tests/
    └── test_analysis_agent.py  # Unit tests
```

# From Analysis Agent
analysis__statistical_results: Dict[str, Dict[str, Dict]]
# Structure: {tf_bnumber: {target_bnumber: {clr_zscore, mi, correlation, pvalue}}}

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DREAMingAgent.git
cd DREAMingAgent

# Create virtual environment
python -m venv AgentVenv
source AgentVenv/bin/activate  # On Windows: AgentVenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### For LLM Features

The system uses ALCF (Argonne Leadership Computing Facility) for LLM-powered reasoning. Configure access to the ALCF API according to your institution's setup.

---

## Usage

### Quick Start (Synthetic Data)

```bash
# Without LLM (rule-based reasoning)
python main.py --synthetic --no-llm --output output/

# With LLM (AI-powered reasoning via ALCF)
python main.py --synthetic --output output/
```

### Test Coverage

```bash
python main.py \
  --network data/network_tf_gene.txt \
  --genes data/gene_product.txt \
  --expression data/E_coli_v4_Build_6_exps.tab \
  --metadata data/E_coli_v4_Build_6_meta.tab \
  --output results/
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--synthetic`, `-s` | Run with generated test data |
| `--no-llm` | Disable LLM reasoning (use rule-based fallbacks) |
| `--llm-model MODEL` | LLM model to use (default: `argonne-llama3`) |
| `--output`, `-o` | Output directory (default: `output/`) |
| `--verbose`, `-v` | Enable debug logging |
| `--max-iterations` | Max processing iterations (default: 100) |

---

## Core Algorithm: CLR/MI

The Analysis Agent implements the **Context Likelihood of Relatedness (CLR)** algorithm:

1. **Mutual Information (MI)**: Captures non-linear dependencies between TF and target gene expression
   
   \[I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}\]

2. **CLR Z-score Normalization**: Distinguishes direct from indirect regulation
   
   \[z = \max\left(0, \frac{\text{MI} - \mu}{\sigma}\right)\]

3. **Significance Thresholds**:
   - **High**: z ≥ 4.0
   - **Moderate**: z ≥ 2.0
   - **Low**: z < 1.0

---

## Reconciliation Logic

The system classifies each TF→Gene edge into four categories:

| Status | Literature | Data | Interpretation |
|--------|------------|------|----------------|
| **Validated** | Strong | High z-score | Confirmed active regulation |
| **Condition-Silent** | Strong | Low z-score | Real binding, but TF inactive in sampled conditions |
| **Probable False Positive** | Weak | Low z-score | Candidate for database pruning |
| **Novel Hypothesis** | None | High z-score | New discovery for experimental validation |

### LLM-Enhanced Reasoning

When LLM is enabled, the Reconciler provides nuanced biological interpretation:

```json
{
  "status": "ConditionSilent",
  "confidence": 0.85,
  "reasoning": "FNR→narG has strong physical evidence (DNA footprinting), but 
               the M3D compendium is dominated by aerobic conditions where FNR 
               is inactive. The low z-score reflects TF inactivity, not absence 
               of regulation.",
  "recommendation": "Re-analyze using only anaerobic samples."
}
```

---

## Output Files

### `reconciled_network.tsv`

| Column | Description |
|--------|-------------|
| Source_TF | Transcription factor ID |
| Target_Gene | Target gene ID |
| RegulonDB_Evidence | Literature evidence level (Strong/Weak/Unknown) |
| RegulonDB_Effect | Regulation type (+/-/+-/?) |
| M3D_Z_Score | CLR-corrected z-score |
| M3D_MI | Raw mutual information |
| Reconciliation_Status | Validated/ConditionSilent/ProbableFalsePos/NovelHypothesis |
| Context_Tags | Experimental context used |
| Notes | AI reasoning or rule-based explanation |

### `reconciliation_results.json`

```json
{
  "reconciliation_log": [...],
  "novel_hypotheses": [...],
  "false_positive_candidates": [...],
  "summary": {
    "total_edges": 113,
    "novel_count": 1,
    "false_positive_count": 28
  }
}
```

---

## Testing

```bash
# Run all unit tests
pytest -v

# Run specific test class
pytest tests/test_analysis_agent.py::TestCLRCalculation -v

# Demo the Analysis Agent in isolation
python examples/demo_analysis_agent.py
```

---

## Data Sources

### RegulonDB (Literature Prior)
- **Version**: v13.5+ recommended
- **Files**: `network_tf_gene.txt`, `gene_product.txt`
- **URL**: https://regulondb.ccg.unam.mx/

### M3D (Data Landscape)
- **Version**: E. coli Build 6 (v4)
- **Files**: `E_coli_v4_Build_6_exps.tab`, `E_coli_v4_Build_6_meta.tab`
- **URL**: http://m3d.mssm.edu/

---

## Configuration

Environment variables:

| Variable | Description |
|----------|-------------|
| `DREAMING_USE_LLM` | Enable/disable LLM (`true`/`false`) |
| `DREAMING_LLM_MODEL` | Model name (default: `argonne-llama3`) |

---

## Extending the System

### Custom Review Thresholds

Edit CLR z-score thresholds in [src/nodes/reviewer_agent.py](src/nodes/reviewer_agent.py):

```python
HIGH_DATA = 4.0      # Strong data support threshold
MODERATE_DATA = 2.0  # Moderate data support threshold
LOW_DATA = 1.0       # Low data support threshold
```

### Adding Literature to Vector Store

The Research Agent uses a vector store for literature retrieval. To add new literature:

1. Add documents to the vector store via [src/utils/vector_store.py](src/utils/vector_store.py)
2. Documents are embedded using BAAI/bge-small-en-v1.5
3. Include metadata with experimental conditions for better context matching

---

## License

MIT License

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{dreaming_agent,
  title = {DREAMing Agent: Agentic Reconciliation of Biological Literature and High-Throughput Data},
  author = {Chandra, Marcus},
  year = {2024},
  url = {https://github.com/yourusername/DREAMingAgent}
}
```

---

## Contact

Marcus Chandra - DREAMing Agent Team
