# DREAMingAgent Evaluation Framework Implementation Plan

## Overview

Implement a comprehensive 3-metric evaluation framework to assess the DREAMingAgent gene regulatory network reconciliation system. The framework will be located in a new `evaluation/` directory and produce JSON results, matplotlib plots, and markdown reports.

## Requirements Summary

- **Scope**: 3 metrics (A, B, D) - *Metric C (Consistency) removed*.
- **Location**: New `evaluation/` directory (separate from `tests/`)
- **Judge Model**: ALCF system (flexible, may use GPT-OSS instead of GPT-4o)
- **Output**: JSON + Matplotlib plots + Markdown report

## Directory Structure

```
evaluation/
├── __init__.py
├── README.md
├── config.py                    # EvaluationConfig dataclass
├── runner.py                    # CLI entry point
├── metrics/
│   ├── __init__.py
│   ├── base_metric.py           # Abstract base class
│   ├── metric_a_sabotage.py     # False positive injection test
│   ├── metric_b_synthetic.py    # Ground truth recovery
│   └── metric_d_llm_judge.py    # Explanation quality (Real Data)
├── utils/
│   ├── __init__.py
│   ├── graph_manipulation.py    # NetworkX edge inject/delete
│   ├── metrics_calculator.py    # Precision/recall/F1
│   ├── plot_generator.py        # Matplotlib utilities
│   └── report_generator.py      # Markdown generation
├── judges/
│   ├── __init__.py
│   ├── alcf_judge.py            # ALCF LLM judge
│   └── prompts.py               # Judge prompt templates
├── data/                        # Generated test data
└── outputs/                     # Results, plots, reports
```

## Implementation Strategy

### Phase 1: Foundation (Core Infrastructure)

**Files to create:**
1. `evaluation/__init__.py` - Package init
2. `evaluation/config.py` - Configuration management
3. `evaluation/metrics/base_metric.py` - Abstract base class with standard interface:
   - `prepare_data()` → Dict
   - `run_evaluation(data)` → Dict
   - `compute_scores(results)` → Dict[str, float]
   - `generate_plots(results, output_dir)` → List[Path]
   - `execute(output_dir)` → Dict (orchestrates all steps)

4. `evaluation/utils/metrics_calculator.py` - Statistical utilities:
   - Extract precision/recall/F1 pattern from `examples/demo_analysis_agent.py`
   - `compute_classification_metrics(tp, fp, tn, fn)` → Dict
   - `compute_detection_rate(detected, ground_truth)` → float

5. `evaluation/utils/graph_manipulation.py` - NetworkX utilities:
   - `inject_false_edges(graph, n_edges, evidence_level, effect)` → (graph, injected_list)
   - `delete_true_edges(graph, ground_truth, n_edges)` → (graph, deleted_list)
   - `save_graph_to_regulondb_format(graph, output_path)` - TSV export

6. `evaluation/utils/plot_generator.py` - Matplotlib functions:
   - `plot_confusion_matrix(y_true, y_pred, labels, output_path)`
   - `plot_bar_chart(categories, values, ylabel, title, output_path)`
   - `plot_histogram(data, bins, xlabel, title, output_path)`
   - Additional: scatter, pie chart, violin plots

### Phase 2: Simple Metrics (No Graph Modification)

**Metric B: Synthetic Ground Truth Recovery**

File: `evaluation/metrics/metric_b_synthetic.py`

**Strategy:**
1. **Data Preparation**:
   - Call `create_synthetic_test_data()` from `src/utils/parsers.py`
   - **Ground Truth**: Parse the generated `NetworkRegulatorGene.tsv`. All edges in this file are bona fide regulatory interactions in the synthetic universe.
   - Load expression matrix to compute actual correlation strengths (optional validation step).

2. **Evaluation**:
   - Run `src/workflow.py::run_reconciliation()` with synthetic data
   - Extract `reconciliation_log` from final state

3. **Scoring**:
   - True Positive Rate: % of ground truth edges marked as "Validated"
   - False Negative Rate: % of ground truth edges marked as "ConditionSilent" or missed
   - Correlation-weighted metrics

4. **Plots**:
   - Scatter: correlation strength vs detection
   - Pie chart: status distribution
   - Recovery rate bar chart

### Phase 3: Complex Metrics

**Metric A: Sabotage Test**

File: `evaluation/metrics/metric_a_sabotage.py`

**Strategy:**
11. **Data Preparation**:
   - Generate synthetic data with `create_synthetic_test_data()`
   - Parse network to get baseline graph
   - Track all baseline edges as ground truth
   - **Inject false positives** (default: 20 edges):
     - Random (TF, target) pairs not in graph
     - Weak evidence level
     - Mark with `is_injected=True` metadata
   - **Delete true edges** (default: 15 edges):
     - Random sample from ground truth
     - Store deleted edge list
   - Save modified graph to new `NetworkRegulatorGene.tsv`

2. **Evaluation**:
   - Run workflow on sabotaged network
   - Extract `false_positive_candidates` and `novel_hypotheses`

3. **Scoring**:
   - FP Detection Rate: `len(injected ∩ flagged_as_fp) / len(injected)`
   - Edge Recovery Rate: `len(deleted ∩ novel_hypotheses) / len(deleted)`
   - Precision/Recall/F1 for FP detection

4. **Plots**:
   - Confusion matrix
   - Detection rate bar chart
   - Venn diagram (overlap analysis)

**Metric D: LLM-as-a-Judge (Real Data)**

File: `evaluation/metrics/metric_d_llm_judge.py`

**Strategy:**
1. **Judge Implementation** (`evaluation/judges/alcf_judge.py`):
   - Use `src/llm_config.py::create_argonne_llm()` for ALCF connection
   - Pydantic model `JudgeScores`:
     - `biological_accuracy: int` (1-5)
     - `statistical_reasoning: int` (1-5)
     - `clarity: int` (1-5)
     - `overall_quality: int` (1-5)
     - `reasoning: str` (100-500 chars)
   - Use `JsonOutputParser` for structured output

2. **Data Preparation**:
   - Use **Real Data** (`data/NetworkRegulatorGene.tsv`).
   - Run workflow on a specific subset of TFs (e.g. FNR, ArcA) to get biologically meaningful explanations.
   - Sample N=30 explanations from the results.

3. **Evaluation**:
   - For each edge explanation:
     - Extract from `reconciliation_log[i]["notes"]`
     - Send to judge with context (edge_id, status, lit_evidence, z_score)
     - Parse judge scores

4. **Scoring**:
   - Mean scores per dimension
   - High quality rate (≥4 threshold)
   - Per-status breakdown

5. **Plots**:
   - Score distribution histogram per dimension
   - Violin plot (scores by reconciliation status)
   - Scatter: overall score vs z-score

**Judge Prompt Template** (`evaluation/judges/prompts.py`):
```
You are an expert biological judge evaluating gene regulatory network explanations.

Evaluate on 1-5 Likert scale:
1. Biological Accuracy
2. Statistical Reasoning
3. Clarity
4. Overall Quality

Edge: {edge_id}
Status: {status}
Literature: {literature_evidence}
CLR z-score: {statistical_score}

Explanation:
{explanation}

Provide valid JSON only.
```

### Phase 4: Integration & Reporting

**File: `evaluation/runner.py`** - CLI entry point

```bash
# Run all metrics
python -m evaluation.runner --all

# Run specific metrics
python -m evaluation.runner --metric A B

# Custom output directory
python -m evaluation.runner --all --output-dir results/
```

**File: `evaluation/utils/report_generator.py`**

Functions:
- `generate_metric_report(metric_name, scores, plots, output_path)` - Single metric
- `generate_combined_report(metric_results, output_path)` - Summary across all 3

**Report Structure:**
```markdown
# DREAMingAgent Evaluation Summary

## Metric A: Sabotage Test
- FP Detection Rate: 0.750
- Edge Recovery Rate: 0.600
- F1 Score: 0.800

![Confusion Matrix](plots/metric_a_confusion_matrix.png)

## Metric B: Synthetic Ground Truth
- True Positive Rate: 0.850
- False Negative Rate: 0.150

## Metric D: LLM Judge
- Mean Overall Quality: 4.1/5
- High Quality Rate: 73%
```

## Critical Files Reference

**No modifications needed to existing workflow** - evaluation is read-only:

1. **`src/workflow.py`**
   - Use: `run_reconciliation()` - programmatic workflow invocation
   - Returns: `final_state` with `reconciliation_log`, `novel_hypotheses`, `false_positive_candidates`

2. **`src/utils/parsers.py`**
   - Use: `create_synthetic_test_data()` - generates RegulonDB-compliant test data
   - Returns: Dict with file paths (network, gene_product, expression, metadata)

3. **`src/state.py`**
   - Use: Understand `AgentState` schema
   - Key fields: `reconciliation_log`, `novel_hypotheses`, `false_positive_candidates`

4. **`src/llm_config.py`**
   - Use: `create_argonne_llm()` - for Metric D judge
   - Pattern: Same ALCF authentication/configuration

5. **`examples/demo_analysis_agent.py`**
   - Use: Precision/recall/F1 calculation pattern
   - Extract to `metrics_calculator.py`

6. **`src/nodes/reconciler.py`**
   - Reference: Threshold constants
   - `HIGH_DATA_SUPPORT = 4.0`, `MODERATE_DATA_SUPPORT = 2.0`

## Implementation Order

### Week 1: Foundation
1. Create directory structure
2. Implement `base_metric.py` abstract class
3. Implement `metrics_calculator.py` (extract from demo_analysis_agent.py)
4. Implement `graph_manipulation.py` (inject/delete functions)
5. Implement `plot_generator.py` (basic plots)

### Week 2: Simple Metrics
6. Implement Metric B (synthetic ground truth)
   - Test with existing synthetic data generator
   - Validate precision/recall calculations

### Week 3: Complex Metrics
7. Implement Metric A (sabotage test)
   - Test graph modification utilities
   - Validate injection/deletion tracking
8. Implement Metric D (LLM judge)
   - Implement ALCF judge with Pydantic models
   - Test with small sample of real data
   - Handle API rate limiting

### Week 4: Integration
9. Implement `runner.py` CLI
10. Implement `report_generator.py`
11. End-to-end testing
12. Documentation (`evaluation/README.md`)

## Testing Strategy

**Unit Tests** (`evaluation/tests/`):
- `test_graph_manipulation.py` - Validate inject/delete functions
- `test_metrics_calculator.py` - Test precision/recall edge cases
- `test_base_metric.py` - Test abstract class contract

**Integration Tests**:
- `test_metric_a_sabotage.py` - Full pipeline test
- `test_metric_b_synthetic.py` - Ground truth extraction
- `test_metric_d_judge.py` - Mock LLM judge responses

**Edge Cases**:
- Empty reconciliation log (workflow failure)
- LLM API failures (judge timeouts)
- Zero ground truth edges
- Missing explanation fields

## Configuration

**File: `evaluation/config.py`**

```python
@dataclass
class EvaluationConfig:
    # Paths
    data_dir: Path = Path("evaluation/data")
    output_dir: Path = Path("evaluation/outputs")

    # Metric A
    sabotage_n_false_positives: int = 20
    sabotage_n_deletions: int = 15

    # Metric B
    synthetic_n_genes: int = 100
    synthetic_n_tfs: int = 10

    # Metric D
    judge_model: str = "openai/gpt-oss-20b"
    judge_sample_size: int = 30
    judge_tfs: List[str] = ["FNR", "ArcA", "CRP"] # TFs to test for Metric D

    # General
    random_seed: int = 42
```

## Expected Outputs

### JSON Results
- `evaluation/outputs/results/metric_a_sabotage.json`
- `evaluation/outputs/results/metric_b_synthetic.json`
- `evaluation/outputs/results/metric_d_llm_judge.json`

### Plots
- `evaluation/outputs/plots/metric_a_confusion_matrix.png`
- `evaluation/outputs/plots/metric_a_detection_rates.png`
- `evaluation/outputs/plots/metric_b_recovery_rates.png`
- `evaluation/outputs/plots/metric_d_score_distribution.png`

### Reports
- `evaluation/outputs/reports/evaluation_summary.md` - Combined summary
- `evaluation/outputs/reports/metric_a.md` - Sabotage test details
- `evaluation/outputs/reports/metric_b.md` - Synthetic recovery details
- `evaluation/outputs/reports/metric_d.md` - Judge scores breakdown

## Success Criteria

- ✅ All 3 metrics execute without errors
- ✅ JSON outputs follow consistent schema
- ✅ Plots generated for each metric
- ✅ Markdown reports are human-readable
- ✅ No modifications to existing workflow code
- ✅ Execution time <10 minutes on synthetic data
- ✅ Reproducible results with fixed random seeds

## Estimated Effort

- **Total**: ~3-4 weeks (1 developer)
- **Lines of Code**: ~1800-2200 (excluding tests)
- **External Dependencies**: None beyond existing requirements.txt
- **Risk**: Metric D depends on ALCF API availability (fallback: skip or use mock)