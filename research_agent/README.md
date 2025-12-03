# Research Agent

The Research Agent is a LangGraph-based system that analyzes gene regulatory relationships by combining literature evidence with experimental dataset conditions.

## Architecture

The agent uses a 6-node LangGraph workflow:

1. **Query Formulation** - Creates optimized search query for gene pair
2. **Vector Retrieval** - Retrieves relevant documents from ChromaDB
3. **Context Extraction** - Extracts regulatory relationship information
4. **Condition Matching** - Compares literature vs dataset conditions
5. **Explanation Generation** - Generates natural language explanation
6. **Output Formatting** - Formats final structured output

## Components

### Core Modules

- `config.py` - ALCF API configuration and parameters
- `auth.py` - Globus authentication wrapper
- `models.py` - Pydantic input/output models
- `alcf_client.py` - Unified ALCF client (handles both LLM and embeddings)
- `vector_store.py` - ChromaDB vector store operations

### Graph Components

- `graph/state.py` - LangGraph state definition
- `graph/prompts.py` - LLM prompts for each node
- `graph/nodes.py` - Node function implementations
- `graph/graph.py` - Graph construction and agent interface

## Usage

### Using the Unified ALCF Client

The unified `ALCFClient` handles LLM chat completions (via ALCF) and embeddings (via multiple providers):

```python
from research_agent.alcf_client import get_alcf_client

# Single client for both operations
client = get_alcf_client()

# Chat completion (always uses ALCF)
response = client.chat(
    messages=[{"role": "user", "content": "What is gene regulation?"}],
    temperature=0.7
)

# Embeddings (server-side via HuggingFace Inference API by default)
embedding = client.embed("lexA represses recA")

# Batch embeddings
embeddings = client.embed_batch(["gene A", "gene B", "gene C"])
```

**Embedding Provider Options:**

1. **HuggingFace Inference API** (default, server-side)
   - Model: `BAAI/bge-small-en-v1.5`
   - Requires: HuggingFace read token
   - Benefits: No local compute, always available, free tier
   ```bash
   export RESEARCH_AGENT_EMBEDDING_PROVIDER="huggingface_api"
   export RESEARCH_AGENT_HUGGINGFACE_API_TOKEN="hf_xxxxxxxxxxxxx"
   ```

2. **HuggingFace Local** (local compute)
   - Model: `sentence-transformers/all-mpnet-base-v2`
   - No API key needed, runs locally
   ```bash
   export RESEARCH_AGENT_EMBEDDING_PROVIDER="huggingface"
   ```

3. **ALCF** (often unavailable)
   - Model: `mistralai/Mistral-7B-Instruct-v0.3-embed`
   - Requires ALCF authentication
   ```bash
   export RESEARCH_AGENT_EMBEDDING_PROVIDER="alcf"
   ```

### Basic Example

```python
from research_agent.models import ResearchAgentInput, DatasetMetadata, StatisticalSummary
from research_agent.graph.graph import get_research_agent
from research_agent.vector_store import get_vector_store

# 1. Setup vector store (one-time)
vector_store = get_vector_store()
vector_store.load_from_json("research_agent/data/sample_literature.json")

# 2. Create input
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
        p_value=0.001
    )
)

# 3. Run analysis
agent = get_research_agent()
output = agent.analyze(agent_input)

# 4. Access results
print(f"Regulatory relationship: {output.regulatory_relationship}")
print(f"Condition match: {output.condition_match}")
print(f"Confidence: {output.confidence:.3f}")
print(f"Explanation: {output.explanation}")
```

### Running Examples

```bash
# Activate environment
source globus_env/bin/activate

# Run example script
python example_usage.py
```

## Configuration

Configuration can be set via environment variables:

```bash
# LLM configuration (ALCF)
export RESEARCH_AGENT_BASE_URL="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
export RESEARCH_AGENT_LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Embedding configuration (HuggingFace Inference API by default - RECOMMENDED)
export RESEARCH_AGENT_EMBEDDING_PROVIDER="huggingface_api"
export RESEARCH_AGENT_HUGGINGFACE_API_TOKEN="hf_xxxxxxxxxxxxx"  # Get from https://huggingface.co/settings/tokens
export RESEARCH_AGENT_HUGGINGFACE_API_MODEL="BAAI/bge-small-en-v1.5"

# Alternative: Use local HuggingFace embeddings (requires more compute)
# export RESEARCH_AGENT_EMBEDDING_PROVIDER="huggingface"
# export RESEARCH_AGENT_HUGGINGFACE_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"

# Alternative: Use ALCF embeddings (often unavailable)
# export RESEARCH_AGENT_EMBEDDING_PROVIDER="alcf"
# export RESEARCH_AGENT_EMBEDDING_MODEL="mistralai/Mistral-7B-Instruct-v0.3-embed"

# Or use Metis cluster for LLM
# export RESEARCH_AGENT_BASE_URL="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
# export RESEARCH_AGENT_LLM_MODEL="gpt-oss-120b-131072"

# Other parameters
export RESEARCH_AGENT_LLM_TEMPERATURE="0.7"
export RESEARCH_AGENT_RETRIEVAL_TOP_K="5"
```

Or create a `.env` file in the project root.

**Available LLM models:**
- **Sophia cluster**: Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct
- **Metis cluster**: gpt-oss-120b-131072

**Available embedding providers:**
- **HuggingFace Inference API** (recommended): Server-side, free tier, always available
  - Models: BAAI/bge-small-en-v1.5, sentence-transformers/all-MiniLM-L6-v2, intfloat/e5-base-v2
- **HuggingFace Local**: Runs on your machine, no API needed
  - Models: sentence-transformers/all-mpnet-base-v2, BAAI/bge-small-en-v1.5
- **ALCF** (often unavailable): mistralai/Mistral-7B-Instruct-v0.3-embed

## Authentication

Before using the agent, authenticate with ALCF:

```bash
python inference_auth_token.py authenticate
```

This will open a browser for Globus login. Tokens are cached in `~/.globus/`.

## Input/Output Models

### ResearchAgentInput

```python
ResearchAgentInput(
    gene_a: str,                          # Regulator gene
    gene_b: str,                          # Target gene
    dataset_metadata: DatasetMetadata,    # Experimental conditions
    statistical_summary: Optional[StatisticalSummary]  # Optional stats
)
```

### ResearchAgentOutput

```python
ResearchAgentOutput(
    gene_pair: str,                       # "lexA -> recA"
    context_found: bool,                  # Literature exists?
    regulatory_relationship: str,         # activation/repression/no_interaction/unknown
    required_conditions: List[str],       # Conditions from literature
    dataset_conditions: List[str],        # Actual dataset conditions
    condition_match: str,                 # match/mismatch/partial/unknown
    explanation: str,                     # Natural language explanation
    confidence: float,                    # 0.0-1.0 confidence score
    supporting_documents: List[SupportingDocument],
    reasoning_trace: List[str]            # Step-by-step reasoning
)
```

## Vector Store Management

### Adding Documents

```python
from research_agent.models import LiteratureDocument
from research_agent.vector_store import get_vector_store

doc = LiteratureDocument(
    doc_id="my_doc_001",
    gene_a="lexA",
    gene_b="recA",
    interaction_type="repression",
    conditions=["normal_growth"],
    evidence="ChIP-seq",
    source="RegulonDB",
    text="LexA represses recA under normal conditions..."
)

vector_store = get_vector_store()
vector_store.add_document(doc)
```

### Querying

```python
# Query by gene pair
results = vector_store.query_by_gene_pair("lexA", "recA", n_results=5)

# General query
results = vector_store.query("DNA damage response", n_results=10)
```

### Resetting

```python
vector_store.reset()  # Delete all documents
```

## Development

### Running Tests

```bash
# TODO: Add tests
pytest research_agent/tests/
```

### Adding New Literature Sources

1. Create JSON file with `LiteratureDocument` format
2. Load into vector store: `vector_store.load_from_json("path/to/file.json")`

### Customizing Prompts

Edit `graph/prompts.py` to modify LLM prompts for each node.

## Troubleshooting

### Authentication Errors

```
Failed to get ALCF token
```

Solution: Run `python inference_auth_token.py authenticate`

### Import Errors

```
ImportError: chromadb is not installed
```

Solution: `pip install chromadb>=0.4.22`

### Empty Results

If vector store returns no results, check:
- Vector store has documents: `vector_store.count()`
- Gene names match exactly (case-sensitive)
- Embedding model is working

## Performance

- Query formulation: ~1-2 seconds
- Vector retrieval: ~0.5 seconds
- Context extraction: ~2-3 seconds (LLM call)
- Condition matching: ~2-3 seconds (LLM call)
- Explanation generation: ~2-3 seconds (LLM call)

**Total per gene pair: ~10-15 seconds**

## Future Enhancements

- Batch processing optimization
- Caching of LLM responses
- Alternative embedding models
- Confidence calibration
- Multi-hop reasoning for complex interactions
