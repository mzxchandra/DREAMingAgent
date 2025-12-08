# Research Agent Implementation Plan

## Overview
The Research Agent ("The Librarian") performs context retrieval to find biological conditions under which gene regulations occur, using RAG (Retrieval Augmented Generation) with a vector store of E. coli literature.

## Architecture

### Component Stack
1. **Agent Framework**: LangGraph (state graph orchestration)
2. **Vector Database**: ChromaDB (lightweight, embedded)
3. **Embedding Model**: SFR-Embedding-Mistral or Mistral-7B-Instruct-v0.3-embed (ALCF Sophia cluster)
4. **LLM**: Meta-Llama-3.1-70B-Instruct or Llama-3.3-70B-Instruct (ALCF Sophia cluster)
5. **API Client**: OpenAI Python SDK (ALCF uses OpenAI-compatible endpoints)

### Dependencies
```python
# Core agent framework
langgraph>=0.0.20
langchain>=0.1.0
langchain-community>=0.0.10

# Vector store
chromadb>=0.4.22

# API client
openai>=1.0.0

# Authentication
globus-sdk  # For ALCF token management

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Optional for local testing
python-dotenv>=1.0.0
```

## Expected Inputs

### Primary Input (per query)
```python
{
    "gene_a": str,              # Gene identifier (e.g., "lexA", "recA")
    "gene_b": str,              # Potential regulatory target
    "dataset_metadata": {
        "conditions": List[str], # e.g., ["aerobic", "37C", "rich_media"]
        "treatment": str,        # e.g., "heat_shock", "acid_stress", "none"
        "growth_phase": str,     # e.g., "exponential", "stationary"
        "time_point": str,       # e.g., "0h", "2h", "24h"
    },
    "statistical_summary": {    # Optional: from Analysis Agent
        "correlation": float,    # e.g., 0.05
        "mutual_information": float,
        "p_value": float
    }
}
```

### Vector Store Data (preprocessed)
```python
# Literature documents to be embedded and stored
{
    "doc_id": str,               # Unique identifier
    "gene_a": str,               # Regulator gene
    "gene_b": str,               # Target gene
    "interaction_type": str,     # e.g., "activation", "repression", "binding"
    "conditions": List[str],     # Required/observed conditions
    "evidence": str,             # Evidence type (e.g., "ChIP-seq", "RNA-seq", "literature")
    "source": str,               # "RegulonDB", "PubMed:PMID"
    "text": str,                 # Full text description for embedding
    "metadata": dict            # Additional structured data
}
```

## Expected Outputs

### Primary Output
```python
{
    "gene_pair": str,                    # "gene_a -> gene_b"
    "context_found": bool,               # Whether relevant literature exists
    "regulatory_relationship": str,      # "activation", "repression", "no_interaction", "unknown"
    "required_conditions": List[str],    # Conditions needed for interaction
    "dataset_conditions": List[str],     # Actual dataset conditions
    "condition_match": str,              # "match", "mismatch", "partial", "unknown"
    "explanation": str,                  # Natural language explanation
    "confidence": float,                 # 0.0-1.0 based on retrieval quality
    "supporting_documents": List[dict],  # Retrieved source documents
    "reasoning_trace": List[str]         # Step-by-step reasoning for transparency
}
```

### Example Output
```python
{
    "gene_pair": "lexA -> recA",
    "context_found": True,
    "regulatory_relationship": "repression",
    "required_conditions": ["SOS_response", "DNA_damage", "UV_exposure"],
    "dataset_conditions": ["aerobic", "37C", "no_stress"],
    "condition_match": "mismatch",
    "explanation": "Literature confirms lexA represses recA, but this regulation is specifically activated during SOS response triggered by DNA damage. The dataset metadata indicates normal growth conditions without UV or DNA-damaging agents. Therefore, the lack of correlation in the data is expected and correct.",
    "confidence": 0.92,
    "supporting_documents": [
        {
            "doc_id": "regulondb_lexA_recA_001",
            "source": "RegulonDB",
            "text": "LexA represses recA transcription under normal conditions...",
            "similarity_score": 0.89
        }
    ],
    "reasoning_trace": [
        "Retrieved 5 documents about lexA-recA interaction",
        "Top document (score: 0.89) indicates repression during SOS response",
        "Dataset conditions analyzed: aerobic, 37C, no_stress",
        "Condition mismatch detected: SOS response required but absent",
        "Conclusion: Lack of correlation expected due to condition mismatch"
    ]
}
```

## LangGraph State Design

### State Schema
```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph
import operator

class ResearchAgentState(TypedDict):
    # Input data
    gene_a: str
    gene_b: str
    dataset_metadata: dict
    statistical_summary: dict

    # Intermediate state
    query_embedding: List[float]
    retrieved_documents: List[dict]
    relevant_conditions: List[str]

    # Analysis results
    regulatory_relationship: str
    required_conditions: List[str]
    condition_match: str

    # Output
    explanation: str
    confidence: float
    reasoning_trace: Annotated[List[str], operator.add]  # Accumulated reasoning

    # Error handling
    errors: Annotated[List[str], operator.add]
```

## LangGraph Workflow

### Node Definitions

1. **query_formulation**
   - Takes gene pair and dataset metadata
   - Generates optimized search query for literature retrieval
   - Creates embedding vector for semantic search

2. **vector_retrieval**
   - Queries ChromaDB with embedding
   - Retrieves top-k most relevant documents
   - Filters by gene pair relevance

3. **context_extraction**
   - Uses LLM to extract key information from retrieved documents
   - Identifies: regulatory relationship, required conditions, evidence strength
   - Structures information for comparison

4. **condition_matching**
   - Compares required conditions (from literature) vs dataset conditions
   - Determines match/mismatch/partial/unknown
   - Calculates confidence based on condition overlap

5. **explanation_generation**
   - Uses LLM to synthesize findings into natural language
   - Explains why correlation exists or doesn't exist
   - Generates reasoning trace for transparency

6. **output_formatting**
   - Structures final output dictionary
   - Adds metadata and supporting documents
   - Returns complete response

### Graph Flow
```
START
  ↓
query_formulation
  ↓
vector_retrieval
  ↓
context_extraction
  ↓
condition_matching
  ↓
explanation_generation
  ↓
output_formatting
  ↓
END
```

### Conditional Edges (Error Handling)
- If vector_retrieval returns no results → generate "no_context_found" response
- If context_extraction fails → retry with simplified prompt or return low-confidence result
- If condition_matching is ambiguous → request human review (future enhancement)

## Sample Data Structure

### Sample RegulonDB-style Document (JSON)
```json
{
    "doc_id": "regulon_001",
    "gene_a": "lexA",
    "gene_b": "recA",
    "interaction_type": "repression",
    "conditions": ["SOS_response", "DNA_damage"],
    "evidence": "ChIP-seq, literature",
    "source": "RegulonDB",
    "text": "LexA acts as a transcriptional repressor of recA. Under normal conditions, LexA binds to the SOS box in the recA promoter, preventing transcription. Upon DNA damage, RecA protein is activated and stimulates autocleavage of LexA, leading to derepression of recA and other SOS genes.",
    "metadata": {
        "confidence": "high",
        "year": 2015,
        "reference": "PMID:12345678"
    }
}
```

### Sample Dataset Metadata (JSON)
```json
{
    "dataset_id": "M3D_ecoli_heat_001",
    "conditions": ["aerobic", "heat_shock", "42C"],
    "treatment": "temperature_shift",
    "growth_phase": "exponential",
    "time_point": "30min",
    "media": "LB_broth",
    "strain": "K12_MG1655"
}
```

## Implementation Steps

### Phase 1: Setup and Infrastructure
1. Set up Python project structure
2. Configure ALCF authentication (Globus token)
3. Create OpenAI client wrapper for ALCF endpoints
4. Initialize ChromaDB with appropriate settings
5. Create sample data (5-10 gene pairs with literature)

### Phase 2: RAG System
1. Implement document ingestion pipeline
2. Generate embeddings using ALCF embedding model
3. Store embeddings in ChromaDB with metadata
4. Test retrieval with sample queries

### Phase 3: LangGraph Agent
1. Define state schema
2. Implement each node function
3. Build state graph with edges
4. Add error handling and conditional logic
5. Test with sample inputs

### Phase 4: Integration and Testing
1. Create end-to-end tests with sample data
2. Validate outputs match expected format
3. Test condition matching logic
4. Evaluate explanation quality
5. Measure retrieval accuracy

### Phase 5: Enhancement (Future)
1. Expand to larger dataset
2. Add multi-hop reasoning (gene A → B → C)
3. Implement caching for repeated queries
4. Add batch processing for multiple gene pairs
5. Create evaluation metrics (precision/recall on known interactions)

## File Structure
```
research_agent/
├── __init__.py
├── config.py                    # ALCF endpoints, model names
├── auth.py                      # Globus token management
├── models.py                    # Pydantic models for I/O
├── vector_store.py              # ChromaDB setup and operations
├── embeddings.py                # ALCF embedding client
├── llm_client.py                # ALCF LLM client
├── graph/
│   ├── __init__.py
│   ├── state.py                 # LangGraph state definition
│   ├── nodes.py                 # Node implementations
│   ├── graph.py                 # Graph construction
│   └── prompts.py               # LLM prompts
├── data/
│   ├── sample_literature.json   # Sample RegulonDB-style data
│   └── sample_metadata.json     # Sample dataset metadata
├── tests/
│   ├── test_vector_store.py
│   ├── test_graph.py
│   └── test_integration.py
└── main.py                      # CLI entry point
```

## Testing Strategy

### Unit Tests
- Vector store operations (add, retrieve, update)
- Embedding generation
- Each LangGraph node in isolation
- Condition matching logic

### Integration Tests
- End-to-end agent execution with sample data
- Multiple gene pair queries
- Edge cases (no context found, ambiguous conditions)

### Evaluation Tests
- Known gene pairs with ground truth
- Precision/recall on condition matching
- Explanation quality (LLM-as-judge)

## Key Considerations

### Performance
- ChromaDB embedding cache to avoid re-embedding
- Batch retrieval for multiple queries
- LLM response caching for repeated patterns

### Accuracy
- Top-k retrieval parameter tuning (start with k=5)
- Similarity threshold for document relevance (start with 0.7)
- Confidence scoring based on retrieval quality

### Explainability
- Reasoning trace at each step
- Return supporting documents with similarity scores
- Transparent condition matching logic

### Scalability
- Start with 10-20 sample documents
- Design for 10,000+ documents eventually
- Consider chunking strategy for long documents
