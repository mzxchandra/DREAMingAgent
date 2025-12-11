# DREAMing Agent: System Architecture & Data Flow

This document details the agentic architecture for reconciling biological knowledge (RegulonDB) with high-throughput data (M3D).

## High-Level Concept
The system acts as a **Scientist Reviewing a Hypothesis**.
1.  **RegulonDB** provides the **Hypothesis / Prior Knowledge** ("We believe protein A regulates gene B").
2.  **M3D** provides the **Experimental Data** ("Do we see A and B correlated in actual experiments?").
3.  **Research Agent** provides the **Context** ("What conditions trigger this interaction?").

## Data Flow Diagram

```mermaid
graph TD
    %% Styling
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef store fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef subflow fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5;
    
    subgraph "1. Data Preparation"
        RDB[("RegulonDB")]:::data
        M3D[("M3D Expression")]:::data
        Loader[("Loader Node")]:::agent
        Batch[("Batch Manager")]:::agent
        
        RDB --> Loader
        M3D --> Loader
        Loader --> Batch
    end

    subgraph "2. Research Agent (The Contextualizer)"
        RA[("Research Agent Node")]:::agent
        VS[("Vector Store")]:::store
        
        subgraph "Internal Research Workflow"
            Query[("Query Formulation")]:::subflow --> Ret[("Vector Retrieval")]:::subflow
            Ret --> Ext[("Context Extraction")]:::subflow
            Ext --> Match[("Condition Matching")]:::subflow
            Match --> Expl[("Explanation")]:::subflow
            Expl --> Format[("Output Formatting")]:::subflow
        end
        
        Batch --> RA
        VS -.->|Literature RAG| Ret
        
        RA -->|Filtered Samples| AA_Input(("Active\nSamples"))
    end

    subgraph "3. Analysis Agent (The Statistician)"
        AA[("Analysis Agent")]:::agent
        
        AA_Input --> AA
        M3D -.-> AA
        AA -->|CLR/MI Stats| Stats[("Statistical Evidence")]
    end
    
    subgraph "4. Reviewer Agent (The Judge)"
        Rev[("Reviewer Agent")]:::agent
        
        Stats --> Rev
        RDB -.->|Prior Evidence| Rev
        VS -.->|Verify Context| Rev
        
        Rev --> Decision{("Decision")}
    end

    Decision -->|Validated| VAL[("Validated Edge")]
    Decision -->|Novel| NOV[("Novel Hypothesis")]
    Decision -->|Silent| SIL[("Condition Silent")]
    Decision -->|FalsePos| FP[("Probable Error")]

    Decision -->|Next Batch| Batch
```

## Component Breakdown

### 1. Research Agent (Dual-Mode)
This advanced agent replaces the simple context filtering of previous versions. It performs two key functions:
1.  **Context Filtering:** Uses biological knowledge (from literature semantics) to intelligently select relevant M3D samples (e.g., filtering for "Anaerobic" conditions for the TF FNR).
2.  **Literature RAG Loop:** Executes an internal 6-step LangGraph workflow for each gene pair:
    *   *Query:* Generates optimized search queries.
    *   *Retrieval:* Fetches docs from ChromaDB.
    *   *Context:* Extracts required conditions (e.g., "Requires pH < 5").
    *   *Matching:* Compares required conditions vs. dataset metadata.
    *   *Explanation:* Generates a scientific rationale.

### 2. Analysis Agent (The "Signal")
*   **Role:** Calculates the **CLR (Context Likelihood of Relatedness)** Z-score using *only* the samples selected by the Research Agent. mechanism.
*   **Logic:**
    *   **Z > 4.0:** Strong Signal.
    *   **Z < 1.0:** No Signal.

### 3. Reviewer Agent (The "Judge")
*   **Role:** Synthesizes all evidence (Prior + Data + Context) to make a final classification.
*   **Process:**
    *   Checks **Data Strength** (from Analysis Agent).
    *   Checks **Literature Support** (from RegulonDB).
    *   Verifies **Context** (queries Vector Store to explain discrepancies).
*   **Output Categories:**
    *   **Validated:** Strong Literature + Strong Data.
    *   **Condition Silent:** Strong Literature + Weak Data (explained by context mismatch).
    *   **Probable False Positive:** Weak Literature + Weak Data.
    *   **Novel Hypothesis:** No Literature + Strong Data.
