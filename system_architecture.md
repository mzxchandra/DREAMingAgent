# DREAMing Agent: System Architecture & Data Flow

This document demystifies how the three primary data sources (RegulonDB, M3D, LitSense) integrate to drive the agent's decision-making process.

## High-Level Concept
The system acts as a **Scientist Reviewing a Hypothesis**.
1.  **RegulonDB** provides the **Hypothesis / Prior Knowledge** ("We believe protein A regulates gene B").
2.  **M3D** provides the **Experimental Data** ("Do we see A and B correlated in actual experiments?").
3.  **LitSense** provides the **Contextual nuance** ("Does this interaction only happen in *acidic* conditions?").

## Data Flow Diagram

```mermaid
graph TD
    %% Styling
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef agent fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef store fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    
    subgraph "1. Input Data"
        RDB[("RegulonDB<br>(The Map)")]:::data
        M3D[("M3D Expression<br>(The Traffic)")]:::data
        LIT[("LitSense<br>(The Guidebook)")]:::data
    end

    subgraph "2. Knowledge Preparation"
        Ingest[("Ingest Script")]
        VDB[("ChromaDB<br>Vector Store")]:::store
        Loader[("Loader Node")]:::agent
        
        LIT -->|Scrape & Embed| Ingest -->|Store Embeddings| VDB
        RDB -->|Load Graph Structure| Loader
        M3D -->|Load Matrices| Loader
    end

    subgraph "3. The Agentic Loop"
        subgraph "Research Agent"
            RA[("Context Filtering")]:::agent
            Loader -->|TF Batch| RA
            M3D -.->|Filter Samples| RA
        end
        
        subgraph "Analysis Agent"
            AA[("Statistical Engine")]:::agent
            RA -->|Filtered Context| AA
            AA -->|Calculate CLR & MI| Stats[("Z-Scores & Stats")]
        end
        
        subgraph "Reviewer Agent (The Judge)"
            REV[("LLM Reviewer")]:::agent
            
            Stats -->|Strength of Signal| REV
            RDB -.->|Evidence Strength| REV
            VDB -.->|Retrieves Context| REV
            
            note["Context Lookup:<br>'Why might this edge be missing?'"]
            VDB -.- note
        end
    end

    subgraph "4. Output"
        Decision{("Decision")}
        
        REV --> Decision
        Decision -->|Validated| VAL[("Validated Edge")]
        Decision -->|Novel| NOV[("Novel Hypothesis")]
        Decision -->|Silent| SIL[("Condition Silent")]
        Decision -->|FalsePos| FP[("Probable Error")]
    end
```

## Component Breakdown

### 1. RegulonDB (The "Prior")
*   **Role:** Defines the baseline "Truth". It tells the system what *should* exist described by decades of manual curation.
*   **Usage:**
    *   **Sabotage Test (Metric A):** We purposely delete true RegulonDB edges to see if the system can "rediscover" them using data.
    *   **Reviewer Agent:** Used to tag an edge as "Has Literature Support". If the data matches RegulonDB, we mark it **Validated**. If the data contradicts RegulonDB (low correlation), it triggers a deeper review (Is it "Condition Silent" or a "False Positive"?).

### 2. M3D (The "Signal")
*   **Role:** Provides unbiased, raw observational data from thousands of E. coli microarrays.
*   **Usage:**
    *   **Analysis Agent:** Calculates the **CLR (Context Likelihood of Relatedness)** Z-score.
        *   **Z > 4.0:** Strong Signal (The genes definitely talk to each other).
        *   **Z < 1.0:** No Signal (They ignore each other).
    *   **Discovery:** If M3D shows a Strong Signal (Z>4) but RegulonDB has *no record* of it, the system proposes a **Novel Hypothesis**.

### 3. LitSense / ChromaDB (The "Context")
*   **Role:** Provides the *reasoning* behind the biology. M3D gives numbers; LitSense gives words.
*   **Usage:**
    *   **Reviewer Agent:** When the Signal (M3D) is weak but the Prior (RegulonDB) is strong, the Agent queries ChromaDB.
        *   *Query:* "What conditions activate TF [Name]?"
        *   *Result:* "TF [Name] is only active under anaerobic starvation."
    *   **Decision:** The Agent sees the M3D data was from *aerobic* users. It concludes: "The edge is real, but silent in this dataset." -> **Condition Silent** (instead of incorrectly calling it a False Positive).

## Summary of Logic
| RegulonDB (Lit) | M3D (Data) | LitSense Context | System Decision |
|---|---|---|---|
| Strong | Strong | (Consistent) | **Validated** (Gold Standard) |
| None | Strong | (Silent) | **Novel Hypothesis** (Discovery!) |
| Strong | Weak | "Requires Acidic pH" | **Condition Silent** (Correctly preserved) |
| Weak | Weak | "Experimental Artifact" | **Probable False Positive** (Correction) |
