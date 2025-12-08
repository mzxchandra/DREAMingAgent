"""ChromaDB vector store for literature documents."""

import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "chromadb is not installed. Please run: pip install chromadb>=0.4.22"
    )

from src.alcf_config import get_alcf_config
from src.llm.alcf_client import get_alcf_client


# ============================================================================
# Pydantic Models (inline for vector store)
# ============================================================================

class LiteratureDocument(BaseModel):
    """A literature document to be stored in the vector database."""

    doc_id: str = Field(description="Unique document identifier")
    gene_a: str = Field(description="Regulator gene")
    gene_b: str = Field(description="Target gene")
    interaction_type: str = Field(
        description="Interaction type: 'activation', 'repression', 'binding', etc."
    )
    conditions: List[str] = Field(description="Required/observed conditions for interaction")
    evidence: str = Field(description="Evidence type (e.g., 'ChIP-seq', 'RNA-seq', 'literature')")
    source: str = Field(description="Source identifier (e.g., 'RegulonDB', 'PubMed:12345678')")
    text: str = Field(description="Full text description for embedding")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SupportingDocument(BaseModel):
    """A supporting document from literature retrieval."""

    doc_id: str = Field(description="Document identifier")
    source: str = Field(description="Source (e.g., 'RegulonDB', 'PubMed:PMID')")
    text: str = Field(description="Document text")
    similarity_score: float = Field(description="Similarity score from retrieval")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


# ============================================================================
# Vector Store Class
# ============================================================================

class LiteratureVectorStore:
    """Vector store for literature documents using ChromaDB."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        alcf_client=None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of ChromaDB collection (defaults to config)
            persist_directory: Directory for persistence (defaults to config)
            alcf_client: ALCFClient instance (creates if None)
        """
        config = get_alcf_config()
        self.collection_name = collection_name or config.chroma_collection_name
        self.persist_directory = persist_directory or config.chroma_persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Initialize ALCF client for embeddings
        self.alcf_client = alcf_client or get_alcf_client()

    def add_document(self, doc: LiteratureDocument) -> None:
        """
        Add a single literature document to the vector store.

        Args:
            doc: LiteratureDocument to add
        """
        self.add_documents([doc])

    def add_documents(self, docs: List[LiteratureDocument]) -> None:
        """
        Add multiple literature documents to the vector store.

        Args:
            docs: List of LiteratureDocument instances to add
        """
        if not docs:
            return

        # Prepare data for ChromaDB
        doc_ids = [doc.doc_id for doc in docs]
        texts = [doc.text for doc in docs]
        metadatas = [
            {
                "gene_a": doc.gene_a,
                "gene_b": doc.gene_b,
                "interaction_type": doc.interaction_type,
                "conditions": json.dumps(doc.conditions),  # Store as JSON string
                "evidence": doc.evidence,
                "source": doc.source,
                **(doc.metadata or {})
            }
            for doc in docs
        ]

        # Generate embeddings
        embeddings = self.alcf_client.embed_batch(texts)

        # Add to ChromaDB
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def query(
        self,
        query_text: str,
        n_results: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SupportingDocument]:
        """
        Query the vector store for relevant documents.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return (defaults to config.retrieval_top_k)
            where: Metadata filter (e.g., {"gene_a": "lexA"})
            where_document: Document content filter

        Returns:
            List of SupportingDocument instances ordered by similarity
        """
        config = get_alcf_config()
        n_results = n_results or config.retrieval_top_k

        # Generate query embedding
        query_embedding = self.alcf_client.embed(query_text)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

        # Convert to SupportingDocument instances
        supporting_docs = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            text = results['documents'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]

            # Convert distance to similarity score (cosine: 1 - distance)
            similarity_score = 1.0 - distance

            # Extract source from metadata
            source = metadata.get('source', 'unknown')

            # Parse conditions back from JSON
            conditions_str = metadata.get('conditions', '[]')
            try:
                conditions = json.loads(conditions_str)
            except json.JSONDecodeError:
                conditions = []

            # Create metadata dict without ChromaDB-specific fields
            doc_metadata = {k: v for k, v in metadata.items() if k not in ['source', 'conditions']}
            doc_metadata['conditions'] = conditions

            supporting_doc = SupportingDocument(
                doc_id=doc_id,
                source=source,
                text=text,
                similarity_score=similarity_score,
                metadata=doc_metadata
            )
            supporting_docs.append(supporting_doc)

        return supporting_docs

    def query_by_gene_pair(
        self,
        gene_a: str,
        gene_b: str,
        n_results: Optional[int] = None
    ) -> List[SupportingDocument]:
        """
        Query for documents about a specific gene pair.

        Args:
            gene_a: First gene (regulator)
            gene_b: Second gene (target)
            n_results: Number of results to return

        Returns:
            List of SupportingDocument instances
        """
        query_text = f"{gene_a} regulation of {gene_b} gene expression"

        # Filter by gene names
        where_filter = {
            "$and": [
                {"gene_a": gene_a},
                {"gene_b": gene_b}
            ]
        }

        return self.query(
            query_text=query_text,
            n_results=n_results,
            where=where_filter
        )

    def load_from_json(self, json_path: str) -> None:
        """
        Load documents from a JSON file into the vector store.

        Args:
            json_path: Path to JSON file containing document list
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert to LiteratureDocument instances
        docs = [LiteratureDocument(**item) for item in data]

        # Add to vector store
        self.add_documents(docs)

    def count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


# Global vector store instance
_vector_store: Optional[LiteratureVectorStore] = None


def get_vector_store(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None
) -> LiteratureVectorStore:
    """
    Get or create global vector store instance.

    Args:
        collection_name: Name of ChromaDB collection
        persist_directory: Directory for persistence

    Returns:
        LiteratureVectorStore instance
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = LiteratureVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

    return _vector_store
