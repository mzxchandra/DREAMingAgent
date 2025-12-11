"""
Ingest literature evidence from NCBI LitSense into local Vector Store.

Usage:
    python scripts/ingest_literature.py --network data/NetworkRegulatorGene.tsv --limit 10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Set, Tuple
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.parsers import parse_tf_gene_network, parse_gene_product_mapping
from src.utils.litsense_client import LitSenseClient
from src.utils.vector_store import get_vector_store, LiteratureDocument

def main():
    parser = argparse.ArgumentParser(description="Ingest literature for DREAMing Agent")
    parser.add_argument("--network", required=True, help="Path to NetworkRegulatorGene.tsv")
    parser.add_argument("--gene-products", help="Path to GeneProductAllIdentifiersSet.tsv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of edges to process")
    parser.add_argument("--output-dir", default="chroma_db", help="ChromaDB persistence directory")
    args = parser.parse_args()

    # Setup
    logger.info("Initializing ingestion...")
    
    # Load Network
    try:
        graph, metadata = parse_tf_gene_network(args.network)
    except Exception as e:
        logger.error(f"Failed to load network: {e}")
        return

    # Load Gene Names (optional but helpful for better queries)
    bnumber_to_name = {}
    if args.gene_products:
        _, bnumber_to_name, _ = parse_gene_product_mapping(args.gene_products)

    # Initialize Client & Store
    client = LitSenseClient()
    vector_store = get_vector_store(persist_directory=args.output_dir)
    
    # Identify Unique TFs to Process
    # We want to iterate TFs, not edges
    unique_tfs = set()
    for u, _, data in graph.edges(data=True):
        tf_id = u
        tf_name = data.get('tf_name', bnumber_to_name.get(tf_id, tf_id))
        unique_tfs.add((tf_id, tf_name))
        
    unique_tfs = sorted(list(unique_tfs)) # Sort for reproducibility
    
    logger.info(f"Processing {len(unique_tfs)} unique transcription factors...")
    
    # Limit per TF
    # Limit per TF
    DOCS_LIMIT_PER_TF = 10
    TOTAL_DOCS_LIMIT = None # Process all TFs
    
    processed_count = 0
    docs_added = 0
    
    for tf_id, tf_name in tqdm(unique_tfs):
        # Global limit check (disabled)
        if TOTAL_DOCS_LIMIT and docs_added >= TOTAL_DOCS_LIMIT:
            logger.info(f"Reached global limit of {TOTAL_DOCS_LIMIT} documents. Stopping.")
            break

        # Skip if names are just IDs (harder to search)
        if tf_name.startswith('RDB'):
            continue
            
        # Query LitSense for TF Broad Context
        try:
            results = client.get_tf_context(tf_name, limit=DOCS_LIMIT_PER_TF)
            
            if not results:
                continue
                
            # Deduplicate by PMID
            unique_results = {}
            for res in results:
                pmid = res.get('pmid')
                if pmid and pmid not in unique_results:
                    unique_results[pmid] = res
            
            # Convert to Documents
            new_docs = []
            for pmid, res in unique_results.items():
                text = res.get('text', '')
                if not text:
                    continue
                
                # Double check per-TF limit (though query limit handles it mostly)
                if len(new_docs) >= DOCS_LIMIT_PER_TF:
                    break
                    
                # Store generic TF context
                # doc_id = litsense_TFNAME_PMID
                doc_id = f"litsense_{tf_name}_{pmid}"
                
                doc = LiteratureDocument(
                    doc_id=doc_id,
                    gene_a=tf_name,
                    gene_b="N/A", # Broad context, no specific target
                    interaction_type="regulation_context", 
                    conditions=[], 
                    evidence="text_mining",
                    source=f"PubMed:{pmid}",
                    text=text,
                    metadata={"score": res.get('score', 0.0), "context_type": "broad_tf"}
                )
                new_docs.append(doc)
            
            # Add to Store
            if new_docs:
                vector_store.add_documents(new_docs)
                docs_added += len(new_docs)
                
        except Exception as e:
            logger.error(f"Error processing TF {tf_name}: {e}")
            
        processed_count += 1
        
    logger.info(f"Ingestion complete. Processed {processed_count} TFs. Added {docs_added} documents.")

if __name__ == "__main__":
    main()
