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
    
    # Identify Edges to Process
    edges = list(graph.edges(data=True))
    if args.limit:
        edges = edges[:args.limit]
        
    logger.info(f"Processing {len(edges)} edges...")
    
    processed_count = 0
    docs_added = 0
    
    for u, v, data in tqdm(edges):
        tf_id = u
        target_id = v
        
        # Resolve names
        tf_name = data.get('tf_name', bnumber_to_name.get(tf_id, tf_id))
        target_name = data.get('gene_name', bnumber_to_name.get(target_id, target_id))
        
        # Skip if names are just IDs (harder to search)
        if tf_name.startswith('RDB') or target_name.startswith('RDB'):
            logger.warning(f"Skipping {tf_id}->{target_id} due to missing common names")
            continue
            
        # Query LitSense
        try:
            results = client.get_evidence_for_interaction(tf_name, target_name)
            
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
                
                # Use deterministic ID based on gene pair and PMID
                doc_id = f"litsense_{tf_name}_{target_name}_{pmid}"
                
                doc = LiteratureDocument(
                    doc_id=doc_id,
                    gene_a=tf_name,
                    gene_b=target_name,
                    interaction_type="regulation", 
                    conditions=[], 
                    evidence="text_mining",
                    source=f"PubMed:{pmid}",
                    text=text,
                    metadata={"score": res.get('score', 0.0)}
                )
                new_docs.append(doc)
            
            # Add to Store
            if new_docs:
                vector_store.add_documents(new_docs)
                docs_added += len(new_docs)
                
        except Exception as e:
            logger.error(f"Error processing {tf_name}->{target_name}: {e}")
            
        processed_count += 1
        
    logger.info(f"Ingestion complete. Processed {processed_count} edges. Added {docs_added} documents.")

if __name__ == "__main__":
    main()
