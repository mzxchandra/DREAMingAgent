"""
Loader Node: The "Librarian"

Ingests data files from RegulonDB and M3D, establishing the ground truth
knowledge graph and expression matrix for the reconciliation workflow.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx
import pandas as pd
from loguru import logger

from ..state import AgentState
from ..utils.parsers import (
    parse_tf_gene_network,
    parse_gene_product_mapping,
    parse_m3d_expression,
    parse_m3d_metadata,
    create_synthetic_test_data
)


class LoaderConfig:
    """Configuration for data loading paths."""
    
    def __init__(
        self,
        regulondb_network_path: str | Path,
        regulondb_gene_product_path: str | Path,
        m3d_expression_path: Optional[str | Path] = None,
        m3d_metadata_path: Optional[str | Path] = None,
        use_synthetic: bool = False,
        synthetic_output_dir: str | Path = "test_data"
    ):
        self.regulondb_network_path = Path(regulondb_network_path)
        self.regulondb_gene_product_path = Path(regulondb_gene_product_path)
        self.m3d_expression_path = Path(m3d_expression_path) if m3d_expression_path else None
        self.m3d_metadata_path = Path(m3d_metadata_path) if m3d_metadata_path else None
        self.use_synthetic = use_synthetic
        self.synthetic_output_dir = Path(synthetic_output_dir)


# Global config - will be set before workflow runs
_loader_config: Optional[LoaderConfig] = None


def set_loader_config(config: LoaderConfig):
    """Set the global loader configuration."""
    global _loader_config
    _loader_config = config


def loader_node(state: AgentState) -> Dict[str, Any]:
    """
    Loader Node: Ingests RegulonDB and M3D data files.
    
    This node is the "Librarian" of the system. It:
    1. Parses the TF-Gene network from RegulonDB
    2. Builds the gene name to b-number mapping
    3. Loads the M3D expression matrix
    4. Loads the M3D experimental metadata
    5. Initializes the TF queue for processing
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state dictionary with loaded data
    """
    global _loader_config
    
    logger.info("=== LOADER NODE: Ingesting data files ===")
    
    errors = []
    
    # Handle synthetic data generation for testing
    if _loader_config and _loader_config.use_synthetic:
        logger.info("Generating synthetic test data...")
        paths = create_synthetic_test_data(
            output_dir=_loader_config.synthetic_output_dir
        )
        regulondb_network_path = paths["network"]
        regulondb_gene_product_path = paths["gene_product"]
        m3d_expression_path = paths["expression"]
        m3d_metadata_path = paths["metadata"]
    elif _loader_config:
        regulondb_network_path = _loader_config.regulondb_network_path
        regulondb_gene_product_path = _loader_config.regulondb_gene_product_path
        m3d_expression_path = _loader_config.m3d_expression_path
        m3d_metadata_path = _loader_config.m3d_metadata_path
    else:
        # Default paths (for testing)
        logger.warning("No loader config set, using default test paths")
        regulondb_network_path = Path("data/network_tf_gene.txt")
        regulondb_gene_product_path = Path("data/gene_product.txt")
        m3d_expression_path = Path("data/E_coli_v4_Build_6_exps.tab")
        m3d_metadata_path = Path("data/E_coli_v4_Build_6_meta.tab")
    
    # --- 1. Load Gene Product Mapping (Rosetta Stone) ---
    logger.info("Loading gene name to b-number mapping...")
    try:
        gene_name_to_bnumber, bnumber_to_gene_name, _ = parse_gene_product_mapping(
            regulondb_gene_product_path
        )
    except Exception as e:
        logger.warning(f"Failed to load gene product mapping: {e}")
        gene_name_to_bnumber = {}
        bnumber_to_gene_name = {}
        errors.append(f"Gene product mapping: {e}")
    
    # --- 2. Load RegulonDB TF-Gene Network ---
    logger.info("Loading RegulonDB TF-Gene regulatory network...")
    try:
        literature_graph, edge_metadata = parse_tf_gene_network(regulondb_network_path)
        
        # Convert node IDs to b-numbers where possible
        literature_graph = _convert_graph_to_bnumbers(
            literature_graph,
            gene_name_to_bnumber
        )
        
        logger.info(
            f"Literature graph: {literature_graph.number_of_nodes()} nodes, "
            f"{literature_graph.number_of_edges()} edges"
        )
    except Exception as e:
        logger.error(f"Failed to load RegulonDB network: {e}")
        literature_graph = nx.DiGraph()
        errors.append(f"RegulonDB network: {e}")
    
    # --- 3. Load M3D Expression Matrix ---
    if m3d_expression_path:
        logger.info("Loading M3D expression matrix...")
        try:
            expression_matrix = parse_m3d_expression(m3d_expression_path)
            
            # Ensure index uses b-numbers
            expression_matrix = _normalize_expression_index(
                expression_matrix,
                gene_name_to_bnumber
            )
            
            logger.info(
                f"Expression matrix: {expression_matrix.shape[0]} genes x "
                f"{expression_matrix.shape[1]} experiments"
            )
        except Exception as e:
            logger.error(f"Failed to load M3D expression matrix: {e}")
            expression_matrix = pd.DataFrame()
            errors.append(f"M3D expression: {e}")
    else:
        logger.info("No M3D expression path provided, skipping...")
        expression_matrix = pd.DataFrame()
    
    # --- 4. Load M3D Metadata ---
    if m3d_metadata_path:
        logger.info("Loading M3D experimental metadata...")
        try:
            metadata = parse_m3d_metadata(m3d_metadata_path)
            logger.info(f"Metadata: {len(metadata)} experiments")
        except Exception as e:
            logger.error(f"Failed to load M3D metadata: {e}")
            metadata = pd.DataFrame()
            errors.append(f"M3D metadata: {e}")
    else:
        logger.info("No M3D metadata path provided, skipping...")
        metadata = pd.DataFrame()
    
    # --- 5. Initialize TF Queue ---
    # Extract all TFs from the literature graph
    tf_queue = _extract_tf_queue(literature_graph)
    logger.info(f"Initialized TF queue with {len(tf_queue)} transcription factors")
    
    # --- 6. Update state ---
    return {
        "literature_graph": literature_graph,
        "gene_name_to_bnumber": gene_name_to_bnumber,
        "bnumber_to_gene_name": bnumber_to_gene_name,
        "expression_matrix": expression_matrix,
        "metadata": metadata,
        "tf_queue": tf_queue,
        "current_batch_tfs": [],
        "active_sample_indices": list(expression_matrix.columns) if not expression_matrix.empty else [],
        "current_context": "all_conditions",
        "analysis_results": {},
        "reconciliation_log": [],
        "novel_hypotheses": [],
        "false_positive_candidates": [],
        "iteration_count": 0,
        "errors": errors,
        "status": "processing" if not errors else "error"
    }


def _convert_graph_to_bnumbers(
    graph: nx.DiGraph,
    name_to_bnumber: Dict[str, str]
) -> nx.DiGraph:
    """
    Convert graph node IDs to b-numbers where possible.
    
    Also updates the 'bnumber' attribute on each node.
    """
    # Create mapping from current IDs to b-numbers
    id_mapping = {}
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_name = node_data.get('name', '').lower()
        
        # Try to find b-number
        bnumber = None
        
        # Check if node ID is already a b-number
        if node.lower().startswith('b') and len(node) == 5:
            bnumber = node.lower()
        # Check by name
        elif node_name in name_to_bnumber:
            bnumber = name_to_bnumber[node_name]
        # Check by ID (ECK format might contain b-number)
        elif 'ECK' in node:
            # Try to extract from associated data
            pass
        
        if bnumber:
            id_mapping[node] = bnumber
            graph.nodes[node]['bnumber'] = bnumber
        else:
            # Keep original if no b-number found
            id_mapping[node] = node
    
    # Relabel nodes
    return nx.relabel_nodes(graph, id_mapping)


def _normalize_expression_index(
    df: pd.DataFrame,
    name_to_bnumber: Dict[str, str]
) -> pd.DataFrame:
    """
    Normalize expression matrix index to use b-numbers.
    """
    new_index = []
    
    for idx in df.index:
        idx_str = str(idx).lower()
        
        # Check if already a b-number
        if idx_str.startswith('b') and len(idx_str) == 5:
            new_index.append(idx_str)
        # Try to map name to b-number
        elif idx_str in name_to_bnumber:
            new_index.append(name_to_bnumber[idx_str])
        else:
            new_index.append(idx_str)
    
    df.index = new_index
    return df


def _extract_tf_queue(graph: nx.DiGraph) -> list:
    """
    Extract list of TFs (regulators) from the literature graph.
    
    A TF is any node with outgoing edges (it regulates something).
    """
    tfs = []
    
    for node in graph.nodes():
        # Check if this node has outgoing edges (is a regulator)
        if graph.out_degree(node) > 0:
            tfs.append(node)
    
    # Sort by number of targets (most connected first)
    tfs.sort(key=lambda x: graph.out_degree(x), reverse=True)
    
    return tfs


