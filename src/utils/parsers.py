"""
Data Parsers for RegulonDB and M3D files.

Handles the ingestion of:
- network_tf_gene.txt (RegulonDB TF-Gene interactions)
- gene_product.txt (Gene name to b-number mapping)
- E_coli_v4_Build_6_exps.tab (M3D expression matrix)
- E_coli_v4_Build_6_meta.tab (M3D experimental metadata)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import networkx as nx
from loguru import logger


def parse_tf_gene_network(filepath: str | Path) -> Tuple[nx.DiGraph, Dict[str, dict]]:
    """
    Parse RegulonDB network_tf_gene.txt file.
    
    Expected Schema (tab-separated):
    TF_ID | TF_Name | Gene_ID | Gene_Name | Effect | Evidence | EvidenceType
    
    Args:
        filepath: Path to network_tf_gene.txt
        
    Returns:
        Tuple of:
        - NetworkX DiGraph with TF->Gene edges
        - Dictionary of edge metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"RegulonDB network file not found: {filepath}")
    
    graph = nx.DiGraph()
    edge_metadata = {}
    
    # Column indices (based on spec)
    COL_TF_ID = 0
    COL_TF_NAME = 1
    COL_GENE_ID = 2
    COL_GENE_NAME = 3
    COL_EFFECT = 4
    COL_EVIDENCE = 5
    COL_EVIDENCE_TYPE = 6
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            
            # Handle varying column counts
            if len(parts) < 5:
                logger.warning(f"Line {line_num}: Insufficient columns, skipping")
                continue
            
            try:
                tf_id = parts[COL_TF_ID].strip()
                tf_name = parts[COL_TF_NAME].strip().lower()
                gene_id = parts[COL_GENE_ID].strip()
                gene_name = parts[COL_GENE_NAME].strip().lower()
                effect = parts[COL_EFFECT].strip() if len(parts) > COL_EFFECT else "?"
                evidence = parts[COL_EVIDENCE].strip() if len(parts) > COL_EVIDENCE else ""
                evidence_type = parts[COL_EVIDENCE_TYPE].strip() if len(parts) > COL_EVIDENCE_TYPE else "Unknown"
                
                # Normalize effect
                effect = _normalize_effect(effect)
                
                # Normalize evidence type
                evidence_type = _normalize_evidence_type(evidence_type)
                
                # Add nodes with attributes
                graph.add_node(tf_id, name=tf_name, node_type="TF")
                graph.add_node(gene_id, name=gene_name, node_type="Gene")
                
                # Add edge with attributes
                edge_key = f"{tf_id}->{gene_id}"
                edge_attrs = {
                    "tf_name": tf_name,
                    "gene_name": gene_name,
                    "effect": effect,
                    "evidence_raw": evidence,
                    "evidence_type": evidence_type
                }
                
                graph.add_edge(tf_id, gene_id, **edge_attrs)
                edge_metadata[edge_key] = edge_attrs
                
            except Exception as e:
                logger.warning(f"Line {line_num}: Parse error - {e}")
                continue
    
    logger.info(f"Loaded RegulonDB network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, edge_metadata


def _normalize_effect(effect: str) -> str:
    """Normalize interaction effect to standard format."""
    effect = effect.strip()
    if effect in ['+', 'activator', 'activation']:
        return '+'
    elif effect in ['-', 'repressor', 'repression']:
        return '-'
    elif effect in ['+-', 'dual', 'both']:
        return '+-'
    else:
        return '?'


def _normalize_evidence_type(evidence: str) -> str:
    """Normalize evidence type to Strong/Weak/Unknown."""
    evidence_lower = evidence.lower().strip()
    
    # Strong evidence indicators
    strong_indicators = [
        'strong', 'footprinting', 'footprint', 'dnase', 
        'chip-seq', 'chipseq', 'crystal', 'structure',
        'site-directed', 'mutagenesis', 'confirmed'
    ]
    
    # Weak evidence indicators  
    weak_indicators = [
        'weak', 'gea', 'expression', 'indirect',
        'prediction', 'computational', 'inferred'
    ]
    
    for indicator in strong_indicators:
        if indicator in evidence_lower:
            return 'Strong'
    
    for indicator in weak_indicators:
        if indicator in evidence_lower:
            return 'Weak'
    
    return 'Unknown'


def parse_gene_product_mapping(filepath: str | Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse RegulonDB gene_product.txt for gene name to b-number mapping.
    
    Creates the "Rosetta Stone" dictionary for identifier resolution.
    
    Args:
        filepath: Path to gene_product.txt
        
    Returns:
        Tuple of:
        - name_to_bnumber: Dict mapping gene names to b-numbers
        - bnumber_to_name: Dict mapping b-numbers to gene names
    """
    filepath = Path(filepath)
    
    name_to_bnumber = {}
    bnumber_to_name = {}
    
    if not filepath.exists():
        logger.warning(f"Gene product file not found: {filepath}. Using empty mapping.")
        return name_to_bnumber, bnumber_to_name
    
    # Regular expression to match b-numbers (e.g., b0001, b1234)
    bnumber_pattern = re.compile(r'\b(b\d{4})\b', re.IGNORECASE)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            # Try to extract gene name and b-number
            # Format varies, but typically: GeneID | GeneName | Synonyms | BNumber
            gene_name = None
            bnumber = None
            
            # Search for b-number in all columns
            for part in parts:
                match = bnumber_pattern.search(part)
                if match:
                    bnumber = match.group(1).lower()
                    break
            
            # First column after ID is typically gene name
            if len(parts) >= 2:
                gene_name = parts[1].strip().lower()
            
            if gene_name and bnumber:
                name_to_bnumber[gene_name] = bnumber
                bnumber_to_name[bnumber] = gene_name
                
                # Also add synonyms if present
                if len(parts) >= 3:
                    synonyms = parts[2].split(',')
                    for syn in synonyms:
                        syn = syn.strip().lower()
                        if syn and syn not in name_to_bnumber:
                            name_to_bnumber[syn] = bnumber
    
    logger.info(f"Loaded gene mapping: {len(name_to_bnumber)} names -> {len(bnumber_to_name)} b-numbers")
    return name_to_bnumber, bnumber_to_name


def parse_m3d_expression(filepath: str | Path) -> pd.DataFrame:
    """
    Parse M3D expression matrix file (E_coli_v4_Build_6_exps.tab).
    
    The matrix has:
    - Row headers: Probe Set ID or Gene Symbol (b-numbers)
    - Column headers: Experiment IDs
    - Values: Log2-transformed RMA-normalized expression values
    
    Args:
        filepath: Path to expression matrix file
        
    Returns:
        DataFrame with genes as rows (index) and experiments as columns
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"M3D expression file not found: {filepath}")
    
    # Detect separator (tab or comma)
    with open(filepath, 'r') as f:
        first_line = f.readline()
        while first_line.startswith('#'):
            first_line = f.readline()
        
        separator = '\t' if '\t' in first_line else ','
    
    # Read the matrix
    df = pd.read_csv(
        filepath,
        sep=separator,
        index_col=0,
        comment='#',
        na_values=['NA', 'NaN', '', 'null']
    )
    
    # Clean index (gene/probe IDs)
    df.index = df.index.astype(str).str.strip().str.lower()
    df.index.name = 'gene_id'
    
    # Clean column names (experiment IDs)
    df.columns = df.columns.astype(str).str.strip()
    
    # Convert to float
    df = df.astype(float)
    
    # Drop rows/columns that are entirely NaN
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    
    logger.info(f"Loaded M3D expression matrix: {df.shape[0]} genes x {df.shape[1]} experiments")
    return df


def parse_m3d_metadata(filepath: str | Path) -> pd.DataFrame:
    """
    Parse M3D experimental metadata file (E_coli_v4_Build_6_meta.tab).
    
    Expected columns:
    - Experiment_ID: Unique ID matching expression matrix columns
    - Condition_Name: Human-readable description
    - Perturbation: Variable changed (Chemical, Gene knockout, etc.)
    - Concentration/Value: Numerical value
    - Time_Point: Time since perturbation
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        DataFrame with experiments as rows (index) and metadata as columns
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"M3D metadata file not found: {filepath}")
    
    # Detect separator
    with open(filepath, 'r') as f:
        first_line = f.readline()
        while first_line.startswith('#'):
            first_line = f.readline()
        
        separator = '\t' if '\t' in first_line else ','
    
    # Read metadata
    df = pd.read_csv(
        filepath,
        sep=separator,
        index_col=0,
        comment='#'
    )
    
    # Clean index
    df.index = df.index.astype(str).str.strip()
    df.index.name = 'experiment_id'
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    logger.info(f"Loaded M3D metadata: {len(df)} experiments with {len(df.columns)} attributes")
    return df


def create_synthetic_test_data(
    n_genes: int = 100,
    n_experiments: int = 50,
    n_tfs: int = 10,
    output_dir: str | Path = "test_data"
) -> Dict[str, Path]:
    """
    Create synthetic test data for development and testing.
    
    Generates:
    - Synthetic TF-Gene network
    - Synthetic expression matrix with embedded regulatory signals
    - Synthetic metadata
    
    Args:
        n_genes: Number of genes
        n_experiments: Number of experiments
        n_tfs: Number of transcription factors
        output_dir: Directory to save test files
        
    Returns:
        Dictionary mapping file types to paths
    """
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate b-numbers
    all_genes = [f"b{i:04d}" for i in range(1, n_genes + 1)]
    tfs = all_genes[:n_tfs]
    
    # --- Create network_tf_gene.txt ---
    network_lines = ["# TF_ID\tTF_Name\tGene_ID\tGene_Name\tEffect\tEvidence\tEvidenceType"]
    
    np.random.seed(42)
    targets_per_tf = {}
    
    for i, tf in enumerate(tfs):
        tf_name = f"tf{i+1}"
        # Each TF regulates 5-15 genes
        n_targets = np.random.randint(5, 16)
        targets = np.random.choice([g for g in all_genes if g != tf], n_targets, replace=False)
        targets_per_tf[tf] = list(targets)
        
        for j, target in enumerate(targets):
            target_name = f"gene{int(target[1:])}"
            effect = np.random.choice(['+', '-', '+-'], p=[0.4, 0.4, 0.2])
            evidence_type = np.random.choice(['Strong', 'Weak'], p=[0.6, 0.4])
            
            # Use b-numbers directly as IDs for consistency with expression matrix
            network_lines.append(
                f"{tf}\t{tf_name}\t{target}\t{target_name}\t{effect}\t\t{evidence_type}"
            )
    
    network_path = output_dir / "network_tf_gene.txt"
    with open(network_path, 'w') as f:
        f.write('\n'.join(network_lines))
    
    # --- Create gene_product.txt ---
    gene_product_lines = ["# GeneID\tGeneName\tSynonyms\tBNumber"]
    for gene in all_genes:
        gene_num = int(gene[1:])
        name = f"gene{gene_num}"
        # Use b-number as primary ID for consistency
        gene_product_lines.append(f"{gene}\t{name}\t\t{gene}")
    
    gene_product_path = output_dir / "gene_product.txt"
    with open(gene_product_path, 'w') as f:
        f.write('\n'.join(gene_product_lines))
    
    # --- Create expression matrix with embedded signals ---
    exp_ids = [f"exp_{i:03d}" for i in range(n_experiments)]
    
    # Base expression (random)
    expression_data = np.random.normal(8.0, 1.5, (n_genes, n_experiments))
    
    # Embed regulatory signals
    for tf in tfs:
        tf_idx = all_genes.index(tf)
        tf_expr = expression_data[tf_idx, :]
        
        for target in targets_per_tf.get(tf, []):
            target_idx = all_genes.index(target)
            # Add correlated signal (positive correlation = activation)
            correlation_strength = np.random.uniform(0.4, 0.8)
            noise = np.random.normal(0, 0.5, n_experiments)
            expression_data[target_idx, :] += correlation_strength * (tf_expr - np.mean(tf_expr)) + noise
    
    expression_df = pd.DataFrame(
        expression_data,
        index=all_genes,
        columns=exp_ids
    )
    
    expression_path = output_dir / "E_coli_v4_Build_6_exps.tab"
    expression_df.to_csv(expression_path, sep='\t')
    
    # --- Create metadata ---
    conditions = ['Glucose', 'Anaerobic', 'Heat_Shock', 'pH_Stress', 'Stationary']
    perturbations = ['Chemical', 'Temperature', 'Oxygen', 'pH', 'Growth_Phase']
    
    metadata_rows = []
    for i, exp_id in enumerate(exp_ids):
        cond = conditions[i % len(conditions)]
        pert = perturbations[i % len(perturbations)]
        metadata_rows.append({
            'Experiment_ID': exp_id,
            'Condition': cond,
            'Perturbation': pert,
            'Value': np.random.uniform(0.1, 10.0),
            'Time_Point': f"{np.random.randint(0, 60)}min"
        })
    
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.set_index('Experiment_ID', inplace=True)
    
    metadata_path = output_dir / "E_coli_v4_Build_6_meta.tab"
    metadata_df.to_csv(metadata_path, sep='\t')
    
    logger.info(f"Created synthetic test data in {output_dir}")
    
    return {
        "network": network_path,
        "gene_product": gene_product_path,
        "expression": expression_path,
        "metadata": metadata_path
    }

