"""
Data Parsers for RegulonDB and M3D files.

Handles the ingestion of:
- NetworkRegulatorGene.tsv (RegulonDB TF-Gene interactions) - REAL FORMAT
- GeneProductAllIdentifiersSet.tsv (Gene name to b-number mapping) - REAL FORMAT
- E_coli_v4_Build_6_exps.tab (M3D expression matrix)
- E_coli_v4_Build_6_meta.tab (M3D experimental metadata)

Updated to handle real RegulonDB format with dynamic column detection.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import networkx as nx
from loguru import logger


# ============================================================================
# Evidence Level Mappings (RegulonDB format)
# ============================================================================

EVIDENCE_LEVEL_MAP = {
    'C': 'Confirmed',
    'S': 'Strong', 
    'W': 'Weak',
    '?': 'Unknown',
    # Fallback for text formats
    'confirmed': 'Confirmed',
    'strong': 'Strong',
    'weak': 'Weak',
    'unknown': 'Unknown'
}

INTERACTION_EFFECT_MAP = {
    '+': '+',
    '-': '-',
    '+-': '+-',
    '-+': '+-',
    '?': '?',
    'activator': '+',
    'repressor': '-',
    'dual': '+-',
    'unknown': '?'
}


# ============================================================================
# RegulonDB Network Parser (Real Format)
# ============================================================================

def parse_tf_gene_network(filepath: str | Path) -> Tuple[nx.DiGraph, Dict[str, dict]]:
    """
    Parse RegulonDB NetworkRegulatorGene.tsv file (real format).
    
    Dynamically detects columns based on header patterns to handle
    year-to-year variations in RegulonDB exports.
    
    Expected columns (flexible order):
    - regulatorId / TF_ID
    - regulatorName / TF_Name  
    - regulatedId / Gene_ID
    - regulatedName / Gene_Name
    - function / Effect
    - confidenceLevel / Evidence
    
    Args:
        filepath: Path to NetworkRegulatorGene.tsv
        
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
    
    # Read file and detect column structure
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find header line (starts with column numbers or contains key terms)
    header_line = None
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('##'):
            continue
        
        # Look for header patterns
        if ('regulator' in line.lower() and 'regulated' in line.lower()) or \
           ('1)' in line and '2)' in line):
            header_line = line
            data_start_idx = i + 1
            break
    
    if header_line is None:
        logger.warning("Could not find header line, using default column mapping")
        col_mapping = _get_default_network_columns()
    else:
        col_mapping = _detect_network_columns(header_line)
    
    logger.info(f"Detected network columns: {col_mapping}")
    
    # Parse data lines
    for line_num, line in enumerate(lines[data_start_idx:], data_start_idx + 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        
        if len(parts) < max(col_mapping.values()) + 1:
            logger.warning(f"Line {line_num}: Insufficient columns ({len(parts)}), skipping")
            continue
        
        try:
            # Extract fields using detected column mapping
            tf_id = parts[col_mapping['tf_id']].strip()
            tf_name = parts[col_mapping['tf_name']].strip().lower()
            gene_id = parts[col_mapping['gene_id']].strip()
            gene_name = parts[col_mapping['gene_name']].strip().lower()
            
            # Handle optional columns
            effect = parts[col_mapping.get('effect', -1)].strip() if col_mapping.get('effect', -1) >= 0 else "?"
            evidence = parts[col_mapping.get('evidence', -1)].strip() if col_mapping.get('evidence', -1) >= 0 else "Unknown"
            
            # Normalize values
            effect = _normalize_effect(effect)
            evidence_level = _normalize_evidence_level(evidence)
            
            # Add nodes with attributes
            graph.add_node(tf_id, name=tf_name, node_type="TF", original_id=tf_id)
            graph.add_node(gene_id, name=gene_name, node_type="Gene", original_id=gene_id)
            
            # Add edge with attributes
            edge_key = f"{tf_id}->{gene_id}"
            edge_attrs = {
                "tf_name": tf_name,
                "gene_name": gene_name,
                "effect": effect,
                "evidence_raw": evidence,
                "evidence_type": evidence_level,
                "original_tf_id": tf_id,
                "original_gene_id": gene_id
            }
            
            graph.add_edge(tf_id, gene_id, **edge_attrs)
            edge_metadata[edge_key] = edge_attrs
            
        except Exception as e:
            logger.warning(f"Line {line_num}: Parse error - {e}")
            continue
    
    logger.info(f"Loaded RegulonDB network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, edge_metadata


def _detect_network_columns(header_line: str) -> Dict[str, int]:
    """
    Dynamically detect column indices from header line.
    
    Looks for patterns like:
    - regulatorId, regulatorName, regulatedId, regulatedName, function, confidenceLevel
    - 1)regulatorId, 2)regulatorName, etc.
    """
    columns = header_line.split('\t')
    mapping = {}
    
    for i, col in enumerate(columns):
        col_lower = col.lower().strip()
        
        # Remove numbering prefixes like "1)", "2)", etc.
        col_clean = re.sub(r'^\d+\)\s*', '', col_lower)
        
        # TF identification
        if 'regulatorid' in col_clean or 'tf_id' in col_clean:
            mapping['tf_id'] = i
        elif 'regulatorname' in col_clean or 'tf_name' in col_clean:
            mapping['tf_name'] = i
        
        # Gene identification  
        elif 'regulatedid' in col_clean or 'gene_id' in col_clean:
            mapping['gene_id'] = i
        elif 'regulatedname' in col_clean or 'gene_name' in col_clean:
            mapping['gene_name'] = i
        
        # Regulation properties
        elif 'function' in col_clean or 'effect' in col_clean:
            mapping['effect'] = i
        elif 'confidence' in col_clean or 'evidence' in col_clean:
            mapping['evidence'] = i
    
    # Validate required columns
    required = ['tf_id', 'tf_name', 'gene_id', 'gene_name']
    missing = [col for col in required if col not in mapping]
    
    if missing:
        logger.warning(f"Missing required columns: {missing}. Using default mapping.")
        return _get_default_network_columns()
    
    return mapping


def _get_default_network_columns() -> Dict[str, int]:
    """Default column mapping for RegulonDB NetworkRegulatorGene.tsv"""
    return {
        'tf_id': 0,        # regulatorId
        'tf_name': 1,      # regulatorName
        'gene_id': 3,      # regulatedId (skip column 2: RegulatorGeneName)
        'gene_name': 4,    # regulatedName
        'effect': 5,       # function
        'evidence': 6      # confidenceLevel
    }


# ============================================================================
# Gene Product Mapping Parser (Real Format)
# ============================================================================

def parse_gene_product_mapping(filepath: str | Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse RegulonDB GeneProductAllIdentifiersSet.tsv for gene name to b-number mapping.
    
    Extracts b-numbers from the complex otherDbsGeneIds field using regex.
    
    Args:
        filepath: Path to GeneProductAllIdentifiersSet.tsv
        
    Returns:
        Tuple of:
        - name_to_bnumber: Dict mapping gene names to b-numbers
        - bnumber_to_name: Dict mapping b-numbers to gene names
        - regulondb_to_bnumber: Dict mapping RegulonDB IDs to b-numbers
    """
    filepath = Path(filepath)
    
    name_to_bnumber = {}
    bnumber_to_name = {}
    regulondb_to_bnumber = {}
    
    if not filepath.exists():
        logger.warning(f"Gene product file not found: {filepath}. Using empty mapping.")
        return name_to_bnumber, bnumber_to_name
    
    # Read file and detect structure
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find header and detect columns
    header_line = None
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('##'):
            continue
        
        if ('geneid' in line.lower() and 'genename' in line.lower()) or \
           ('1)' in line and '2)' in line):
            header_line = line
            data_start_idx = i + 1
            break
    
    if header_line is None:
        logger.warning("Could not find header line in gene product file")
        col_mapping = _get_default_gene_product_columns()
    else:
        col_mapping = _detect_gene_product_columns(header_line)
    
    logger.info(f"Detected gene product columns: {col_mapping}")
    
    # Parse data lines
    for line_num, line in enumerate(lines[data_start_idx:], data_start_idx + 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        
        if len(parts) < max(col_mapping.values()) + 1:
            continue
        
        try:
            # Extract basic info
            gene_id = parts[col_mapping['gene_id']].strip()
            gene_name = parts[col_mapping['gene_name']].strip().lower()
            
            # Extract b-number from otherDbsGeneIds field
            other_dbs_col = col_mapping.get('other_dbs', -1)
            if other_dbs_col >= 0 and other_dbs_col < len(parts):
                other_dbs = parts[other_dbs_col]
                bnumber = _extract_bnumber(other_dbs)
                
                if bnumber and gene_name:
                    name_to_bnumber[gene_name] = bnumber
                    bnumber_to_name[bnumber] = gene_name
                    regulondb_to_bnumber[gene_id] = bnumber
                    
                    # Also add synonyms if present
                    synonyms_col = col_mapping.get('synonyms', -1)
                    if synonyms_col >= 0 and synonyms_col < len(parts):
                        synonyms_text = parts[synonyms_col]
                        synonyms = _extract_synonyms(synonyms_text)
                        for syn in synonyms:
                            if syn and syn not in name_to_bnumber:
                                name_to_bnumber[syn] = bnumber
            
        except Exception as e:
            logger.debug(f"Line {line_num}: Parse error - {e}")
            continue
    
    logger.info(f"Loaded gene mapping: {len(name_to_bnumber)} names -> {len(bnumber_to_name)} b-numbers")
    return name_to_bnumber, bnumber_to_name, regulondb_to_bnumber


def _detect_gene_product_columns(header_line: str) -> Dict[str, int]:
    """Detect column structure for gene product file."""
    columns = header_line.split('\t')
    mapping = {}
    
    for i, col in enumerate(columns):
        col_lower = col.lower().strip()
        col_clean = re.sub(r'^\d+\)\s*', '', col_lower)
        
        # Be more specific with matching to avoid conflicts
        if col_clean == 'geneid':
            mapping['gene_id'] = i
        elif col_clean == 'genename':
            mapping['gene_name'] = i
        elif 'genesynonyms' in col_clean:
            mapping['synonyms'] = i
        elif 'otherdbsgeneid' in col_clean:
            mapping['other_dbs'] = i
        elif 'productsynonyms' in col_clean:
            # Don't override gene synonyms with product synonyms
            if 'synonyms' not in mapping:
                mapping['synonyms'] = i
    
    return mapping


def _get_default_gene_product_columns() -> Dict[str, int]:
    """Default column mapping for GeneProductAllIdentifiersSet.tsv"""
    return {
        'gene_id': 0,      # geneId
        'gene_name': 1,    # geneName
        'synonyms': 5,     # geneSynonyms
        'other_dbs': 6     # otherDbsGeneIds
    }


def _extract_bnumber(other_dbs_string: str) -> Optional[str]:
    """
    Extract b-number from complex otherDbsGeneIds string.
    
    Example input: "[ASKA:ECK1450+b1456+JW1451+rhsE][KEIO:ECK1450+b1456+...]"
    Expected output: "b1456"
    """
    if not other_dbs_string:
        return None
    
    # Look for b-number pattern: b followed by 4 digits
    match = re.search(r'\b(b\d{4})\b', other_dbs_string, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _extract_synonyms(synonyms_string: str) -> List[str]:
    """Extract gene synonyms from synonyms field."""
    if not synonyms_string:
        return []
    
    # Split by common delimiters and clean
    synonyms = re.split(r'[,;|]', synonyms_string)
    cleaned = []
    
    for syn in synonyms:
        syn = syn.strip().lower()
        if syn and not syn.startswith('[') and len(syn) > 1:
            cleaned.append(syn)
    
    return cleaned


# ============================================================================
# Normalization Functions
# ============================================================================

def _normalize_effect(effect: str) -> str:
    """Normalize interaction effect to standard format."""
    effect_clean = effect.strip().lower()
    return INTERACTION_EFFECT_MAP.get(effect_clean, effect_clean)


def _normalize_evidence_level(evidence: str) -> str:
    """Normalize evidence level to standard format."""
    evidence_clean = evidence.strip()
    
    # Try direct mapping first
    if evidence_clean in EVIDENCE_LEVEL_MAP:
        return EVIDENCE_LEVEL_MAP[evidence_clean]
    
    # Try lowercase mapping
    evidence_lower = evidence_clean.lower()
    if evidence_lower in EVIDENCE_LEVEL_MAP:
        return EVIDENCE_LEVEL_MAP[evidence_lower]
    
    # Fallback pattern matching
    if 'strong' in evidence_lower or 's' == evidence_lower:
        return 'Strong'
    elif 'weak' in evidence_lower or 'w' == evidence_lower:
        return 'Weak'
    elif 'confirm' in evidence_lower or 'c' == evidence_lower:
        return 'Confirmed'
    else:
        return 'Unknown'


# ============================================================================
# M3D Parsers (Unchanged)
# ============================================================================

def parse_m3d_expression(filepath: str | Path) -> pd.DataFrame:
    """Parse M3D expression matrix file (unchanged from original)."""
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
    """Parse M3D experimental metadata file (unchanged from original)."""
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


# ============================================================================
# Synthetic Data Generator (For Testing)
# ============================================================================

def create_synthetic_test_data(
    n_genes: int = 100,
    n_experiments: int = 50,
    n_tfs: int = 10,
    output_dir: str | Path = "test_data"
) -> Dict[str, Path]:
    """Create synthetic test data matching real RegulonDB format."""
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate b-numbers
    all_genes = [f"b{i:04d}" for i in range(1, n_genes + 1)]
    tfs = all_genes[:n_tfs]
    
    # --- Create NetworkRegulatorGene.tsv (Real format) ---
    network_lines = [
        "## Network Regulatory Interactions in Escherichia coli K-12 substr. MG1655",
        "## Columns:",
        "## (1) regulatorId. Regulator identifier", 
        "## (2) regulatorName. Regulator Name",
        "## (3) RegulatorGeneName. Gene(s) coding for the TF",
        "## (4) regulatedId. Gene ID regulated by the Regulator",
        "## (5) regulatedName. Gene regulated by the Regulator", 
        "## (6) function. Regulatory Function (+/-/+-/?)",
        "## (7) confidenceLevel. Evidence level (C/S/W/?)",
        "1)regulatorId\t2)regulatorName\t3)RegulatorGeneName\t4)regulatedId\t5)regulatedName\t6)function\t7)confidenceLevel"
    ]
    
    np.random.seed(42)
    targets_per_tf = {}
    
    for i, tf in enumerate(tfs):
        tf_name = f"tf{i+1}"
        tf_gene_name = f"gene{i+1}"
        
        # Each TF regulates 5-15 genes
        n_targets = np.random.randint(5, 16)
        targets = np.random.choice([g for g in all_genes if g != tf], n_targets, replace=False)
        targets_per_tf[tf] = list(targets)
        
        for target in targets:
            target_name = f"gene{int(target[1:])}"
            effect = np.random.choice(['+', '-', '+-'], p=[0.4, 0.4, 0.2])
            evidence_type = np.random.choice(['S', 'W'], p=[0.6, 0.4])
            
            # Create RegulonDB-style IDs
            tf_id = f"RDBECOLICNC{i:05d}"
            target_id = f"RDBECOLIGNC{int(target[1:]):05d}"
            
            network_lines.append(
                f"{tf_id}\t{tf_name}\t{tf_gene_name}\t{target_id}\t{target_name}\t{effect}\t{evidence_type}"
            )
    
    network_path = output_dir / "NetworkRegulatorGene.tsv"
    with open(network_path, 'w') as f:
        f.write('\n'.join(network_lines))
    
    # --- Create GeneProductAllIdentifiersSet.tsv (Real format) ---
    gene_product_lines = [
        "## Gene and Gene Product information for Escherichia coli K-12 substr. MG1655",
        "## Columns:",
        "## (1) Gene identifier assigned by RegulonDB",
        "## (2) Gene name", 
        "## (3) Gene left end position in the genome",
        "## (4) Gene right end position in the genome",
        "## (5) DNA strand where the gene is coded",
        "## (6) other gene synonyms",
        "## (7) Other database's id related to gene",
        "## (8) Product identifier of the gene",
        "## (9) Product name of the gene",
        "## (10) Other products synonyms", 
        "## (11) Other database's id related to product",
        "1)geneId\t2)geneName\t3)leftEndPos\t4)rightEndPos\t5)strand\t6)geneSynonyms\t7)otherDbsGeneIds\t8)productId\t9)productName\t10)productSynonyms\t11)otherDbsProductsIds"
    ]
    
    for gene in all_genes:
        gene_num = int(gene[1:])
        name = f"gene{gene_num}"
        gene_id = f"RDBECOLIGNC{gene_num:05d}"
        
        # Create realistic otherDbsGeneIds with b-number embedded
        other_dbs = f"[ASKA:ECK{gene_num:04d}+{gene}+JW{gene_num:04d}+{name}][KEIO:ECK{gene_num:04d}+{gene}+JW{gene_num:04d}+{name}]"
        
        gene_product_lines.append(
            f"{gene_id}\t{name}\t{gene_num*1000}\t{gene_num*1000+500}\tforward\t{name}5,EG{gene_num:05d}\t{other_dbs}\tRDBECOLIPDC{gene_num:05d}\tprotein {name}\t\t"
        )
    
    gene_product_path = output_dir / "GeneProductAllIdentifiersSet.tsv"
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
            # Add correlated signal
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