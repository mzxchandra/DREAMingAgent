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
    # Extract b-numbers if present (e.g. "aaaD_b4634_3" -> "b4634")
    def _extract_bnum(s):
        s = str(s).strip().lower()
        match = re.search(r'b\d{4}', s)
        return match.group(0) if match else s

    df.index = df.index.map(_extract_bnum)
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
