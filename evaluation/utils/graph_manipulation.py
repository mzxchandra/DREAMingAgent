import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Tuple, List, Set

def load_network_as_graph(network_path: Path) -> nx.DiGraph:
    """Load RegulonDB network file into a NetworkX DiGraph."""
    df = pd.read_csv(network_path, sep='\t', comment='#')
    
    # Expected columns based on parsers.py:
    # 1)regulatorId	2)regulatorName	3)RegulatorGeneName	4)regulatedId	5)regulatedName	6)function	7)confidenceLevel
    
    # Rename columns to be friendlier
    df.columns = [
        "regulatorId", "regulatorName", "regulatorGeneName",
        "regulatedId", "regulatedName", "function", "confidenceLevel"
    ]
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = str(row['regulatorName']).strip().lower() # Lowercase for consistency
        target = str(row['regulatedName']).strip().lower()
        
        # Store all attributes
        attrs = row.to_dict()
        edge_id = f"{source}→{target}"
        attrs['edge_id'] = edge_id
        
        G.add_edge(source, target, **attrs)
        
    return G

def inject_false_edges(
    graph: nx.DiGraph, 
    n_edges: int, 
    evidence_level: str = "W", 
    effect: str = "+",
    focus_node: str = None
) -> Tuple[nx.DiGraph, List[str]]:
    """Inject false positive edges into the graph.
    
    Args:
        focus_node: If provided, all injected edges will have this node as SOURCE.
    """
    G = graph.copy()
    nodes = list(G.nodes())
    injected_edges = []
    
    if len(nodes) < 2:
        return G, []

    attempts = 0
    max_attempts = n_edges * 100
    
    while len(injected_edges) < n_edges and attempts < max_attempts:
        if focus_node and focus_node in nodes:
            source = focus_node
            target = np.random.choice(nodes)
        else:
            source = np.random.choice(nodes)
            target = np.random.choice(nodes)
        
        if source != target and not G.has_edge(source, target):
            # Create plausible attributes
            attrs = {
                "regulatorId": f"FAKE_{source}",
                "regulatorName": source,
                "regulatorGeneName": source, # simplified
                "regulatedId": f"FAKE_{target}",
                "regulatedName": target,
                "function": effect,
                "confidenceLevel": evidence_level,
                "is_injected": True
            }
            edge_id = f"{source}→{target}"
            attrs['edge_id'] = edge_id
            
            G.add_edge(source, target, **attrs)
            injected_edges.append(edge_id)
            
        attempts += 1
        
    return G, injected_edges

def delete_true_edges(
    graph: nx.DiGraph, 
    n_edges: int,
    evidence_level: str = None,
    focus_node: str = None
) -> Tuple[nx.DiGraph, List[str]]:
    """Delete existing true edges from the graph.
    
    Args:
        evidence_level: If provided, only delete edges with this confidence level (e.g. 'S', 'C').
        focus_node: If provided, only delete edges where this node is the SOURCE.
    """
    G = graph.copy()
    deleted_edges = []
    
    # Filter candidates
    candidates = []
    for u, v in G.edges():
        # Check focus node constraint
        if focus_node and u != focus_node:
            continue
            
        # Check evidence level constraint
        if evidence_level:
            edge_evidence = G.edges[u, v].get('confidenceLevel', '')
            # Simple containment check handles 'Strong' vs 'S' variations
            if evidence_level.lower() not in edge_evidence.lower() and \
               edge_evidence.lower() not in evidence_level.lower():
                continue
                
        # Don't delete edges we just injected!
        if G.edges[u, v].get('is_injected', False):
            continue
            
        candidates.append((u, v))
    
    if not candidates:
        return G, []
        
    # Randomly select edges to delete
    indices = np.random.choice(len(candidates), size=min(n_edges, len(candidates)), replace=False)
    
    for idx in indices:
        u, v = candidates[idx]
        edge_id = G.edges[u, v].get('edge_id')
        G.remove_edge(u, v)
        deleted_edges.append(edge_id)
        
    return G, deleted_edges

def save_graph_to_regulondb_format(graph: nx.DiGraph, output_path: Path):
    """Save graph back to RegulonDB TSV format."""
    
    # Header from parsers.py
    header_lines = [
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
    
    rows = []
    for u, v, data in graph.edges(data=True):
        row = [
            data.get('regulatorId', f"ID_{u}"),
            data.get('regulatorName', u),
            data.get('regulatorGeneName', u),
            data.get('regulatedId', f"ID_{v}"),
            data.get('regulatedName', v),
            data.get('function', '+'),
            data.get('confidenceLevel', '?')
        ]
        rows.append('\t'.join(str(x) for x in row))
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(header_lines + rows))
