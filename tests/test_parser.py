#!/usr/bin/env python3
"""
Test script for the updated RegulonDB parsers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.parsers import parse_tf_gene_network, parse_gene_product_mapping

def test_network_parser():
    print("=== Testing NetworkRegulatorGene.tsv Parser ===")
    
    try:
        graph, metadata = parse_tf_gene_network('NetworkRegulatorGene.tsv')
        print(f"✓ Successfully parsed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Show first few edges with proper data
        print("\nFirst 5 valid edges:")
        count = 0
        for tf, gene, attrs in graph.edges(data=True):
            if count >= 5:
                break
            if attrs.get('tf_name') and attrs.get('gene_name'):  # Only show edges with valid data
                print(f"  {attrs['tf_name']} -> {attrs['gene_name']}: {attrs['effect']} ({attrs['evidence_type']})")
                count += 1
        
        # Show some statistics
        evidence_counts = {}
        effect_counts = {}
        for _, _, attrs in graph.edges(data=True):
            evidence = attrs.get('evidence_type', 'Unknown')
            effect = attrs.get('effect', '?')
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
            effect_counts[effect] = effect_counts.get(effect, 0) + 1
        
        print(f"\nEvidence distribution: {evidence_counts}")
        print(f"Effect distribution: {effect_counts}")
        
    except Exception as e:
        print(f"✗ Network parser error: {e}")
        import traceback
        traceback.print_exc()


def test_gene_product_parser():
    print("\n=== Testing GeneProductAllIdentifiersSet.tsv Parser ===")
    
    try:
        name_to_b, b_to_name, reg_to_b = parse_gene_product_mapping('GeneProductAllIdentifiersSet.tsv')
        print(f"✓ Successfully parsed:")
        print(f"  - {len(name_to_b)} name->b-number mappings")
        print(f"  - {len(b_to_name)} b-number->name mappings") 
        print(f"  - {len(reg_to_b)} RegulonDB->b-number mappings")
        
        # Show first few mappings
        print("\nFirst 5 name->b-number mappings:")
        for i, (name, bnumber) in enumerate(name_to_b.items()):
            if i >= 5: break
            print(f"  {name} -> {bnumber}")
        
        print("\nFirst 5 RegulonDB->b-number mappings:")
        for i, (reg_id, bnumber) in enumerate(reg_to_b.items()):
            if i >= 5: break
            print(f"  {reg_id} -> {bnumber}")
            
        # Test some specific lookups
        test_names = ['alr', 'modb', 'cysz', 'fnr', 'crp']
        print(f"\nTesting specific gene lookups:")
        for name in test_names:
            bnumber = name_to_b.get(name, 'NOT_FOUND')
            print(f"  {name} -> {bnumber}")
        
    except Exception as e:
        print(f"✗ Gene product parser error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_network_parser()
    test_gene_product_parser()
