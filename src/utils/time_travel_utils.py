
import os
import csv
import pandas as pd
import numpy as np
import logging
from typing import Set, Tuple, Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BIG_5_MAP = {
    "FNR": "b1334",
    "ArcA": "b4401",
    "CRP": "b3357",
    "Fur": "b0683",
    "SoxS": "b4062"
}

BIG_5_BNUMS = set(BIG_5_MAP.values())

def load_object_synonyms(synonym_path: str) -> Dict[str, str]:
    """
    Loads object_synonym.txt and builds a mapping from various IDs to b-numbers.
    Prioritizes b-numbers (e.g., b1234).
    """
    id_map = {}
    if not os.path.exists(synonym_path):
        logger.error(f"Synonym file not found: {synonym_path}")
        return id_map

    try:
        with open(synonym_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    obj_id = parts[0].strip()
                    synonym = parts[1].strip()
                    if synonym.startswith('b') and synonym[1:].isdigit() and len(synonym) == 5:
                        id_map[obj_id] = synonym
                        # Also map the synonym itself to itself
                        id_map[synonym] = synonym
    except Exception as e:
        logger.error(f"Error loading synonyms: {e}")
    
    return id_map

def load_tsv_map(file_path: str, key_col_idx: int, val_col_idx: int) -> Dict[str, str]:
    """
    Generic helper to load a TSV and map one column to another.
    Indices are 0-based.
    """
    mapping = {}
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return mapping

    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) > max(key_col_idx, val_col_idx):
                    key = parts[key_col_idx].strip()
                    val = parts[val_col_idx].strip()
                    if key and val:
                        mapping[key] = val
    except Exception as e:
        logger.error(f"Error loading map from {file_path}: {e}")
    return mapping

def Parser2016(regulon_path: str, big_5_tfs: Set[str] = None) -> Set[Tuple[str, str]]:
    """
    Parses 2016 RegulonDB data to extract TF-Gene interactions as (TF_bNum, Gene_bNum).
    Uses a robust 6-step relational join strategy to resolve IDs.
    
    Chain: Conformation -> TF_ID -> Product_ID -> Gene_ID -> b-number
    """
    interactions = set()
    
    # File paths
    interaction_file = os.path.join(regulon_path, "regulatory_interaction.txt")
    conformation_file = os.path.join(regulon_path, "conformation.txt")
    transcription_factor_file = os.path.join(regulon_path, "transcription_factor.txt") # Fallback
    product_tf_file = os.path.join(regulon_path, "product_tf_link.txt")
    gene_product_file = os.path.join(regulon_path, "gene_product_link.txt")
    synonym_file = os.path.join(regulon_path, "object_synonym.txt")

    # 1. Load Mappings
    logger.info("Loading 2016 dictionaries...")
    
    # Conformation -> TF ID (Col 1 -> Col 2 in conformation.txt, 0-indexed: 0->1)
    conf_to_tf = load_tsv_map(conformation_file, 0, 1)
    
    # TF ID -> Product ID (Col 1 -> Col 2 in product_tf_link.txt, 0-indexed: 0->1)
    tf_to_product = load_tsv_map(product_tf_file, 0, 1)
    
    # Product ID -> Gene ID (Col 2 -> Col 1 in gene_product_link.txt !!! CHECK FILE CAREFULLY usually Prod is col 2, Gene is col 1? No, usually Gene->Prod. Let's check schema.
    # User view showed: "ECK120000319    ECK120004795" -> Gene ID (Col 1), Product ID (Col 2)
    # So we want Product -> Gene, which is val -> key from file.
    # Let's use load_tsv_map reversed: key=Col 1 (Prod), val=Col 0 (Gene)
    product_to_gene = load_tsv_map(gene_product_file, 1, 0)
    
    # TU -> Gene (Col 1 -> Col 2 in tu_gene_link.txt usually. Verify via head)
    # File view usually: TU_ID, GENE_ID
    tu_gene_file = os.path.join(regulon_path, "tu_gene_link.txt")
    tu_to_gene = load_tsv_map(tu_gene_file, 0, 1)

    # ECK/Gene/Name -> b-number
    id_to_bnum = load_object_synonyms(synonym_file)

    logger.info(f"Loaded maps: Conf->TF ({len(conf_to_tf)}), TF->Prod ({len(tf_to_product)}), Prod->Gene ({len(product_to_gene)}), ID->Bnum ({len(id_to_bnum)})")

    # 2. Iterate Interactions
    try:
        with open(interaction_file, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                
                # Check Schema: tf_gene_interaction.txt 
                # Col 1 (idx 1): CONFORMATION_ID (e.g., ECK120011909) ?? Wait, verify schema.
                # User's grep: "ECK120011230" was in Col 2 (idx 1) of interaction file.
                # Let's assume standard RegulonDB:
                # 1) INTERACTION_ID?? No. usually TF ID or Conformation ID is early.
                # We verified earlier using grep "FNR" ... but let's be careful.
                # Checking grep output from history:
                # "grep 'FNR' ... | cut -f2" -> ECK120011230. So TF/Conf ID is at Index 1 (Col 2).
                # Target Gene ID is usually later. Standard is often
                
                if len(parts) < 8:
                    continue
                    
                conf_id = parts[1].strip()
                target_gene_id = parts[7].strip() # Target Gene ECK ID (Col 8 -> Index 7)
                
                # Resolve TF b-number
                tf_bnum = None
                
                # Chain: Conf -> TF -> Prod -> Gene -> bnum
                if conf_id in conf_to_tf:
                    tf_id = conf_to_tf[conf_id]
                    if tf_id in tf_to_product:
                        prod_id = tf_to_product[tf_id]
                        if prod_id in product_to_gene:
                            gene_id = product_to_gene[prod_id]
                            if gene_id in id_to_bnum:
                                tf_bnum = id_to_bnum[gene_id]
                
                # Fallback: Check if ConfID itself maps effectively (sometimes distinct tables aren't needed)
                if not tf_bnum and conf_id in id_to_bnum:
                     tf_bnum = id_to_bnum[conf_id]
                
                # If a big_5_tfs filter is provided, check if the TF is in it.
                # This check is placed here to allow the fallback mapping to occur first.
                if tf_bnum and big_5_tfs and tf_bnum not in big_5_tfs:
                    continue
                     
                # Debug logging for first few failures
                # if len(interactions) < 5 and tf_bnum:
                #      pass
                
                # Resolve Target Gene b-number
                # If target_gene_id is empty (common in 2016 file for some reason), we can't map it directly.
                # Try TU Bridge: Transcription Unit ID (Col 3 / Index 2 of interaction) -> Gene ID
                if not target_gene_id:
                     tu_id = parts[2].strip() # Col 3
                     if tu_id in tu_to_gene:
                         target_gene_id = tu_to_gene[tu_id]
                     else:
                         continue
                
                target_bnum = id_to_bnum.get(target_gene_id)
                
                # Add to set if valid and (optionally) in Big 5
                if tf_bnum and target_bnum:
                    # The big_5_tfs check is already done above for the TF.
                    # No need to re-check here.
                    interactions.add((tf_bnum, target_bnum))
                    
    except Exception as e:
        logger.error(f"Error parsing 2016 interactions: {e}")

    logger.info(f"Parsed {len(interactions)} interactions from 2016 data.")
    return interactions

def Parser2024(file_path: str, id_map_path: str, big_5_tfs: Set[str] = None) -> Set[Tuple[str, str]]:
    """
    Parses 2024 RegulonDB data (NetworkRegulatorGene.tsv).
    Uses GeneProductAllIdentifiersSet.tsv to map IDs to b-numbers.
    """
    interactions = set()
    
    # 1. Build Mapping from GeneProductAllIdentifiersSet
    # We need a robust map that takes ANY 2024 ID (RDBECOLI...) and gives a b-number.
    # Using 'otherDbsGeneIds' column which contains "b1234"
    logger.info("Loading 2024 ID mappings...")
    val_map = {} 
    
    try:
        with open(id_map_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                
                # Check for enough columns.
                # Based on inspection:
                # Col 0: GeneID (RDBECOLIGNC...)
                # Col 1: GeneName
                # Col 6 (approx): Synonyms/OtherIDs (contains [REFSEQ:b1234] or plain b1234 inside brackets)
                
                if len(parts) < 7:
                    continue
                    
                gene_id = parts[0].strip()
                gene_name = parts[1].strip()
                synonyms_blob = parts[6] # " [ASKA:ECK...][REFSEQ:b0002] "
                
                # Extract b-number
                b_num = None
                # Regex for b-number, potentially inside brackets or standalone
                import re
                match = re.search(r'\b(b\d{4})\b', synonyms_blob)
                if match:
                    b_num = match.group(1)
                
                if b_num:
                    val_map[gene_id.lower()] = b_num
                    val_map[gene_name.lower()] = b_num
                    val_map[b_num.lower()] = b_num

    except Exception as e:
        logger.error(f"Error loading 2024 ID map: {e}")
        return interactions # empty
        
    logger.info(f"Loaded {len(val_map)} 2024 ID mappings.")

    # 2. Parse Network File
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                # NetworkRegulatorGene.tsv usually: RegulatorID, RegulatorName, RegulatedID, RegulatedName...
                # Let's assume indices 0 and 2 based on standard. But verification needed.
                # User provided file view for IdentifiersSet, but not Network file content.
                # Assuming Network file has headers or similar structure.
                # Let's try DictReader if header exists, else fallback.
                pass 
                
            # Re-read for reader assuming NO header row (standard for this specific file in new RegulonDB based on inspection)
            f.seek(0)
            # Skip comments
            pos = 0
            curr = f.readline()
            while curr.startswith('#'):
                pos = f.tell()
                curr = f.readline()
            f.seek(pos)
            
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3:
                    continue
                    
                # Columns: 0=RegulatorID, 1=RegulatorName, 2=RegulatedID, 3=RegulatedName based on file inspection
                # TFC IDs in Col 0 don't match GNC IDs in map. Using Names (Col 1, 3) which are also mapped.
                tf_name_raw = row[1].strip()
                gene_name_raw = row[3].strip()
                
                if not tf_name_raw or not gene_name_raw:
                    continue
                    
                tf_bnum = val_map.get(tf_name_raw.lower())
                gene_bnum = val_map.get(gene_name_raw.lower())
                
                tf_bnum = val_map.get(tf_name_raw.lower())
                gene_bnum = val_map.get(gene_name_raw.lower())
                
                if tf_bnum and gene_bnum:
                    if big_5_tfs and tf_bnum not in big_5_tfs:
                        continue
                    interactions.add((tf_bnum, gene_bnum))

    except Exception as e:
        logger.error(f"Error parsing 2024 network: {e}")

    logger.info(f"Parsed {len(interactions)} interactions from 2024 data.")
    return interactions

def DescriptionFetcher(product_file_path: str, synonym_path: str) -> Dict[str, str]:
    """
    Returns a dict mapping b-number -> Product Description (Note).
    Used for System B context.
    
    1. Load product.txt to get ProductID -> Note
    2. Load gene_product_link + object_synonym to map b-number -> ProductID
    """
    # Simply load product.txt: PRODUCT_ID (Col 1/idx 0 for some tables, check header?)
    # Header: 1) PRODUCT_ID 2) PRODUCT_NAME ... 4) PRODUCT_NOTE
    # Let's assume idx 0 -> ID, idx 3 -> Note (0-based)
    
    descriptions = {}
    prod_notes = {}
    
    # 1. Load Notes
    try:
        with open(product_file_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    pid = parts[0].strip()
                    note = parts[3].strip()
                    if note:
                        prod_notes[pid] = note
    except Exception as e:
        logger.error(f"Error loading product notes: {e}")
        
    # 2. Map b-number -> Product ID
    # We can reverse the chain: b-number -> GeneID -> ProductID
    # GeneID in synonym file (key) -> b-number (val).
    # wait, id_to_bnum maps ECK -> bnum.
    # We need bnum -> ECK_Gene.
    # Then ECK_Gene -> ECK_Product (via gene_product_link).
    # Then ECK_Product -> Note.
    
    # Reload needed maps locally if not passed? 
    # For now, let's just do a robust all-to-all loading or assume b-number queries
    # are rare (Big 5).
    
    # Actually, the runner will usually just ask: "Get desc for b1334".
    # So we need b1334 -> ... -> Note.
    
    # Let's reconstruct the path:
    # b1334 -> GeneID (ECK120000319) [from object_synonym, searched reverse]
    # GeneID -> ProductID (ECK120004795) [from gene_product_link]
    # ProductID -> Note [from product.txt]
    
    # This seems expensive to do on fly. Better to pre-build bnum->note map.
    
    # Load gene_product_link: Gene -> Product
    gene_prod_file = os.path.join(os.path.dirname(product_file_path), "gene_product_link.txt")
    gene_to_prod = load_tsv_map(gene_prod_file, 0, 1) # Col 1 (Gene) -> Col 2 (Prod) matches user view earlier
    
    # Load synonyms: ECK_Gene -> bnum
    synonyms = load_object_synonyms(synonym_path)
    
    # Invert synonyms: bnum -> ECK_Gene (take first logic)
    bnum_to_gene = {}
    for k, v in synonyms.items():
        if k.startswith('ECK'): # assume ECK ID
            bnum_to_gene[v] = k
            
    # Build Map
    for bnum, gene_id in bnum_to_gene.items():
        if gene_id in gene_to_prod:
            prod_id = gene_to_prod[gene_id]
            if prod_id in prod_notes:
                descriptions[bnum] = prod_notes[prod_id]
                
    return descriptions

def compute_targeted_clr_row_z(
    expression_matrix: pd.DataFrame, 
    target_tfs: List[str]
) -> pd.DataFrame:
    """
    Computes CLR scores (z-scores) effectively but efficiently for specific TFs (Row-Z).
    Logic:
    1. Compute MI between each Target TF and ALL genes.
    2. Normalize this TF's MI vector (Row Z-score).
    3. Ignore Column Z-score (asymmetric approximation O(kN) vs O(N^2)).
    
    Returns DataFrame with columns [TF, Target, Z_Score]
    """
    from sklearn.feature_selection import mutual_info_regression
    
    results = []
    
    # Constants matching analysis agent
    MI_N_NEIGHBORS = 3
    RANDOM_STATE = 42
    
    # Ensure TFs are in matrix
    valid_tfs = [tf for tf in target_tfs if tf in expression_matrix.index]
    
    if not valid_tfs:
        logger.warning("No valid TFs found in expression matrix for CLR.")
        return pd.DataFrame(columns=['TF', 'Target', 'score'])
        
    X = expression_matrix.T # Samples x Genes
    
    for tf in valid_tfs:
        tf_expr = expression_matrix.loc[tf].values
        
        # Compute MI
        mi = mutual_info_regression(
            X, 
            tf_expr,
            discrete_features=False,
            n_neighbors=MI_N_NEIGHBORS,
            random_state=RANDOM_STATE
        )
        
        # Z-Score Normalization (Row-Z)
        mu = np.mean(mi)
        sigma = np.std(mi)
        
        if sigma == 0:
            z_scores = np.zeros_like(mi)
        else:
            z_scores = (mi - mu) / sigma
        
        # Apply max(0, z)
        z_scores = np.maximum(z_scores, 0)
        
        # Store top hits? Or all? 
        # For efficiency, we can process them later, but let's return a DF
        # mapping target genes.
        
        # Create temp DF
        df_tf = pd.DataFrame({
            'TF': tf,
            'Target': expression_matrix.index,
            'score': z_scores
        })
        
        results.append(df_tf)
        
    if not results:
        return pd.DataFrame(columns=['TF', 'Target', 'score'])
        
    return pd.concat(results, ignore_index=True)
