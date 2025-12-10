import os
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr
from loguru import logger
from typing import Tuple, Optional, Dict, Any

from src.llm_config import create_argonne_llm

class PathwayAssessor:
    def __init__(self, regulondb_path: str = "data/NetworkRegulatorGene.tsv", m3d_path: str = "data/E_coli_v4_Build_6_exps.tab"):
        """
        Initialize the Pathway Assessor with data paths.
        """
        self.regulondb_path = regulondb_path
        self.m3d_path = m3d_path
        self.regulondb = None
        self.m3d_data = None
        self.gene_map = {} # Maps gene name -> M3D index name
        
        self._load_data()

    def _load_data(self):
        """Loads and processes RegulonDB and M3D data."""
        logger.info("Loading Pathway Assessor data...")
        
        # Load RegulonDB
        try:
            # We explicitly define columns based on file inspection
            cols = ['regulatorId', 'regulatorName', 'regulatorGeneName', 'regulatedId', 'regulatedName', 'function', 'confidenceLevel']
            
            # Read, skipping comments. Note: checking for the '1)' line manually after load
            # or we can rely on pandas identifying comments if they start with #
            df = pd.read_csv(self.regulondb_path, sep='\t', comment='#', names=cols, header=None, on_bad_lines='skip')
            
            # Filter out the weird header line if it persists ("1)regulatorId"...)
            # It seems the comment char handles the preamble, but let's be safe
            df = df[~df['regulatorId'].astype(str).str.startswith('1)')]
            
            self.regulondb = df
            logger.info(f"Loaded RegulonDB: {len(self.regulondb)} interactions")
        except Exception as e:
            logger.error(f"Failed to load RegulonDB: {e}")
            raise

        # Load M3D
        try:
            # M3D has gene IDs as the first column (index). 
            # We assume the first row contains headers for conditions, but the first column is the index.
            self.m3d_data = pd.read_csv(self.m3d_path, sep='\t', index_col=0)
            
            # Create gene name mapping
            # Index is like "aaeR_b3243_15". We want "aaeR".
            for idx in self.m3d_data.index:
                idx_str = str(idx)
                parts = idx_str.split('_')
                if parts:
                    gene_name = parts[0]
                    self.gene_map[gene_name.lower()] = idx_str
            
            logger.info(f"Loaded M3D Data: {self.m3d_data.shape} - Mapped {len(self.gene_map)} genes")
        except Exception as e:
            logger.error(f"Failed to load M3D data: {e}")
            raise

    def _parse_genes(self, pathway: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts gene names from the pathway string.
        Assumes format like "Regulator -> Target" or just mentions two genes.
        Simple heuristic: find words that map to our gene database.
        """
        # Normalize and split
        words = pathway.replace('-', ' ').replace('>', ' ').replace(',', ' ').split()
        found_genes = []
        
        # Prioritize finding known genes
        for word in words:
            clean_word = word.strip().lower()
            if clean_word in self.gene_map:
                found_genes.append(clean_word)
        
        # We need exactly two unique genes for a simple link
        # If order matters (Regulator -> Target), we assume the order they appear
        # matches the description approximately.
        
        # Filter duplicates while preserving order
        unique_genes = []
        seen = set()
        for g in found_genes:
            if g not in seen:
                unique_genes.append(g)
                seen.add(g)
                
        if len(unique_genes) >= 2:
            return unique_genes[0], unique_genes[1]
        
        return None, None

    def assess_pathway(self, pathway_description: str) -> str:
        """
        Assesses if the connection is WEAK or STRONG.
        """
        regulator, target = self._parse_genes(pathway_description)
        
        if not regulator or not target:
            logger.warning(f"Could not parse two genes from: {pathway_description}")
            return "weak" # Default fail safe, or could return "error"
            
        logger.info(f"Assessing: {regulator} -> {target}")
        
        # 1. Check RegulonDB
        # Case insensitive match
        reg_df = self.regulondb[
            (self.regulondb['regulatorName'].str.lower() == regulator) &
            (self.regulondb['regulatedName'].str.lower() == target)
        ]
        
        regulondb_info = "No direct evidence found."
        if not reg_df.empty:
            row = reg_df.iloc[0]
            regulondb_info = f"Found interaction. Function: {row['function']}, Confidence: {row['confidenceLevel']}"
        
        # 2. Check M3D Correlation
        correlation_info = "Data not available."
        corr_val = 0.0
        
        m3d_reg = self.gene_map.get(regulator)
        m3d_tgt = self.gene_map.get(target)
        
        if m3d_reg and m3d_tgt:
            try:
                vec_a = self.m3d_data.loc[m3d_reg]
                vec_b = self.m3d_data.loc[m3d_tgt]
                
                # Check for NaNs
                valid_mask = ~np.isnan(vec_a) & ~np.isnan(vec_b)
                if valid_mask.sum() > 2:
                    corr, _ = pearsonr(vec_a[valid_mask], vec_b[valid_mask])
                    corr_val = corr
                    correlation_info = f"{corr:.3f}"
                else:
                    correlation_info = "Insufficient valid data points."
            except Exception as e:
                logger.warning(f"Error calculating correlation: {e}")
                correlation_info = "Calculation error."
        
        # 3. LLM Decision
        prompt = f"""
        You are an expert biologist. Evaluate the strength of this gene regulation pathway:
        "{regulator} -> {target}"
        
        Evidence:
        1. RegulonDB (Gold Standard): {regulondb_info}
        2. M3D Expression Data (Pearson Correlation): {correlation_info}
        
        Task:
        Determine if this connection is "weak" or "strong".
        - "strong": High correlation (>0.7) OR confirmed by RegulonDB with high confidence.
        
        RegulonDB Confidence Codes:
        - C: Confirmed (High Confidence) -> Strong evidence
        - S: Strong (High Confidence) -> Strong evidence
        - W: Weak -> Weak evidence
        
        - "weak": Low/Negative correlation AND no RegulonDB evidence, or expressly explicitly weak evidence.
        
        Return Valid JSON only: {{ "result": "strong" | "weak", "reasoning": "short explanation" }}
        """
        
        try:
            llm = create_argonne_llm()
            # Invoke the LLM
            response_msg = llm.invoke(prompt)
            content = response_msg.content.strip()
            logger.info(f"LLM Raw Content: {content}")
            
            # Basic JSON parsing
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content.strip())
            if "result" in data:
                return data["result"].lower()
                
        except Exception as e:
            logger.error(f"Argonne LLM call failed: {e}")
        
        logger.info(f"Fallback active. Corr: {corr_val} RegDB: {not reg_df.empty}")
        if not reg_df.empty and abs(corr_val) > 0.3:
            return "strong"
        if abs(corr_val) > 0.7:
            return "strong"
            
        return "weak"
