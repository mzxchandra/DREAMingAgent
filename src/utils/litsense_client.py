"""
LitSense API Client for retrieving biomedical literature sentences.
"""

import time
import requests
from typing import List, Dict, Any, Optional
from loguru import logger

class LitSenseClient:
    """Client for NCBI LitSense API."""
    
    BASE_URL = "https://www.ncbi.nlm.nih.gov/research/litsense-api/api/"
    
    def __init__(self, user_agent: str = "DREAMingAgent/1.0 (research_tool)"):
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "application/json"
        }
        self.last_request_time = 0.0
        # Rate limit: 2 requests per second (aggressive but needed for bulk)
        self.rate_limit_delay = 0.5

    def search(self, query: str, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Search LitSense for sentences matching the query.
        
        Args:
            query: Search query (e.g., "FNR regulation of arcA")
            rerank: Whether to use neural reranking (recommended)
            
        Returns:
            List of result dictionaries. Each result typically contains:
            - text: The sentence text
            - pmid: PubMed ID
            - pmcid: PubMed Central ID
            - score: Relevance score
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
            
        params = {
            "query": query,
            "rerank": "true" if rerank else "false"
        }
        
        try:
            # Note: Removed trailing slash before ? just in case, though requests handles it
            url = self.BASE_URL
            
            logger.debug(f"Querying LitSense: {query}")
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                results = response.json()
                logger.debug(f"LitSense found {len(results)} results for '{query}'")
                return results
            else:
                logger.warning(f"LitSense API error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"LitSense request failed: {e}")
            return []

    def get_evidence_for_interaction(self, tf_name: str, target_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Convenience method to search for regulation evidence.
        Tries a few query variations to maximize recall.
        
        Args:
            tf_name: Name of transcription factor
            target_name: Name of target gene
            limit: Maximum results to return
        """
        queries = [
            f"{tf_name} regulates {target_name}",
            f"{tf_name} binding {target_name}",
            f"{tf_name} activation {target_name}",
            f"{tf_name} repression {target_name}"
        ]
        
        all_results = []
        seen_extracts = set()
        
        for q in queries[:2]: # limit to first 2 to save time/calls during testing
            results = self.search(q)
            for res in results:
                text = res.get('text', '')
                if text and text not in seen_extracts:
                    all_results.append(res)
                    seen_extracts.add(text)
            
            if len(all_results) >= limit: # Stop if we have enough
                break
                
        return all_results[:limit]

    def get_tf_context(self, tf_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for general regulatory context for a Transcription Factor.
        
        Args:
            tf_name: Name of transcription factor
            limit: Maximum results to return
        """
        queries = [
            f"{tf_name} regulation",
            f"{tf_name} binding site",
            f"{tf_name} transcription factor"
        ]
        
        all_results = []
        seen_extracts = set()
        
        for q in queries:
            results = self.search(q)
            for res in results:
                text = res.get('text', '')
                if text and text not in seen_extracts:
                    all_results.append(res)
                    seen_extracts.add(text)
            
            if len(all_results) >= limit:
                break
                
        return all_results[:limit]
