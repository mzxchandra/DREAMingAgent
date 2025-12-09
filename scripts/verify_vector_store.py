"""
Verify vector store contains ingested documents.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.vector_store import get_vector_store

def main():
    print("Initializing Vector Store...")
    vs = get_vector_store(persist_directory="chroma_db")
    
    count = vs.count()
    print(f"Total documents: {count}")
    
    if count > 0:
        query = "ppgpp regulation"
        print(f"\nQuerying for: '{query}'...")
        results = vs.query(query, n_results=1)
        
        if results:
            print("\n[Top Result]")
            print(f"Source: {results[0].source}")
            print(f"Score: {results[0].similarity_score:.4f}")
            print(f"Text: {results[0].text[:200]}...")
        else:
            print("No results found.")
    else:
        print("Vector store is empty.")

if __name__ == "__main__":
    main()
