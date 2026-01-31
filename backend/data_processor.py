"""
Data Processing Pipeline for ArXiv Papers
Handles: Data ingestion, chunking, embedding generation, and indexing
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens


class DataProcessor:
    def __init__(self):
        self.data_path = Path(DATA_PATH)
        self.chroma_path = Path(CHROMA_PATH)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections
        self.papers_collection = None
        self.chunks_collection = None
        
    def download_dataset(self):
        """Download ArXiv dataset from Kaggle"""
        print("📥 Downloading ArXiv dataset from Kaggle...")
        
        # Create data directory
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Set Kaggle credentials
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        
        if not kaggle_username or not kaggle_key:
            print("⚠️  Kaggle credentials not found in .env file")
            print("Please download manually from: https://www.kaggle.com/datasets/neelshah18/arxivdataset")
            print(f"Extract to: {self.data_path}")
            return False
        
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key
        
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                "neelshah18/arxivdataset",
                path=str(self.data_path),
                unzip=True
            )
            print("✅ Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/neelshah18/arxivdataset")
            return False
    
    def load_papers(self) -> pd.DataFrame:
        """Load papers from JSON file"""
        print("📂 Loading papers data...")
        
        # Look for the data file
        json_files = list(self.data_path.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.data_path}")
            print("Please ensure the dataset is downloaded and extracted.")
            return pd.DataFrame()
        
        # Load the JSON file (ArXiv dataset is typically in JSON Lines format)
        papers = []
        data_file = json_files[0]
        
        print(f"📖 Reading {data_file.name}...")
        
        try:
            # Try reading as JSON Lines
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading papers"):
                    try:
                        paper = json.loads(line.strip())
                        papers.append(paper)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            # Try reading as regular JSON
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
            except Exception as e2:
                print(f"❌ Error reading data file: {e2}")
                return pd.DataFrame()
        
        df = pd.DataFrame(papers)
        print(f"✅ Loaded {len(df)} papers")
        return df
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks based on token count"""
        if not text:
            return []
        
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
        
        return chunks
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation references from text"""
        citations = []
        
        # Pattern for author-year citations like [Smith 2020] or (Smith et al., 2020)
        patterns = [
            r'\[([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4})\]',
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4})\)',
            r'\b([A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\(\d{4}\))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))
    
    def process_papers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process papers: clean, chunk, and extract metadata"""
        print("🔄 Processing papers...")
        
        processed_papers = []
        all_chunks = []
        
        # Determine column names (ArXiv dataset may have different naming)
        title_col = 'title' if 'title' in df.columns else df.columns[0]
        abstract_col = 'abstract' if 'abstract' in df.columns else 'summary'
        
        # Get available columns for metadata
        available_cols = df.columns.tolist()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            paper_id = str(idx)
            
            # Extract fields safely
            title = str(row.get(title_col, '')) if title_col in available_cols else ''
            abstract = str(row.get(abstract_col, '')) if abstract_col in available_cols else ''
            
            # Get additional metadata if available
            authors = row.get('author', row.get('authors', ''))
            if isinstance(authors, list):
                authors = ', '.join(authors)
            
            categories = row.get('categories', row.get('category', ''))
            year = row.get('year', row.get('update_date', ''))[:4] if row.get('year', row.get('update_date', '')) else ''
            
            # Preprocess
            title = self.preprocess_text(title)
            abstract = self.preprocess_text(abstract)
            
            # Combine title and abstract for full text
            full_text = f"{title}. {abstract}"
            
            # Extract citations
            citations = self.extract_citations(abstract)
            
            # Create paper record
            paper = {
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'authors': str(authors),
                'categories': str(categories),
                'year': str(year),
                'citations': citations,
                'citation_count': len(citations)
            }
            processed_papers.append(paper)
            
            # Create chunks
            chunks = self.chunk_text(full_text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{paper_id}_chunk_{chunk_idx}",
                    'paper_id': paper_id,
                    'text': chunk,
                    'chunk_index': chunk_idx,
                    'title': title,
                    'authors': str(authors),
                    'year': str(year)
                })
        
        print(f"✅ Processed {len(processed_papers)} papers into {len(all_chunks)} chunks")
        
        return {
            'papers': processed_papers,
            'chunks': all_chunks
        }
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for texts in batches"""
        print("🧠 Generating embeddings...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def build_index(self, processed_data: Dict[str, Any]):
        """Build ChromaDB index from processed data"""
        print("📊 Building search index...")
        
        papers = processed_data['papers']
        chunks = processed_data['chunks']
        
        # Create or get collections
        try:
            self.chroma_client.delete_collection("papers")
        except:
            pass
        try:
            self.chroma_client.delete_collection("chunks")
        except:
            pass
        
        self.papers_collection = self.chroma_client.create_collection(
            name="papers",
            metadata={"description": "Research papers metadata"}
        )
        
        self.chunks_collection = self.chroma_client.create_collection(
            name="chunks",
            metadata={"description": "Paper chunks for semantic search"}
        )
        
        # Index papers
        print("📄 Indexing papers...")
        paper_texts = [f"{p['title']}. {p['abstract']}" for p in papers]
        paper_embeddings = self.generate_embeddings(paper_texts)
        
        # Add papers in batches
        batch_size = 1000
        for i in tqdm(range(0, len(papers), batch_size), desc="Adding papers to index"):
            batch_papers = papers[i:i + batch_size]
            batch_embeddings = paper_embeddings[i:i + batch_size].tolist()
            
            self.papers_collection.add(
                ids=[p['id'] for p in batch_papers],
                embeddings=batch_embeddings,
                metadatas=[{
                    'title': p['title'],
                    'authors': p['authors'],
                    'year': p['year'],
                    'categories': p['categories'],
                    'citation_count': p['citation_count']
                } for p in batch_papers],
                documents=[p['abstract'] for p in batch_papers]
            )
        
        # Index chunks
        print("📝 Indexing chunks...")
        chunk_texts = [c['text'] for c in chunks]
        chunk_embeddings = self.generate_embeddings(chunk_texts)
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks to index"):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = chunk_embeddings[i:i + batch_size].tolist()
            
            self.chunks_collection.add(
                ids=[c['id'] for c in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[{
                    'paper_id': c['paper_id'],
                    'title': c['title'],
                    'authors': c['authors'],
                    'year': c['year'],
                    'chunk_index': c['chunk_index']
                } for c in batch_chunks],
                documents=[c['text'] for c in batch_chunks]
            )
        
        print(f"✅ Index built successfully!")
        print(f"   Papers indexed: {self.papers_collection.count()}")
        print(f"   Chunks indexed: {self.chunks_collection.count()}")
    
    def save_metadata(self, processed_data: Dict[str, Any]):
        """Save processed metadata to JSON"""
        metadata_path = self.data_path / "processed_metadata.json"
        
        # Save paper metadata (without embeddings)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_papers': len(processed_data['papers']),
                'total_chunks': len(processed_data['chunks']),
                'papers': processed_data['papers']
            }, f, indent=2)
        
        print(f"✅ Metadata saved to {metadata_path}")
    
    def run_pipeline(self):
        """Run the complete data processing pipeline"""
        print("=" * 60)
        print("🚀 Starting Data Processing Pipeline")
        print("=" * 60)
        
        # Step 1: Download dataset (if not present)
        if not list(self.data_path.glob("*.json")):
            self.download_dataset()
        
        # Step 2: Load papers
        df = self.load_papers()
        if df.empty:
            print("❌ No data to process. Please download the dataset first.")
            return
        
        # Step 3: Process papers
        processed_data = self.process_papers(df)
        
        # Step 4: Build index
        self.build_index(processed_data)
        
        # Step 5: Save metadata
        self.save_metadata(processed_data)
        
        print("=" * 60)
        print("✅ Pipeline completed successfully!")
        print(f"   Indexed {len(processed_data['papers'])} papers")
        print("=" * 60)


if __name__ == "__main__":
    processor = DataProcessor()
    processor.run_pipeline()
