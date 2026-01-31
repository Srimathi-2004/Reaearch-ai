"""
Search Engine with LLM-Powered Synthesis
Handles: Semantic search, cross-document analysis, and response generation
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import json
from collections import Counter
import re

load_dotenv()

# Configuration
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SearchEngine:
    def __init__(self):
        self.chroma_path = Path(CHROMA_PATH)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.llm_client = None
            print("⚠️  OpenAI API key not found. LLM features will be limited.")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collections
        try:
            self.papers_collection = self.chroma_client.get_collection("papers")
            self.chunks_collection = self.chroma_client.get_collection("chunks")
            print(f"✅ Loaded {self.papers_collection.count()} papers and {self.chunks_collection.count()} chunks")
        except Exception as e:
            print(f"⚠️  Collections not found. Please run data processing first. Error: {e}")
            self.papers_collection = None
            self.chunks_collection = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        if not self.papers_collection:
            return {"papers_count": 0, "chunks_count": 0, "status": "not_indexed"}
        
        return {
            "papers_count": self.papers_collection.count(),
            "chunks_count": self.chunks_collection.count(),
            "status": "ready"
        }
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 20,
        search_type: str = "chunks"
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on papers or chunks"""
        
        collection = self.chunks_collection if search_type == "chunks" else self.papers_collection
        
        if not collection:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def aggregate_by_paper(self, chunk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate chunk results by paper"""
        paper_map = {}
        
        for chunk in chunk_results:
            paper_id = chunk['metadata'].get('paper_id', chunk['id'].split('_chunk_')[0])
            
            if paper_id not in paper_map:
                paper_map[paper_id] = {
                    'paper_id': paper_id,
                    'title': chunk['metadata'].get('title', ''),
                    'authors': chunk['metadata'].get('authors', ''),
                    'year': chunk['metadata'].get('year', ''),
                    'chunks': [],
                    'max_score': 0
                }
            
            paper_map[paper_id]['chunks'].append({
                'text': chunk['text'],
                'score': chunk['score'],
                'chunk_index': chunk['metadata'].get('chunk_index', 0)
            })
            paper_map[paper_id]['max_score'] = max(
                paper_map[paper_id]['max_score'],
                chunk['score']
            )
        
        # Sort by max score
        papers = list(paper_map.values())
        papers.sort(key=lambda x: x['max_score'], reverse=True)
        
        return papers
    
    def synthesize_response(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """Generate LLM-powered synthesis from search results"""
        
        if not self.llm_client:
            return self._fallback_synthesis(query, search_results)
        
        # Aggregate results by paper
        papers = self.aggregate_by_paper(search_results)
        
        # Build context from top results
        context_parts = []
        total_length = 0
        papers_used = []
        
        for paper in papers[:15]:  # Top 15 papers
            paper_context = f"Paper: {paper['title']}"
            if paper['authors']:
                paper_context += f" by {paper['authors']}"
            if paper['year']:
                paper_context += f" ({paper['year']})"
            paper_context += "\n"
            
            # Add best chunks
            sorted_chunks = sorted(paper['chunks'], key=lambda x: x['score'], reverse=True)
            for chunk in sorted_chunks[:2]:  # Top 2 chunks per paper
                paper_context += f"Content: {chunk['text']}\n"
            
            if total_length + len(paper_context) > max_context_length:
                break
            
            context_parts.append(paper_context)
            papers_used.append(paper)
            total_length += len(paper_context)
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt for LLM
        system_prompt = """You are a research assistant analyzing academic papers. 
Your task is to synthesize information from multiple papers to answer the user's query.

Guidelines:
1. Cite specific papers by author/year when making claims
2. Group findings by themes or categories when applicable
3. Note consensus and disagreements between papers
4. Provide specific quotes or findings when relevant
5. Be concise but comprehensive
6. Format your response with clear sections using bullet points
7. End with a count of papers cited"""

        user_prompt = f"""Query: {query}

Research Papers Context:
{context}

Based on the above papers, provide a comprehensive answer to the query. 
Synthesize findings across multiple papers and cite sources."""

        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            synthesis = response.choices[0].message.content
            
            return {
                'query': query,
                'synthesis': synthesis,
                'papers_analyzed': len(papers_used),
                'papers': [{
                    'title': p['title'],
                    'authors': p['authors'],
                    'year': p['year'],
                    'relevance_score': round(p['max_score'], 3)
                } for p in papers_used],
                'total_chunks_searched': len(search_results)
            }
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._fallback_synthesis(query, search_results)
    
    def _fallback_synthesis(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback synthesis without LLM"""
        papers = self.aggregate_by_paper(search_results)
        
        synthesis_parts = [f"Found {len(papers)} relevant papers for: '{query}'\n"]
        
        for i, paper in enumerate(papers[:10], 1):
            synthesis_parts.append(f"\n{i}. {paper['title']}")
            if paper['authors']:
                synthesis_parts.append(f"   Authors: {paper['authors']}")
            if paper['year']:
                synthesis_parts.append(f"   Year: {paper['year']}")
            synthesis_parts.append(f"   Relevance: {round(paper['max_score'] * 100, 1)}%")
            
            # Add snippet from best chunk
            if paper['chunks']:
                best_chunk = max(paper['chunks'], key=lambda x: x['score'])
                snippet = best_chunk['text'][:200] + "..."
                synthesis_parts.append(f"   Excerpt: {snippet}")
        
        return {
            'query': query,
            'synthesis': '\n'.join(synthesis_parts),
            'papers_analyzed': len(papers),
            'papers': [{
                'title': p['title'],
                'authors': p['authors'],
                'year': p['year'],
                'relevance_score': round(p['max_score'], 3)
            } for p in papers[:10]],
            'total_chunks_searched': len(search_results),
            'note': 'LLM synthesis unavailable - showing direct results'
        }
    
    def analyze_topics(self, query: str, n_results: int = 50) -> Dict[str, Any]:
        """Analyze common topics/themes in search results"""
        results = self.semantic_search(query, n_results=n_results)
        papers = self.aggregate_by_paper(results)
        
        # Extract and count categories
        categories = []
        years = []
        
        for paper in papers:
            if paper.get('categories'):
                cats = paper['categories'].split()
                categories.extend(cats)
            if paper.get('year'):
                years.append(paper['year'])
        
        category_counts = Counter(categories).most_common(10)
        year_counts = Counter(years).most_common(10)
        
        return {
            'query': query,
            'papers_found': len(papers),
            'top_categories': [{'category': c, 'count': n} for c, n in category_counts],
            'year_distribution': [{'year': y, 'count': n} for y, n in sorted(year_counts)],
            'papers': [{
                'title': p['title'],
                'authors': p['authors'],
                'year': p['year']
            } for p in papers[:10]]
        }
    
    def search(
        self,
        query: str,
        n_results: int = 20,
        include_synthesis: bool = True
    ) -> Dict[str, Any]:
        """Main search function with optional LLM synthesis"""
        
        # Perform semantic search
        search_results = self.semantic_search(query, n_results=n_results)
        
        if not search_results:
            return {
                'query': query,
                'synthesis': 'No relevant papers found for your query.',
                'papers_analyzed': 0,
                'papers': [],
                'total_chunks_searched': 0
            }
        
        if include_synthesis:
            return self.synthesize_response(query, search_results)
        else:
            papers = self.aggregate_by_paper(search_results)
            return {
                'query': query,
                'papers_analyzed': len(papers),
                'papers': [{
                    'title': p['title'],
                    'authors': p['authors'],
                    'year': p['year'],
                    'relevance_score': round(p['max_score'], 3),
                    'snippets': [c['text'][:200] for c in p['chunks'][:2]]
                } for p in papers],
                'total_chunks_searched': len(search_results)
            }
    
    def get_citation_network(self, paper_id: str = None) -> Dict[str, Any]:
        """Get citation network data (bonus feature)"""
        if not self.papers_collection:
            return {"error": "Papers not indexed"}
        
        # Get all papers with citation data
        all_papers = self.papers_collection.get(
            include=["metadatas", "documents"]
        )
        
        # Build citation statistics
        citation_counts = []
        for i, metadata in enumerate(all_papers['metadatas']):
            citation_counts.append({
                'id': all_papers['ids'][i],
                'title': metadata.get('title', '')[:100],
                'citation_count': metadata.get('citation_count', 0),
                'year': metadata.get('year', '')
            })
        
        # Sort by citation count
        citation_counts.sort(key=lambda x: x['citation_count'], reverse=True)
        
        return {
            'total_papers': len(citation_counts),
            'most_cited': citation_counts[:20],
            'citation_distribution': self._get_citation_distribution(citation_counts)
        }
    
    def _get_citation_distribution(self, papers: List[Dict]) -> Dict[str, int]:
        """Calculate citation distribution"""
        ranges = {
            '0': 0,
            '1-5': 0,
            '6-10': 0,
            '11-20': 0,
            '20+': 0
        }
        
        for paper in papers:
            count = paper['citation_count']
            if count == 0:
                ranges['0'] += 1
            elif count <= 5:
                ranges['1-5'] += 1
            elif count <= 10:
                ranges['6-10'] += 1
            elif count <= 20:
                ranges['11-20'] += 1
            else:
                ranges['20+'] += 1
        
        return ranges


# Test function
if __name__ == "__main__":
    engine = SearchEngine()
    
    print("\n" + "="*60)
    print("Testing Search Engine")
    print("="*60)
    
    stats = engine.get_stats()
    print(f"\nCorpus Stats: {stats}")
    
    if stats['status'] == 'ready':
        # Test search
        test_query = "machine learning neural networks"
        print(f"\nSearching for: '{test_query}'")
        
        results = engine.search(test_query, n_results=10, include_synthesis=True)
        print(f"\nFound {results['papers_analyzed']} papers")
        print(f"\nSynthesis:\n{results['synthesis'][:500]}...")
