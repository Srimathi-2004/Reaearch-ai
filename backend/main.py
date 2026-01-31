"""
FastAPI Backend for Intelligent Document Search
Main API endpoints for search, synthesis, and analytics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
from contextlib import asynccontextmanager

from search_engine import SearchEngine


# Initialize search engine globally
search_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global search_engine
    print("🚀 Starting Intelligent Document Search API...")
    search_engine = SearchEngine()
    yield
    print("👋 Shutting down...")


# FastAPI app
app = FastAPI(
    title="Intelligent Document Search API",
    description="Search across 24,000+ research papers with LLM-powered synthesis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    n_results: int = Field(20, description="Number of results", ge=1, le=100)
    include_synthesis: bool = Field(True, description="Include LLM synthesis")


class PaperResult(BaseModel):
    title: str
    authors: str
    year: str
    relevance_score: float
    snippets: Optional[List[str]] = None


class SearchResponse(BaseModel):
    query: str
    synthesis: Optional[str] = None
    papers_analyzed: int
    papers: List[Dict[str, Any]]
    total_chunks_searched: int
    response_time_ms: float
    note: Optional[str] = None


class StatsResponse(BaseModel):
    papers_count: int
    chunks_count: int
    status: str


class TopicAnalysisResponse(BaseModel):
    query: str
    papers_found: int
    top_categories: List[Dict[str, Any]]
    year_distribution: List[Dict[str, Any]]
    papers: List[Dict[str, Any]]


class CitationNetworkResponse(BaseModel):
    total_papers: int
    most_cited: List[Dict[str, Any]]
    citation_distribution: Dict[str, int]


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Intelligent Document Search API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "search": "/api/search",
            "stats": "/api/stats",
            "topics": "/api/topics",
            "citations": "/api/citations"
        }
    }


@app.get("/api/stats", response_model=StatsResponse, tags=["Analytics"])
async def get_stats():
    """Get corpus statistics"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    stats = search_engine.get_stats()
    return StatsResponse(**stats)


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search across research papers with optional LLM synthesis
    
    - **query**: Your search query (e.g., "What are the main criticisms of BERT?")
    - **n_results**: Number of papers to search (default: 20)
    - **include_synthesis**: Whether to include LLM-powered synthesis (default: true)
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    start_time = time.time()
    
    try:
        results = search_engine.search(
            query=request.query,
            n_results=request.n_results,
            include_synthesis=request.include_synthesis
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=results['query'],
            synthesis=results.get('synthesis'),
            papers_analyzed=results['papers_analyzed'],
            papers=results['papers'],
            total_chunks_searched=results['total_chunks_searched'],
            response_time_ms=round(response_time, 2),
            note=results.get('note')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(20, description="Number of results", ge=1, le=100),
    include_synthesis: bool = Query(True, description="Include LLM synthesis")
):
    """Search using GET request (alternative to POST)"""
    request = SearchRequest(
        query=query,
        n_results=n_results,
        include_synthesis=include_synthesis
    )
    return await search(request)


@app.get("/api/topics", response_model=TopicAnalysisResponse, tags=["Analytics"])
async def analyze_topics(
    query: str = Query(..., description="Topic query"),
    n_results: int = Query(50, description="Number of papers to analyze", ge=1, le=200)
):
    """Analyze topics and trends related to a query"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        results = search_engine.analyze_topics(query, n_results=n_results)
        return TopicAnalysisResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/citations", response_model=CitationNetworkResponse, tags=["Analytics"])
async def get_citation_network():
    """Get citation network analysis (bonus feature)"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        results = search_engine.get_citation_network()
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        return CitationNetworkResponse(**results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Citation analysis error: {str(e)}")


@app.get("/api/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    if not search_engine:
        return {"status": "unhealthy", "reason": "Search engine not initialized"}
    
    stats = search_engine.get_stats()
    return {
        "status": "healthy" if stats['status'] == 'ready' else "degraded",
        "papers_indexed": stats['papers_count'],
        "chunks_indexed": stats['chunks_count']
    }


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
