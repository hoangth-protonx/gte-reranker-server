"""
FastAPI server for serving the GTE multilingual reranker model.

This server provides endpoints for:
- Reranking search results based on query relevance
- Health checks and model info
"""

# server.py
import os
os.environ["TRUST_REMOTE_CODE"] = "true"

import torch
from typing import Optional, List, Literal, Any, Dict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


import signal
if not hasattr(signal, "SIGALRM"):
    signal.SIGALRM = signal.SIGINT
if not hasattr(signal, "alarm"):
    signal.alarm = lambda *args, **kwargs: None

# --- Configuration ---
DEFAULT_MODEL_PATH = Path("./reranker_model")
DEFAULT_MAX_LENGTH = 1024


# --- Pydantic Models ---
class RerankRequest(BaseModel):
    """Request model for reranking endpoint."""
    query: str = Field(..., description="The search query string")
    documents: List[str] = Field(..., description="List of document texts to rerank")
    max_length: int = Field(default=128, description="Maximum sequence length for tokenization")
    top_k: Optional[int] = Field(default=None, description="Return only top K results")
    use_title: bool = Field(default=False, description="If True, treat documents as titles")


class RerankResult(BaseModel):
    """Single reranking result."""
    document: str = Field(..., description="The document text")
    relevance_score: float = Field(..., description="Relevance score from the model")
    index: int = Field(..., description="Original index in the input documents list")


class RerankResponse(BaseModel):
    """Response model for reranking endpoint."""
    results: List[RerankResult] = Field(..., description="Reranked results with scores")
    query: str = Field(..., description="The original query")


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    model_path: str
    max_length: int
    device: str
    dtype: str


# --- Global model state ---
_model_cache: Dict[str, Any] = {}


def load_reranker_model(model_path: str = str(DEFAULT_MODEL_PATH), max_length: int = DEFAULT_MAX_LENGTH):
    """Load the reranker model and tokenizer."""
    print(f"DEBUG: Loading reranker model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"DEBUG: Model loaded on device: {device}")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_length": max_length
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model_cache
    _model_cache = load_reranker_model()
    yield
    # Cleanup (if needed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- FastAPI App ---
app = FastAPI(
    title="GTE Reranker Server",
    description="FastAPI server for serving the Alibaba GTE multilingual reranker model",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_path=str(DEFAULT_MODEL_PATH),
        max_length=_model_cache["max_length"],
        device=_model_cache["device"],
        dtype="float16"
    )


@app.post("/rerank", response_model=RerankResponse, tags=["Reranking"])
async def rerank(request: RerankRequest):
    """
    Rerank a list of documents based on their relevance to a query.
    
    This endpoint takes a query and a list of documents, computes relevance scores
    using the GTE multilingual reranker model, and returns the documents sorted
    by relevance score in descending order.
    """
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = _model_cache["model"]
    tokenizer = _model_cache["tokenizer"]
    device = _model_cache["device"]
    
    if not request.documents:
        return RerankResponse(results=[], query=request.query)
    
    # Prepare sentence pairs
    sentence_pairs = [[request.query, doc] for doc in request.documents]
    
    # Tokenize and compute scores
    with torch.no_grad():
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=request.max_length
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        
        if torch.is_tensor(scores):
            scores = scores.cpu().tolist()
    
    # Build results with original indices
    results_with_scores = []
    for idx, (doc, score) in enumerate(zip(request.documents, scores)):
        results_with_scores.append({
            "document": doc,
            "relevance_score": float(score),
            "index": idx
        })
    
    # Sort by relevance score (descending)
    results_with_scores.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Limit to top_k if specified
    if request.top_k is not None:
        results_with_scores = results_with_scores[:request.top_k]
    
    # Convert to Pydantic models
    rerank_results = [
        RerankResult(
            document=r["document"],
            relevance_score=r["relevance_score"],
            index=r["index"]
        )
        for r in results_with_scores
    ]
    
    return RerankResponse(results=rerank_results, query=request.query)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
