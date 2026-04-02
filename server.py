import os
os.environ["TRUST_REMOTE_CODE"] = "true"

import torch
from typing import Optional, List, Literal, Any, Dict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import signal
if not hasattr(signal, "SIGALRM"):
    signal.SIGALRM = signal.SIGINT
if not hasattr(signal, "alarm"):
    signal.alarm = lambda *args, **kwargs: None

# --- Configuration ---
DEFAULT_MODEL_PATH = Path("./reranker_model")
DEFAULT_MAX_LENGTH = 8192


# --- Pydantic Models ---
class RerankRequest(BaseModel):
    """Request model for reranking endpoint."""
    query: str = Field(..., description="The search query string")
    documents: List[str] = Field(..., description="List of document texts to rerank")
    max_length: int = Field(default=8192, description="Maximum sequence length for tokenization")
    top_k: Optional[int] = Field(default=20, description="Return only top K results")
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16  # ✅ Fixed: `torch_dtype` is deprecated, use `dtype`
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = model.to(device)
    except Exception as e:
        print("⚠️ CUDA failed, falling back to CPU:", e)
        device = "cpu"
        model = model.to(device)

    model.eval()

    # ✅ Clamp max_length to the model's actual max position embeddings if available
    try:
        model_max_pos = model.config.max_position_embeddings
        if max_length > model_max_pos:
            print(f"⚠️ Clamping max_length from {max_length} to model's max_position_embeddings={model_max_pos}")
            max_length = model_max_pos
    except AttributeError:
        pass

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
    if not _model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = _model_cache["model"]
    tokenizer = _model_cache["tokenizer"]
    device = _model_cache["device"]

    if not request.documents:
        return RerankResponse(results=[], query=request.query)

    # ✅ Respect the request's max_length but clamp it to the model's safe limit
    effective_max_length = min(request.max_length, _model_cache["max_length"])

    sentence_pairs = [[request.query, doc] for doc in request.documents]

    with torch.no_grad():
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=effective_max_length  # ✅ Use effective (clamped) length
        )

        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ Validate position_ids won't exceed RoPE table before calling model
        seq_len = inputs["input_ids"].shape[1]
        if seq_len > _model_cache["max_length"]:
            raise HTTPException(
                status_code=400,
                detail=f"Tokenized sequence length {seq_len} exceeds model max {_model_cache['max_length']}"
            )

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
