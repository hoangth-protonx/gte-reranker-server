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


def get_safe_max_length(model, requested_max: int) -> int:
    # Exact path now confirmed from /debug/buffers
    for name, buf in model.named_buffers():
        if "cos_cached" in name:
            table_size = buf.shape[0]
            print(f"✅ RoPE cos_cached '{name}' shape={list(buf.shape)} → limit={table_size}")
            return min(requested_max, table_size)

    print("⚠️ cos_cached not found, using fallback=512")
    return 512


def load_reranker_model(model_path: str = str(DEFAULT_MODEL_PATH), max_length: int = DEFAULT_MAX_LENGTH):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = model.to(device)
    except Exception as e:
        print("⚠️ CUDA failed, falling back to CPU:", e)
        device = "cpu"
        model = model.to(device)

    model.eval()

    safe_max = get_safe_max_length(model, max_length)
    print(f"✅ Final safe_max_length = {safe_max}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_length": safe_max
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
    safe_max = _model_cache["max_length"]

    if not request.documents:
        return RerankResponse(results=[], query=request.query)

    effective_max_length = min(request.max_length, safe_max)
    sentence_pairs = [[request.query, doc] for doc in request.documents]

    with torch.no_grad():
        inputs = tokenizer(
            sentence_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=effective_max_length
        )
        inputs.pop("token_type_ids", None)

        actual_seq_len = inputs["input_ids"].shape[1]
        print(f"[rerank] docs={len(request.documents)} | requested_max={request.max_length} "
              f"| effective_max={effective_max_length} | actual_seq_len={actual_seq_len} "
              f"| safe_max={safe_max}")

        # Hard block before it reaches CUDA
        if actual_seq_len > safe_max:
            raise HTTPException(
                status_code=400,
                detail=f"Sequence length {actual_seq_len} exceeds RoPE table size {safe_max}"
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ✅ Explicitly pass position_ids so the model doesn't compute them wrong
        batch_size = inputs["input_ids"].shape[0]
        position_ids = torch.arange(actual_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        inputs["position_ids"] = position_ids

        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = scores.cpu().tolist()

    results_with_scores = [
        {"document": doc, "relevance_score": float(score), "index": idx}
        for idx, (doc, score) in enumerate(zip(request.documents, scores))
    ]
    results_with_scores.sort(key=lambda x: x["relevance_score"], reverse=True)

    if request.top_k is not None:
        results_with_scores = results_with_scores[:request.top_k]

    return RerankResponse(
        results=[RerankResult(**r) for r in results_with_scores],
        query=request.query
    )


@app.get("/debug/rope", tags=["Debug"])
async def debug_rope():
    model = _model_cache["model"]
    info = {"safe_max_length": _model_cache["max_length"]}
    for path in ["new.embeddings.rope", "model.embeddings.rope", "embeddings.rope"]:
        try:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part)
            info[path] = list(obj.cos_cached.shape)
        except AttributeError:
            info[path] = "not found"
    return info

@app.get("/debug/model-tree", tags=["Debug"])
async def debug_model_tree():
    model = _model_cache["model"]
    # Find any module with 'rope' or 'cos' in its name/attributes
    rope_modules = {}
    for name, module in model.named_modules():
        if "rope" in name.lower() or "rotary" in name.lower():
            attrs = {}
            for attr in ["cos_cached", "sin_cached", "max_seq_len", "max_position_embeddings"]:
                if hasattr(module, attr):
                    val = getattr(module, attr)
                    attrs[attr] = list(val.shape) if hasattr(val, "shape") else val
            rope_modules[name] = attrs
    return rope_modules

@app.get("/debug/buffers", tags=["Debug"])
async def debug_buffers():
    model = _model_cache["model"]
    return {
        name: list(buf.shape)
        for name, buf in model.named_buffers()
    }