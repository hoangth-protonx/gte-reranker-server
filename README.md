# GTE Reranker Server

A high-performance FastAPI server for serving the **Alibaba GTE Multilingual Reranker** model (`Alibaba-NLP/gte-multilingual-reranker-base`).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    API Endpoints                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │  POST       │  │  GET        │  │  GET            │   │  │
│  │  │  /rerank    │  │  /model/info│  │  /health        │   │  │
│  │  └──────┬──────┘  └─────────────┘  └─────────────────┘   │  │
│  └─────────┼────────────────────────────────────────────────┘  │
│            │                                                    │
│  ┌─────────▼────────────────────────────────────────────────┐  │
│  │              Model Inference Layer                        │  │
│  │  ┌─────────────────────┐  ┌────────────────────────────┐ │  │
│  │  │   Tokenizer         │  │   Model (float16)          │ │  │
│  │  │   (AutoTokenizer)   │  │   (AutoModelForSeqClass)   │ │  │
│  │  └─────────┬───────────┘  └────────────┬───────────────┘ │  │
│  └────────────┼────────────────────────────┼────────────────┘  │
│               │                            │                    │
│  ┌────────────▼────────────────────────────▼────────────────┐  │
│  │              Model Cache (Global State)                   │  │
│  │  { model, tokenizer, device, max_length }                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Local Model Directory       │
              │   ./reranker_model/           │
              │   - config.json               │
              │   - tokenizer files           │
              │   - model weights (.bin)      │
              │   - modeling.py (custom)      │
              │   - configuration.py (custom) │
              └───────────────────────────────┘
```

## Components

### 1. **API Layer** (`server.py`)

The FastAPI application exposes three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for load balancers |
| `/model/info` | GET | Returns model configuration and runtime info |
| `/rerank` | POST | Main reranking endpoint |

#### Request/Response Flow for `/rerank`:

```
Client Request
    ↓
┌─────────────────────────────────────┐
│ RerankRequest                       │
│ - query: str                        │
│ - documents: List[str]              │
│ - max_length: int (default: 128)    │
│ - top_k: Optional[int]              │
│ - use_title: bool (default: False)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Tokenization                        │
│ - Create sentence pairs [query, doc]│
│ - Batch tokenize with truncation    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Model Inference (GPU/CPU)           │
│ - torch.no_grad() context           │
│ - float16 precision                 │
│ - Returns logits → relevance scores │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Post-processing                     │
│ - Sort by score (descending)        │
│ - Apply top_k limit                 │
│ - Preserve original indices         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ RerankResponse                      │
│ - results: List[RerankResult]       │
│ - query: str                        │
└─────────────────────────────────────┘
```

### 2. **Model Loading**

The model is loaded once at startup using FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_cache
    _model_cache = load_reranker_model()
    yield
    # Cleanup on shutdown
```

**Key features:**
- **Single initialization**: Model loaded once, shared across all requests
- **GPU acceleration**: Automatically uses CUDA if available
- **Memory efficiency**: Uses `float16` precision for reduced memory footprint

### 3. **Data Models** (Pydantic)

```python
RerankRequest       # Input: query + documents
RerankResult        # Single result with score + original index
RerankResponse      # Output: sorted results
ModelInfoResponse   # Model metadata
```

## Setup

### 1. Download the Model

Run the model export script to download and prepare the model for offline use:

```bash
python save_pretrained.py
```

This will:
- Download the tokenizer and model from HuggingFace
- Save custom `modeling.py` and `configuration.py` files (required for `trust_remote_code=True`)
- Store everything in `./reranker_model/`

### 2. Install Dependencies

```bash
pip install fastapi uvicorn transformers torch accelerate
```

### 3. Run the Server

```bash
# Development
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Production (recommended)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Usage

### Example: Rerank Documents
https://dilan-aedilitian-subadditively.ngrok-free.dev/
```bash
curl -X POST "https://dilan-aedilitian-subadditively.ngrok-free.dev/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "documents": [
      "Introduction to Python programming",
      "Deep learning with neural networks",
      "Basic statistics for data science",
      "Advanced machine learning algorithms"
    ],
    "top_k": 2
  }'
```

**Response:**

```json
{
  "results": [
    {
      "document": "Advanced machine learning algorithms",
      "relevance_score": 8.523,
      "index": 3
    },
    {
      "document": "Deep learning with neural networks",
      "relevance_score": 7.891,
      "index": 1
    }
  ],
  "query": "machine learning tutorial"
}
```

### Example: Check Model Info

```bash
curl "http://localhost:8000/model/info"
```

## Configuration

| Environment/Parameter | Default | Description |
|----------------------|---------|-------------|
| `DEFAULT_MODEL_PATH` | `./reranker_model` | Local path to model directory |
| `DEFAULT_MAX_LENGTH` | `1024` | Default max sequence length |
| `--max_length` (request) | `128` | Per-request tokenization limit |
| `--top_k` (request) | `null` | Limit number of returned results |
| `torch_dtype` | `float16` | Model precision for inference |

## Performance Considerations

1. **Batch Processing**: All documents are tokenized and scored in a single batch
2. **GPU Memory**: `float16` reduces VRAM usage by ~50% vs `float32`
3. **Model Caching**: Loaded once at startup, zero cold-start latency per request
4. **Async Support**: FastAPI's async nature handles concurrent requests efficiently

## Project Structure

```
gte-reranker-server/
├── server.py              # Main FastAPI application
├── save_pretrained.py     # Model download/export script
├── inference.py           # Reference inference logic
├── README.md              # This documentation
└── reranker_model/        # Local model directory (after download)
    ├── config.json
    ├── tokenizer.json
    ├── pytorch_model.bin
    ├── modeling.py        # Custom model implementation
    └── configuration.py   # Custom configuration
```

## API Documentation

Interactive API docs are available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Set up Nvidia runtime 

```bash
# 1) kiểm tra host đã thấy GPU chưa
nvidia-smi

# 2) cài NVIDIA Container Toolkit (nếu chưa cài)
apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# 3) cấu hình Docker dùng NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker

# 4) restart Docker
systemctl restart docker
# nếu máy không có systemd thì thử:
# service docker restart
```
