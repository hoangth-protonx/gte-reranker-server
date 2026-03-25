
import torch
from typing import Optional, List, Literal, Any, Dict
from functools import lru_cache


# @lru_cache(maxsize=1)
def _load_reranker_model(model_name: str = './reranker_model', max_length=1024):
    """Load and cache the reranker model."""

    print(f"DEBUG: Load reranker model")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16
    )

    return model, tokenizer

def rerank_results(
    model,
    tokenizer,
    query: str,
    results: List[Dict[str, Any]],
    *,
    max_length: int = 128,
    top_k: int = None,
    use_title: bool = False,
    model_name: str = 'Alibaba-NLP/gte-multilingual-reranker-base',
) -> List[Dict[str, Any]]:
    """
    Rerank search results using a transformer-based reranker model.
    
    Args:
        query: The search query string
        results: List of dicts from search_by_title with keys: title, url, content, num_matches
        max_length: Maximum sequence length for the model
        top_k: If specified, return only top K results after reranking
        use_title_only: If True, rerank based on title only; if False, use title + content
        model_name: HuggingFace model name for reranking
    
    Returns:
        Reranked list of results with added 'relevance_score' field, sorted by score descending
    """
    if not results:
        return []
    
    # # Load model (cached)
    # try:
    #     model = _load_reranker_model(model_name=model_name, max_length=max_length)
    # except Exception as e:
    #     return f"Error loading reranker model: {e}"
    
    # Prepare documents for reranking
    documents = []
    for result in results:
        if use_title:
            doc = str(result['title'])
        else:
            # Combine title and content for better relevance
            doc = str(result['text'])
            max_length = 2048
        documents.append(doc)
    
    # Construct sentence pairs
    sentence_pairs = [[query, doc] for doc in documents]
    print(f"DEBUG reranker: num_sentence_pairs: {len(sentence_pairs)}")
    # Compute relevance scores
    try:
        with torch.no_grad():
            
            print(f"DEBUG reranker: start rerank: ....")
            inputs = tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            print(f"DEBUG reranker: finish rerank: ....")
        # Handle both single score and list of scores
        if isinstance(scores, (int, float)):
            scores = [scores]
        elif torch.is_tensor(scores):
            scores = scores.tolist()
    except Exception as e:
        return f"Error computing scores: {e}"
    
    # Add relevance scores to results
    reranked_results = []
    for result, score in zip(results, scores):
        result_copy = result.copy()
        result_copy['relevance_score'] = float(score)
        reranked_results.append(result_copy)
    
    # Sort by relevance score (descending)
    reranked_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Limit to top_k if specified
    if top_k is not None:
        reranked_results = reranked_results[:top_k]
    
    return reranked_results
