"""
Cohere Adapters Example
=======================

This example demonstrates how to use Cohere's specialized adapters:
1. Embedding Adapter - Generate embeddings for semantic search
2. Rerank Adapter - Rerank search results by relevance
"""

import os
from jugglerr import cohere_embed, cohere_rerank

# Get API key from environment
api_key = os.getenv('COHERE_API_KEY')

# ============================================================================
# Example 1: Generate Embeddings
# ============================================================================
print("Example 1: Generate Embeddings")
print("-" * 70)

documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming technology",
    "Python is great for data science"
]

# Generate embeddings for documents
embeddings = cohere_embed(
    texts=documents,
    model="embed-english-v3.0",  # or "embed-v4.0" for higher quality
    api_key=api_key,
    input_type="search_document"  # or "search_query" for queries
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")

# ============================================================================
# Example 2: Semantic Search with Embeddings
# ============================================================================
print("\n\nExample 2: Semantic Search")
print("-" * 70)

# Embed a query
query = "Tell me about AI"
query_embedding = cohere_embed(
    texts=[query],
    model="embed-english-v3.0",
    api_key=api_key,
    input_type="search_query"
)[0]

# Calculate cosine similarity (simple dot product for normalized vectors)
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar document
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in embeddings]
best_match_idx = np.argmax(similarities)

print(f"Query: {query}")
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.4f}")

# ============================================================================
# Example 3: Rerank Search Results
# ============================================================================
print("\n\nExample 3: Rerank Search Results")
print("-" * 70)

query = "What is machine learning?"
documents = [
    "Machine learning is a subset of AI that focuses on data and algorithms.",
    "Python is a programming language used for web development.",
    "Deep learning uses neural networks to learn from data.",
    "JavaScript is used for creating interactive websites.",
    "Supervised learning trains models on labeled data."
]

# Rerank documents by relevance
results = cohere_rerank(
    query=query,
    documents=documents,
    model="rerank-v3.5",
    api_key=api_key,
    top_n=3  # Return top 3 most relevant
)

print(f"Query: {query}\n")
print("Top 3 most relevant documents:")
for i, result in enumerate(results, 1):
    print(f"{i}. [Score: {result['relevance_score']:.4f}]")
    print(f"   {result['document']}")
    print()

# ============================================================================
# Example 4: Using CohereEmbeddingAdapter Directly
# ============================================================================
print("\n\nExample 4: Using Adapter Classes Directly")
print("-" * 70)

from jugglerr.cohere_adapters import CohereEmbeddingAdapter, CohereRerankAdapter

# Initialize adapters
embed_adapter = CohereEmbeddingAdapter(api_key=api_key)
rerank_adapter = CohereRerankAdapter(api_key=api_key)

# Use embedding adapter
embeddings = embed_adapter.embed(
    texts=["Hello world", "Test document"],
    model="embed-english-v3.0",
    input_type="search_document"
)
print(f"Embeddings generated: {len(embeddings)}")

# Use rerank adapter
results = rerank_adapter.rerank(
    query="machine learning",
    documents=["AI is cool", "ML is a subset of AI", "Python is great"],
    model="rerank-v3.5",
    top_n=2
)
print(f"Rerank results: {len(results)}")
for result in results:
    print(f"  - Score: {result['relevance_score']:.4f}, Doc: {result['document'][:40]}...")

print("\n" + "=" * 70)
print("✅ All examples completed!")
