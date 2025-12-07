"""
NVIDIA Reranking Example

Demonstrates using NVIDIA's reranking model for improved RAG accuracy.
Based on Perplexity AI research (December 5, 2025).

Impact: ~24% improvement in recall@5 (0.5699 → 0.7070)
"""

import os
from jugglerr import RerankAdapter, rerank

# Get API key
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEYS", "").split(",")[0]

if not api_key:
    print("❌ Please set NVIDIA_API_KEY environment variable")
    exit(1)

print("="*70)
print("NVIDIA Reranking Examples")
print("Model: nvidia/llama-3.2-nv-rerankqa-1b-v2")
print("="*70)

# Sample documents for reranking
documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "JavaScript is primarily used for web development and runs in browsers.",
    "Natural language processing helps computers understand and generate human language.",
    "React is a JavaScript library for building user interfaces.",
    "Supervised learning requires labeled training data to make predictions.",
    "CSS is used for styling web pages and controlling layout.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "HTML provides the structure for web pages."
]

# Example 1: Basic Reranking
print("\n1. Basic Reranking")
print("-" * 70)

query = "What is machine learning?"
print(f"Query: {query}\n")

reranker = RerankAdapter(api_key=api_key)
results = reranker.rerank(query, documents, top_n=3)

print(f"Top 3 results:")
for i, result in enumerate(results.results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   {result.text}")

# Example 2: Quick Reranking Function
print("\n\n2. Quick Reranking Function")
print("-" * 70)

query = "How does deep learning work?"
print(f"Query: {query}\n")

# Use convenience function
top_docs = rerank(query, documents, api_key=api_key, top_n=2)

print(f"Top 2 results:")
for i, doc in enumerate(top_docs, 1):
    print(f"\n{i}. {doc}")

# Example 3: Reranking with Scores
print("\n\n3. Reranking with Scores")
print("-" * 70)

query = "web development technologies"
print(f"Query: {query}\n")

results_with_scores = reranker.rerank_with_scores(query, documents, top_n=3)

print(f"Top 3 results with scores:")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   {doc}")

# Example 4: Reranking Indices (for large documents)
print("\n\n4. Reranking Indices (Memory Efficient)")
print("-" * 70)

query = "programming languages"
print(f"Query: {query}\n")

# Get indices instead of copying documents
indices = reranker.rerank_indices(query, documents, top_n=3)

print(f"Top 3 document indices: {indices}")
print(f"\nTop 3 documents:")
for i, idx in enumerate(indices, 1):
    print(f"\n{i}. Index {idx}: {documents[idx]}")

# Example 5: RAG Pipeline Simulation
print("\n\n5. Complete RAG Pipeline Simulation")
print("-" * 70)

# Simulate semantic search results (100 documents)
print("Step 1: Semantic search retrieves 100 documents")
print("Step 2: Reranking picks best 5 documents")

query = "explain neural networks"
print(f"\nQuery: {query}")

# In real RAG, you'd have 100 docs from semantic search
# Here we'll use our 10 sample docs
print(f"\nRetrieved documents: {len(documents)}")

# Rerank to get best 5
top_5 = rerank(query, documents, api_key=api_key, top_n=5)

print(f"\nTop 5 after reranking:")
for i, doc in enumerate(top_5, 1):
    print(f"\n{i}. {doc[:80]}...")

print("\nStep 3: Send top 5 to LLM for generation")
print("Result: Higher accuracy, lower cost!")

# Example 6: Comparing Before/After Reranking
print("\n\n6. Impact of Reranking")
print("-" * 70)

query = "What is AI?"

print(f"Query: {query}\n")

# Without reranking (just first 3 docs)
print("WITHOUT RERANKING (first 3 docs):")
for i, doc in enumerate(documents[:3], 1):
    print(f"{i}. {doc[:60]}...")

# With reranking
top_3_reranked = rerank(query, documents, api_key=api_key, top_n=3)

print("\nWITH RERANKING (top 3 by relevance):")
for i, doc in enumerate(top_3_reranked, 1):
    print(f"{i}. {doc[:60]}...")

print("\n" + "="*70)
print("✅ Reranking Examples Complete!")
print("="*70)
print("\nKey Takeaways:")
print("- Reranking improves RAG accuracy by ~24%")
print("- Reduces token costs by 80-90%")
print("- Essential for production RAG systems")
print("- Works with any semantic search backend")
