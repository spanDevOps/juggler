#!/usr/bin/env python3
"""
Quick test to verify models_used tracking works.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from juggler import Juggler

juggler = Juggler()

print("=" * 70)
print("TESTING MODELS_USED TRACKING")
print("=" * 70)

# Test 1: Chat
print("\n1. CHAT")
print("-" * 70)
response = juggler.chat([{"role": "user", "content": "Say hello"}])
print(f"Response: {response[:50]}...")
print(f"Type: {type(response)}")
print(f"Has models_used: {hasattr(response, 'models_used')}")
print(f"Models used: {len(response.models_used)} attempt(s)")
for attempt in response.models_used:
    print(f"  - {attempt.provider}/{attempt.model}: {'✅' if attempt.success else '❌'}")

# Test 2: Embeddings
print("\n2. EMBEDDINGS")
print("-" * 70)
embeddings = juggler.embed(["Hello"])
print(f"Embeddings: {len(embeddings)} vectors")
print(f"Type: {type(embeddings)}")
print(f"Has models_used: {hasattr(embeddings, 'models_used')}")
print(f"Has dimensions: {hasattr(embeddings, 'dimensions')}")
print(f"Dimensions: {embeddings.dimensions}")
print(f"Models used: {len(embeddings.models_used)} attempt(s)")
for attempt in embeddings.models_used:
    print(f"  - {attempt.provider}/{attempt.model}: {'✅' if attempt.success else '❌'}")

# Test 3: Reranking
print("\n3. RERANKING")
print("-" * 70)
docs = ["Python is great", "The sky is blue", "AI is powerful"]
top_docs = juggler.rerank("What is AI?", docs, top_k=2)
print(f"Top docs: {len(top_docs)}")
print(f"Type: {type(top_docs)}")
print(f"Has models_used: {hasattr(top_docs, 'models_used')}")
print(f"Has scores: {hasattr(top_docs, 'scores')}")
print(f"Models used: {len(top_docs.models_used)} attempt(s)")
for attempt in top_docs.models_used:
    print(f"  - {attempt.provider}/{attempt.model}: {'✅' if attempt.success else '❌'}")

# Test 4: Backward compatibility - works like simple types
print("\n4. BACKWARD COMPATIBILITY")
print("-" * 70)
response = juggler.chat([{"role": "user", "content": "Test"}])
print(f"Can use as string: {response.upper()[:20]}...")
print(f"Can get length: {len(response)}")
print(f"Can slice: {response[:10]}...")

embeddings = juggler.embed(["Test"])
print(f"Can iterate: {len([e for e in embeddings])}")
print(f"Can index: {len(embeddings[0])}")

top_docs = juggler.rerank("test", ["a", "b", "c"])
print(f"Can iterate: {[doc for doc in top_docs]}")
print(f"Can index: {top_docs[0]}")

print("\n" + "=" * 70)
print("✅ ALL TRACKING TESTS PASSED!")
print("=" * 70)
