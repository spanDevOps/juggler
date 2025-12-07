#!/usr/bin/env python3
"""
Example showing models_used tracking in Juggler responses.

All Juggler methods return objects that behave like their simple types
(string, list) but include a models_used attribute for transparency.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from juggler import Juggler

juggler = Juggler()

print("=" * 70)
print("MODELS_USED TRACKING EXAMPLES")
print("=" * 70)

# Example 1: Chat - behaves like a string
print("\n1. CHAT RESPONSE")
print("-" * 70)
response = juggler.chat([
    {"role": "user", "content": "Say 'Hello' in one word."}
])

# Works like a string
print(f"Response: {response}")
print(f"Length: {len(response)}")
print(f"Upper: {response.upper()}")

# But also has models_used
print(f"\nModels used: {len(response.models_used)} attempt(s)")
for attempt in response.models_used:
    status = "✅" if attempt.success else "❌"
    print(f"  {status} Attempt {attempt.attempt}: {attempt.provider}/{attempt.model}")
    if not attempt.success:
        print(f"     Error: {attempt.error}")

# Example 2: Embeddings - behaves like a list
print("\n2. EMBEDDING RESPONSE")
print("-" * 70)
embeddings = juggler.embed(["Hello", "World"])

# Works like a list
print(f"Number of embeddings: {len(embeddings)}")
print(f"First embedding dimensions: {len(embeddings[0])}")
print(f"Can iterate: {[len(e) for e in embeddings]}")

# But also has models_used
print(f"\nModels used: {len(embeddings.models_used)} attempt(s)")
for attempt in embeddings.models_used:
    status = "✅" if attempt.success else "❌"
    print(f"  {status} {attempt.provider}/{attempt.model}")

# Also has dimensions attribute
print(f"Dimensions: {embeddings.dimensions}")

# Example 3: Reranking - behaves like a list
print("\n3. RERANK RESPONSE")
print("-" * 70)
docs = ["Python is great", "The sky is blue", "AI is powerful"]
top_docs = juggler.rerank("What is AI?", docs, top_k=2)

# Works like a list
print(f"Top documents: {len(top_docs)}")
for i, doc in enumerate(top_docs, 1):
    print(f"  {i}. {doc}")

# But also has models_used
print(f"\nModels used: {len(top_docs.models_used)} attempt(s)")
for attempt in top_docs.models_used:
    status = "✅" if attempt.success else "❌"
    print(f"  {status} {attempt.provider}/{attempt.model}")

# Example 4: Serializing models_used to JSON
print("\n4. SERIALIZING TO JSON")
print("-" * 70)
response = juggler.chat([
    {"role": "user", "content": "Count to 3"}
])

# Convert models_used to JSON
tracking_data = {
    "response": str(response),
    "models_used": [dict(attempt) for attempt in response.models_used]
}

print(json.dumps(tracking_data, indent=2))

# Example 5: Checking fallback path
print("\n5. UNDERSTANDING FALLBACK PATH")
print("-" * 70)
response = juggler.chat([
    {"role": "user", "content": "Hello"}
])

print("Fallback chain:")
for attempt in response.models_used:
    if attempt.success:
        print(f"  ✅ SUCCESS on attempt {attempt.attempt}: {attempt.provider}/{attempt.model}")
        print(f"     Status: {attempt.status_code}")
        if attempt.elapsed_ms:
            print(f"     Time: {attempt.elapsed_ms:.0f}ms")
        break
    else:
        print(f"  ❌ FAILED attempt {attempt.attempt}: {attempt.provider}/{attempt.model}")
        print(f"     Error: {attempt.error_type}: {attempt.error}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
