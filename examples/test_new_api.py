#!/usr/bin/env python3
"""
Test the new Jugglerr API.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jugglerr import Jugglerr

print("Testing new Jugglerr API...")
print("=" * 70)

# Test 1: Initialization
print("\n1. Testing initialization (auto-load from .env)")
try:
    jugglerr = Jugglerr()
    print("✅ Jugglerr initialized successfully")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Test 2: Chat
print("\n2. Testing chat()")
try:
    response = jugglerr.chat([
        {"role": "user", "content": "Say 'Hello' in one word."}
    ])
    print(f"✅ Chat works: {response[:50]}...")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Chat streaming
print("\n3. Testing chat_stream()")
try:
    chunks = []
    for chunk in jugglerr.chat_stream([
        {"role": "user", "content": "Say 'Test' in one word."}
    ]):
        chunks.append(chunk)
    response = ''.join(chunks)
    print(f"✅ Streaming works: {response[:50]}...")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Embeddings
print("\n4. Testing embed()")
try:
    embeddings = jugglerr.embed(["Hello world"])
    print(f"✅ Embeddings work: {len(embeddings[0])} dimensions")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Reranking
print("\n5. Testing rerank()")
try:
    docs = ["Python is great", "The sky is blue", "AI is powerful"]
    top_docs = jugglerr.rerank("What is AI?", docs, top_k=2)
    print(f"✅ Reranking works: {len(top_docs)} docs returned")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
