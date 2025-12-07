#!/usr/bin/env python3
"""
Naive user style examples - just specify what you need, let Jugglerr choose.

This shows how a typical user would use Jugglerr without worrying about
specific models or providers.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jugglerr import Jugglerr

jugglerr = Jugglerr()

print("=" * 70)
print("NAIVE USER STYLE - LET JUGGLERR CHOOSE")
print("=" * 70)

# Example 1: Simple chat - no requirements
print("\n1. SIMPLE CHAT (no requirements)")
print("-" * 70)
print("User: Just wants a response, doesn't care which model")
response = jugglerr.chat([
    {"role": "user", "content": "What is 2+2?"}
])
print(f"Response: {response}")
print(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")

# Example 2: Need a smart model
print("\n2. COMPLEX TASK (power='super')")
print("-" * 70)
print("User: Needs a powerful model for complex reasoning")
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Explain quantum entanglement in simple terms"}],
    power="super"  # Use 70B+ models
)
print(f"Response: {response[:100]}...")
print(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")

# Example 3: Need vision
print("\n3. VISION TASK (capabilities=['vision'])")
print("-" * 70)
print("User: Needs to analyze an image")
print("(Skipping actual image - would include base64 image data)")
# response = jugglerr.chat(
#     messages=[{
#         "role": "user",
#         "content": "What's in this image?",
#         "images": ["base64_image_data"]
#     }],
#     capabilities=["vision"]
# )
print("Would use: Groq/Mistral/NVIDIA vision models")

# Example 4: Need tool calling
print("\n4. TOOL CALLING (capabilities=['tool_calling'])")
print("-" * 70)
print("User: Needs function calling")
response = jugglerr.chat(
    messages=[{"role": "user", "content": "What's the weather like?"}],
    capabilities=["tool_calling"]
)
print(f"Response: {response[:100]}...")
print(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")

# Example 5: Need reasoning
print("\n5. REASONING TASK (capabilities=['reasoning'])")
print("-" * 70)
print("User: Needs advanced reasoning")
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Solve this logic puzzle: ..."}],
    capabilities=["reasoning"]
)
print(f"Response: {response[:100]}...")
print(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")

# Example 6: Multiple capabilities
print("\n6. MULTIPLE CAPABILITIES (vision + reasoning)")
print("-" * 70)
print("User: Needs both vision and reasoning")
print("(Would use models that support both)")
# response = jugglerr.chat(
#     messages=[{
#         "role": "user",
#         "content": "Analyze this chart and explain the trend",
#         "images": ["base64_chart_data"]
#     }],
#     capabilities=["vision", "reasoning"]
# )
print("Would use: Models with both vision and reasoning")

# Example 7: Embeddings - just works
print("\n7. EMBEDDINGS (no model specified)")
print("-" * 70)
print("User: Just wants embeddings")
embeddings = jugglerr.embed(["Hello", "World"])
print(f"Generated {len(embeddings)} embeddings of {embeddings.dimensions} dimensions")
print(f"Used: {embeddings.models_used[0].provider}/{embeddings.models_used[0].model}")

# Example 8: Reranking - just works
print("\n8. RERANKING (no model specified)")
print("-" * 70)
print("User: Just wants documents reranked")
docs = ["Python is great", "The sky is blue", "AI is powerful"]
top_docs = jugglerr.rerank("What is AI?", docs, top_k=2)
print(f"Top {len(top_docs)} documents:")
for i, doc in enumerate(top_docs, 1):
    print(f"  {i}. {doc}")
print(f"Used: {top_docs.models_used[0].provider}/{top_docs.models_used[0].model}")

# Example 9: Streaming - just works
print("\n9. STREAMING (no requirements)")
print("-" * 70)
print("User: Wants real-time streaming")
print("Response: ", end='', flush=True)
for chunk in jugglerr.chat_stream([
    {"role": "user", "content": "Count from 1 to 5"}
]):
    print(chunk, end='', flush=True)
print()

# Example 10: Fast model for simple task
print("\n10. FAST MODEL (power='regular')")
print("-" * 70)
print("User: Simple task, wants fast response")
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Say hello"}],
    power="regular"  # Use 7B-32B models (faster)
)
print(f"Response: {response}")
print(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")

print("\n" + "=" * 70)
print("KEY TAKEAWAY:")
print("Users don't need to know about specific models or providers.")
print("Just specify WHAT you need (power, capabilities), not HOW.")
print("=" * 70)
