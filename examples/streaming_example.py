"""
Streaming Chat Example

Demonstrates real-time token streaming from LLM models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from jugglerr import LLMJugglerr
from dotenv import load_dotenv

load_dotenv()

# Initialize jugglerr
jugglerr = LLMJugglerr()

print("="*80)
print("STREAMING CHAT EXAMPLES")
print("="*80)

# Example 1: Simple streaming
print("\n1. Simple Streaming")
print("-" * 80)
print("User: Tell me a short story about a robot.")
print("Assistant: ", end='', flush=True)

try:
    for chunk in jugglerr.juggle_stream([
        {"role": "user", "content": "Tell me a short story about a robot learning to paint. Keep it under 100 words."}
    ]):
        print(chunk, end='', flush=True)
    print("\n")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 2: Streaming with temperature
print("\n2. Streaming with High Temperature (Creative)")
print("-" * 80)
print("User: Write a haiku about coding.")
print("Assistant: ", end='', flush=True)

try:
    for chunk in jugglerr.juggle_stream(
        messages=[{"role": "user", "content": "Write a haiku about coding."}],
        temperature=1.2
    ):
        print(chunk, end='', flush=True)
    print("\n")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 3: Streaming with conversation history
print("\n3. Streaming with Conversation History")
print("-" * 80)

conversation = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"}
]

print("User: What's the capital of France?")
print("Assistant: The capital of France is Paris.")
print("User: What's its population?")
print("Assistant: ", end='', flush=True)

try:
    for chunk in jugglerr.juggle_stream(messages=conversation):
        print(chunk, end='', flush=True)
    print("\n")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 4: Streaming with preferred provider
print("\n4. Streaming with Preferred Provider (Groq)")
print("-" * 80)
print("User: Explain quantum computing in one sentence.")
print("Assistant: ", end='', flush=True)

try:
    for chunk in jugglerr.juggle_stream(
        messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
        preferred_provider="groq"
    ):
        print(chunk, end='', flush=True)
    print("\n")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 5: Streaming with power selection
print("\n5. Streaming with Super Model")
print("-" * 80)
print("User: Write a complex algorithm explanation.")
print("Assistant: ", end='', flush=True)

try:
    for chunk in jugglerr.juggle_stream(
        messages=[{"role": "user", "content": "Explain the quicksort algorithm in simple terms."}],
        power="super"
    ):
        print(chunk, end='', flush=True)
    print("\n")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 6: Collecting streamed response
print("\n6. Collecting Streamed Response")
print("-" * 80)
print("User: Count from 1 to 5.")
print("Collecting chunks...")

try:
    chunks = []
    for chunk in jugglerr.juggle_stream(
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}]
    ):
        chunks.append(chunk)
    
    full_response = ''.join(chunks)
    print(f"✅ Collected {len(chunks)} chunks")
    print(f"Full response:\n{full_response}")
except Exception as e:
    print(f"❌ Error: {e}")

# Example 7: Streaming vs Non-Streaming Comparison
print("\n7. Streaming vs Non-Streaming Comparison")
print("-" * 80)

import time

# Non-streaming
print("Non-streaming (wait for complete response):")
start = time.time()
try:
    response = jugglerr.juggle([
        {"role": "user", "content": "Say hello in 5 different languages."}
    ])
    elapsed = time.time() - start
    print(f"Response: {response}")
    print(f"Time: {elapsed:.2f}s (waited for complete response)")
except Exception as e:
    print(f"❌ Error: {e}")

print("\nStreaming (tokens arrive immediately):")
start = time.time()
first_token_time = None
try:
    print("Response: ", end='', flush=True)
    for i, chunk in enumerate(jugglerr.juggle_stream([
        {"role": "user", "content": "Say hello in 5 different languages."}
    ])):
        if i == 0:
            first_token_time = time.time() - start
        print(chunk, end='', flush=True)
    
    elapsed = time.time() - start
    print(f"\nFirst token: {first_token_time:.2f}s")
    print(f"Total time: {elapsed:.2f}s")
except Exception as e:
    print(f"\n❌ Error: {e}")

# Example 8: Error Handling
print("\n8. Error Handling with Streaming")
print("-" * 80)

try:
    print("Attempting to stream with invalid configuration...")
    for chunk in jugglerr.juggle_stream(
        messages=[{"role": "user", "content": "Hello"}],
        preferred_provider="nonexistent"
    ):
        print(chunk, end='', flush=True)
except Exception as e:
    print(f"✅ Caught error: {e}")

print("\n" + "="*80)
print("STREAMING EXAMPLES COMPLETE")
print("="*80)

print("\nKey Benefits of Streaming:")
print("  ✅ Immediate feedback (first token arrives quickly)")
print("  ✅ Better user experience (progressive display)")
print("  ✅ Lower perceived latency")
print("  ✅ Can process tokens as they arrive")
print("\nUse Cases:")
print("  - Chat applications")
print("  - Real-time assistants")
print("  - Interactive demos")
print("  - Long-form content generation")
