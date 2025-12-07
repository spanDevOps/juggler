"""
Cohere Streaming Example

Demonstrates streaming with Cohere models using the newly implemented
JSON lines format support.

Cohere uses a different streaming format than other providers:
- JSON lines (not SSE)
- Event types: stream-start, text-generation, stream-end
- Text in 'text' field (not 'delta.content')
"""

from jugglerr import LLMJugglerr
import sys

def example_1_simple_streaming():
    """Example 1: Simple Cohere streaming."""
    print("="*60)
    print("Example 1: Simple Cohere Streaming")
    print("="*60)
    
    jugglerr = LLMJugglerr()
    
    messages = [
        {"role": "user", "content": "Write a haiku about streaming data."}
    ]
    
    print("\nStreaming response:")
    print("-" * 40)
    
    for chunk in jugglerr.juggle_stream(
        messages=messages,
        preferred_provider="cohere",
        preferred_model="command-r",
        temperature=0.7
    ):
        print(chunk, end='', flush=True)
    
    print("\n" + "-" * 40)
    print("✅ Streaming complete\n")


def example_2_conversation_streaming():
    """Example 2: Multi-turn conversation with streaming."""
    print("="*60)
    print("Example 2: Conversation with Streaming")
    print("="*60)
    
    jugglerr = LLMJugglerr()
    
    messages = [
        {"role": "user", "content": "What are the benefits of streaming?"},
        {"role": "assistant", "content": "Streaming provides immediate feedback, lower perceived latency, and better user experience."},
        {"role": "user", "content": "Give me 3 specific examples."}
    ]
    
    print("\nStreaming response:")
    print("-" * 40)
    
    for chunk in jugglerr.juggle_stream(
        messages=messages,
        preferred_provider="cohere",
        preferred_model="command-r-08-2024",
        temperature=0.5
    ):
        print(chunk, end='', flush=True)
    
    print("\n" + "-" * 40)
    print("✅ Streaming complete\n")


def example_3_collect_response():
    """Example 3: Collect full response while streaming."""
    print("="*60)
    print("Example 3: Collect Full Response")
    print("="*60)
    
    jugglerr = LLMJugglerr()
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    print("\nStreaming and collecting:")
    print("-" * 40)
    
    chunks = []
    for chunk in jugglerr.juggle_stream(
        messages=messages,
        preferred_provider="cohere",
        preferred_model="command",
        temperature=0.3
    ):
        print(chunk, end='', flush=True)
        chunks.append(chunk)
    
    full_response = ''.join(chunks)
    
    print("\n" + "-" * 40)
    print(f"\n📊 Statistics:")
    print(f"   Chunks received: {len(chunks)}")
    print(f"   Total characters: {len(full_response)}")
    print(f"   Full response: {full_response[:100]}...")
    print("✅ Collection complete\n")


def example_4_super_model_streaming():
    """Example 4: Streaming with Cohere's super model."""
    print("="*60)
    print("Example 4: Super Model Streaming")
    print("="*60)
    
    jugglerr = LLMJugglerr()
    
    messages = [
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ]
    
    print("\nStreaming with command-r-plus:")
    print("-" * 40)
    
    for chunk in jugglerr.juggle_stream(
        messages=messages,
        preferred_provider="cohere",
        preferred_model="command-r-plus-08-2024",
        temperature=0.7,
        max_tokens=150
    ):
        print(chunk, end='', flush=True)
    
    print("\n" + "-" * 40)
    print("✅ Streaming complete\n")


def example_5_error_handling():
    """Example 5: Error handling with streaming."""
    print("="*60)
    print("Example 5: Error Handling")
    print("="*60)
    
    jugglerr = LLMJugglerr()
    
    messages = [
        {"role": "user", "content": "Hello!"}
    ]
    
    try:
        print("\nAttempting to stream:")
        print("-" * 40)
        
        for chunk in jugglerr.juggle_stream(
            messages=messages,
            preferred_provider="cohere",
            temperature=0.7
        ):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 40)
        print("✅ Streaming successful\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("This is expected if no Cohere API key is configured\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("COHERE STREAMING EXAMPLES")
    print("="*60)
    print("\nCohere uses JSON lines format (not SSE)")
    print("All 5 Cohere models support streaming:")
    print("  - command-r-plus-08-2024 (super)")
    print("  - command-r-plus (super)")
    print("  - command-r-08-2024 (regular)")
    print("  - command-r (regular)")
    print("  - command (regular)")
    print()
    
    try:
        example_1_simple_streaming()
        example_2_conversation_streaming()
        example_3_collect_response()
        example_4_super_model_streaming()
        example_5_error_handling()
        
        print("="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("  1. Cohere API key in .env file (COHERE_API_KEYS)")
        print("  2. jugglerr package installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
