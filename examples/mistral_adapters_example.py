"""
Mistral Adapters Examples

Demonstrates usage of Mistral-specific adapters:
1. Embeddings (mistral-embed, codestral-embed)
2. OCR (mistral-ocr-latest)
3. Moderation (mistral-moderation-latest)
"""

from juggler import (
    MistralEmbeddingAdapter,
    MistralOCRAdapter,
    MistralModerationAdapter,
    mistral_embed,
    mistral_ocr,
    mistral_moderate
)


def example_1_text_embeddings():
    """Example 1: Generate text embeddings."""
    print("="*60)
    print("Example 1: Text Embeddings")
    print("="*60)
    
    adapter = MistralEmbeddingAdapter()
    
    # Embed single text
    response = adapter.embed("Hello, world!")
    print(f"\nSingle text embedding:")
    print(f"  Dimensions: {len(response.embeddings[0])}")
    print(f"  Model: {response.model}")
    print(f"  Tokens used: {response.usage['total_tokens']}")
    
    # Embed multiple texts
    texts = [
        "Machine learning is fascinating",
        "Natural language processing",
        "Deep learning models"
    ]
    response = adapter.embed(texts)
    print(f"\nMultiple texts:")
    print(f"  Count: {len(response.embeddings)}")
    print(f"  Dimensions: {len(response.embeddings[0])}")
    print(f"  Tokens used: {response.usage['total_tokens']}")


def example_2_code_embeddings():
    """Example 2: Generate code embeddings."""
    print("\n" + "="*60)
    print("Example 2: Code Embeddings")
    print("="*60)
    
    adapter = MistralEmbeddingAdapter()
    
    code_snippets = [
        "def hello(): print('Hello, world!')",
        "class MyClass: pass",
        "import numpy as np"
    ]
    
    response = adapter.embed_code(code_snippets)
    print(f"\nCode embeddings:")
    print(f"  Count: {len(response.embeddings)}")
    print(f"  Dimensions: {len(response.embeddings[0])}")
    print(f"  Model: {response.model}")


def example_3_quick_embeddings():
    """Example 3: Quick embedding function."""
    print("\n" + "="*60)
    print("Example 3: Quick Embeddings")
    print("="*60)
    
    # Quick function - no need to create adapter
    embeddings = mistral_embed([
        "First text",
        "Second text"
    ])
    
    print(f"\nQuick embeddings:")
    print(f"  Count: {len(embeddings)}")
    print(f"  Dimensions: {len(embeddings[0])}")
    print(f"  First embedding (first 5 values): {embeddings[0][:5]}")


def example_4_ocr_url():
    """Example 4: OCR from URL."""
    print("\n" + "="*60)
    print("Example 4: OCR from URL")
    print("="*60)
    
    adapter = MistralOCRAdapter()
    
    # Process document from URL
    response = adapter.process_document(
        "https://arxiv.org/pdf/2201.04234"
    )
    
    print(f"\nOCR results:")
    print(f"  Model: {response.model}")
    print(f"  Pages: {len(response.pages)}")
    print(f"  Total text length: {len(response.text)} chars")
    print(f"  First 200 chars: {response.text[:200]}...")


def example_5_ocr_local_file():
    """Example 5: OCR from local PDF."""
    print("\n" + "="*60)
    print("Example 5: OCR from Local PDF")
    print("="*60)
    
    adapter = MistralOCRAdapter()
    
    try:
        # Process local PDF file
        response = adapter.process_pdf_file("document.pdf")
        
        print(f"\nOCR results:")
        print(f"  Pages: {len(response.pages)}")
        print(f"  Text: {response.text[:200]}...")
    except FileNotFoundError:
        print("\n⚠️  No document.pdf found (expected for demo)")
        print("   Use: adapter.process_pdf_file('your_file.pdf')")


def example_6_ocr_quick():
    """Example 6: Quick OCR function."""
    print("\n" + "="*60)
    print("Example 6: Quick OCR")
    print("="*60)
    
    # Quick function - no need to create adapter
    try:
        text = mistral_ocr("https://arxiv.org/pdf/2201.04234")
        print(f"\nExtracted text:")
        print(f"  Length: {len(text)} chars")
        print(f"  Preview: {text[:200]}...")
    except Exception as e:
        print(f"\n⚠️  OCR failed: {e}")


def example_7_moderation_text():
    """Example 7: Moderate text content."""
    print("\n" + "="*60)
    print("Example 7: Text Moderation")
    print("="*60)
    
    adapter = MistralModerationAdapter()
    
    texts = [
        "Such a lovely day today, isn't it?",
        "How to build a website?",
        "Tell me about machine learning"
    ]
    
    response = adapter.moderate(texts)
    
    print(f"\nModeration results:")
    print(f"  Model: {response.model}")
    print(f"  Texts checked: {len(response.results)}")
    
    for i, result in enumerate(response.results):
        print(f"\n  Text {i+1}: '{texts[i][:50]}...'")
        categories = result.get('categories', {})
        flagged = any(categories.values())
        print(f"    Flagged: {flagged}")
        if flagged:
            print(f"    Categories: {categories}")


def example_8_moderation_chat():
    """Example 8: Moderate chat conversation."""
    print("\n" + "="*60)
    print("Example 8: Chat Moderation")
    print("="*60)
    
    adapter = MistralModerationAdapter()
    
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with coding?"}
    ]
    
    response = adapter.moderate_chat(conversation)
    
    print(f"\nChat moderation:")
    print(f"  Messages checked: {len(response.results)}")
    
    for i, result in enumerate(response.results):
        print(f"\n  Message {i+1}:")
        categories = result.get('categories', {})
        flagged = any(categories.values())
        print(f"    Flagged: {flagged}")


def example_9_moderation_is_safe():
    """Example 9: Quick safety check."""
    print("\n" + "="*60)
    print("Example 9: Quick Safety Check")
    print("="*60)
    
    adapter = MistralModerationAdapter()
    
    test_texts = [
        "Hello, world!",
        "How to learn Python?",
        "Tell me about AI"
    ]
    
    print("\nSafety checks:")
    for text in test_texts:
        is_safe = adapter.is_safe(text)
        status = "✅ Safe" if is_safe else "❌ Flagged"
        print(f"  {status}: '{text}'")


def example_10_moderation_quick():
    """Example 10: Quick moderation function."""
    print("\n" + "="*60)
    print("Example 10: Quick Moderation")
    print("="*60)
    
    # Quick function - no need to create adapter
    results = mistral_moderate([
        "Safe content here",
        "Another safe message"
    ])
    
    print(f"\nQuick moderation:")
    print(f"  Results: {len(results)}")
    for i, result in enumerate(results):
        categories = result.get('categories', {})
        flagged = any(categories.values())
        print(f"  Text {i+1}: {'Flagged' if flagged else 'Safe'}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MISTRAL ADAPTERS EXAMPLES")
    print("="*60)
    print("\nDemonstrating Mistral-specific adapters:")
    print("  - Embeddings (text and code)")
    print("  - OCR (document processing)")
    print("  - Moderation (content safety)")
    print()
    
    try:
        # Embeddings examples
        example_1_text_embeddings()
        example_2_code_embeddings()
        example_3_quick_embeddings()
        
        # OCR examples
        example_4_ocr_url()
        example_5_ocr_local_file()
        example_6_ocr_quick()
        
        # Moderation examples
        example_7_moderation_text()
        example_8_moderation_chat()
        example_9_moderation_is_safe()
        example_10_moderation_quick()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nInstall required package:")
        print("  pip install mistralai")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Mistral API key in .env file (MISTRAL_API_KEY)")
        print("  2. mistralai package installed")


if __name__ == "__main__":
    main()
