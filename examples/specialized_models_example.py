"""
Example: Using NVIDIA Specialized Models

This example demonstrates how to use NVIDIA's specialized models:
- Safety/Content Moderation
- Translation
- (Vision and TTS coming soon)
"""

import sys
sys.path.insert(0, '.')

import os
from dotenv import load_dotenv
from jugglerr.specialized_adapters import SafetyAdapter, TranslationAdapter

load_dotenv()


def safety_example():
    """Example: Content moderation with Llama Guard."""
    print("="*70)
    print("SAFETY & CONTENT MODERATION EXAMPLE")
    print("="*70)
    
    # Initialize adapter
    nvidia_key = os.getenv('NVIDIA_API_KEY') or os.getenv('NVIDIA_API_KEYS', '').split(',')[0].strip()
    
    if not nvidia_key:
        print("❌ ERROR: NVIDIA_API_KEY not found in environment")
        return
    
    safety = SafetyAdapter(api_key=nvidia_key)
    
    # Example 1: Check safe content
    print("\n[Example 1] Checking safe content")
    safe_text = "Welcome to our AI-powered application!"
    result = safety.check_safety(safe_text)
    
    print(f"Text: {safe_text}")
    print(f"Safe: {result['safe']}")
    print(f"Response: {result['content']}")
    
    # Example 2: Moderate user input
    print("\n[Example 2] Moderating user input")
    user_input = "This is a test message from a user."
    result = safety.moderate_content(user_input)
    
    print(f"User Input: {user_input}")
    print(f"Safe: {result['safe']}")
    
    if result['safe']:
        print("✅ Content approved - can be displayed")
    else:
        print("❌ Content flagged - should be filtered")
    
    # Example 3: Use in content pipeline
    print("\n[Example 3] Content moderation pipeline")
    
    user_messages = [
        "Hello, I need help with my account.",
        "Can you explain how machine learning works?",
        "Thank you for your assistance!"
    ]
    
    for msg in user_messages:
        result = safety.check_safety(msg)
        status = "✅" if result['safe'] else "❌"
        print(f"{status} {msg[:50]}...")


def translation_example():
    """Example: Multi-language translation."""
    print("\n" + "="*70)
    print("TRANSLATION EXAMPLE")
    print("="*70)
    
    # Initialize adapter
    nvidia_key = os.getenv('NVIDIA_API_KEY') or os.getenv('NVIDIA_API_KEYS', '').split(',')[0].strip()
    
    if not nvidia_key:
        print("❌ ERROR: NVIDIA_API_KEY not found in environment")
        return
    
    translator = TranslationAdapter(api_key=nvidia_key)
    
    # Example 1: Simple translation
    print("\n[Example 1] Simple translation")
    text = "Hello, how can I help you today?"
    result = translator.translate(text, target_language="Spanish")
    
    print(f"English: {text}")
    print(f"Spanish: {result['translation']}")
    
    # Example 2: Multiple languages
    print("\n[Example 2] Translate to multiple languages")
    text = "Welcome to our service"
    languages = ["Spanish", "French", "German", "Italian"]
    
    print(f"Original: {text}")
    for lang in languages:
        result = translator.translate(text, target_language=lang)
        print(f"{lang}: {result['translation']}")
    
    # Example 3: Batch translation
    print("\n[Example 3] Batch translation")
    phrases = [
        "Good morning",
        "Thank you",
        "You're welcome",
        "Goodbye"
    ]
    
    results = translator.batch_translate(phrases, target_language="French")
    
    print("English → French:")
    for i, result in enumerate(results):
        print(f"  {phrases[i]} → {result['translation']}")
    
    # Example 4: Reverse translation
    print("\n[Example 4] Reverse translation")
    spanish_text = "Hola, ¿cómo estás?"
    result = translator.translate(
        spanish_text,
        target_language="English",
        source_language="Spanish"
    )
    
    print(f"Spanish: {spanish_text}")
    print(f"English: {result['translation']}")


def combined_example():
    """Example: Combining safety and translation."""
    print("\n" + "="*70)
    print("COMBINED EXAMPLE: SAFETY + TRANSLATION")
    print("="*70)
    
    nvidia_key = os.getenv('NVIDIA_API_KEY') or os.getenv('NVIDIA_API_KEYS', '').split(',')[0].strip()
    
    if not nvidia_key:
        print("❌ ERROR: NVIDIA_API_KEY not found in environment")
        return
    
    safety = SafetyAdapter(api_key=nvidia_key)
    translator = TranslationAdapter(api_key=nvidia_key)
    
    print("\n[Use Case] Multilingual content moderation")
    print("Scenario: User submits content in English, needs to be safe and translated")
    
    # User input
    user_text = "I love using this application! It's very helpful."
    
    # Step 1: Check safety
    print(f"\n1. Original text: {user_text}")
    safety_result = safety.check_safety(user_text)
    print(f"2. Safety check: {'✅ Safe' if safety_result['safe'] else '❌ Unsafe'}")
    
    if safety_result['safe']:
        # Step 2: Translate to multiple languages
        print("3. Translating to multiple languages...")
        
        languages = ["Spanish", "French", "German"]
        for lang in languages:
            translation = translator.translate(user_text, target_language=lang)
            print(f"   {lang}: {translation['translation']}")
    else:
        print("3. ❌ Content blocked - not translated")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("NVIDIA SPECIALIZED MODELS - EXAMPLES")
    print("="*70)
    print("\nDemonstrating working specialized models:")
    print("- Safety/Moderation (meta/llama-guard-4-12b)")
    print("- Translation (nvidia/riva-translate-4b-instruct)")
    print()
    
    # Run examples
    safety_example()
    translation_example()
    combined_example()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("- Integrate into your application")
    print("- Add error handling for production")
    print("- Monitor API usage and rate limits")
    print("- Check docs for more specialized models as they become available")
    print()


if __name__ == "__main__":
    main()
