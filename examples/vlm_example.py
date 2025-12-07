"""
NVIDIA Vision Language Models (VLM) Example

Demonstrates using VLMs for vision tasks instead of specialized CV models.
"""
import os
from pathlib import Path
from juggler import VLMVisionAdapter

# Get API key
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEYS", "").split(",")[0]

if not api_key:
    print("❌ Please set NVIDIA_API_KEY environment variable")
    exit(1)

print("="*70)
print("NVIDIA Vision Language Models (VLM) Examples")
print("="*70)

# Initialize VLM adapter
vlm = VLMVisionAdapter(api_key=api_key)

# Example 1: Object Detection
print("\n1. Object Detection")
print("-" * 70)
try:
    result = vlm.detect_objects("test-images/street.jpg")
    print(f"Objects detected: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 2: OCR / Text Extraction
print("\n2. OCR / Text Extraction")
print("-" * 70)
try:
    result = vlm.extract_text("test-images/document.jpg")
    print(f"Extracted text: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 3: Image Description
print("\n3. Image Description")
print("-" * 70)
try:
    result = vlm.describe_image("test-images/scene.jpg", detail_level="detailed")
    print(f"Description: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 4: Count Objects
print("\n4. Count Objects")
print("-" * 70)
try:
    result = vlm.count_objects("test-images/parking.jpg", "cars")
    print(f"Count result: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 5: Visual Question Answering
print("\n5. Visual Question Answering")
print("-" * 70)
try:
    result = vlm.answer_question(
        "test-images/photo.jpg",
        "What is the weather like in this image?"
    )
    print(f"Answer: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 6: Custom Analysis
print("\n6. Custom Analysis")
print("-" * 70)
try:
    result = vlm.analyze_image(
        "test-images/product.jpg",
        "Analyze this product image. What is it? What are its key features? What condition is it in?"
    )
    print(f"Analysis: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 7: Compare Images
print("\n7. Compare Images")
print("-" * 70)
try:
    result = vlm.compare_images(
        "test-images/before.jpg",
        "test-images/after.jpg"
    )
    print(f"Comparison: {result}")
except Exception as e:
    print(f"Error: {e}")

# Example 8: Using Different VLM Models
print("\n8. Using Different VLM Models")
print("-" * 70)

# Llama 4 Scout (faster, smaller)
vlm_scout = VLMVisionAdapter(
    api_key=api_key,
    model="meta/llama-4-scout-17b-16e-instruct"
)

# Phi-3.5 Vision (compact)
vlm_phi = VLMVisionAdapter(
    api_key=api_key,
    model="microsoft/phi-3.5-vision-instruct"
)

try:
    result_scout = vlm_scout.describe_image("test-images/test.jpg", detail_level="brief")
    print(f"Llama 4 Scout: {result_scout}")
    
    result_phi = vlm_phi.describe_image("test-images/test.jpg", detail_level="brief")
    print(f"Phi-3.5 Vision: {result_phi}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("✅ VLM Examples Complete!")
print("="*70)
