"""Vision model usage example."""

from jugglerr import LLMJugglerr, Capabilities

jugglerr = LLMJugglerr(
    groq_keys=["gsk_..."]  # Vision only available on Groq
)

# Example 1: Analyze an image URL
print("=== Image Analysis ===")
response = jugglerr.juggle(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    capabilities=[Capabilities.VISION]
)
print(response)

# Example 2: Compare two images
print("\n=== Image Comparison ===")
response = jugglerr.juggle(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the differences between these two images?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
        ]
    }],
    capabilities=[Capabilities.VISION]
)
print(response)

# Example 3: OCR - Extract text from image
print("\n=== OCR Example ===")
response = jugglerr.juggle(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/document.jpg"}}
        ]
    }],
    capabilities=[Capabilities.VISION]
)
print(response)

# Note: Vision is only available on:
# - Llama 4 Maverick (meta-llama/llama-4-maverick-17b-128e-instruct)
# - Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct)
# Both on Groq only
