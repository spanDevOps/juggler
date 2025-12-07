"""Capability-based routing examples."""

from juggler import LLMJuggler, Capabilities

juggler = LLMJuggler(
    groq_keys=["gsk_..."],
    cerebras_keys=["csk_..."]
)

# Example 1: Vision request (routes to Llama 4 on Groq)
print("=== Vision Example ===")
response = juggler.juggle(
    messages=[{
        "role": "user",
        "content": "What's in this image? https://example.com/image.jpg"
    }],
    capabilities=[Capabilities.VISION]
)
print(response)

# Example 2: Reasoning request (routes to GPT-OSS or Qwen3)
print("\n=== Reasoning Example ===")
response = juggler.juggle(
    messages=[{
        "role": "user",
        "content": "Solve this step by step: If a train travels 120 km in 2 hours, what's its speed?"
    }],
    capabilities=[Capabilities.REASONING],
    power="super"
)
print(response)

# Example 3: Tool calling with structured outputs
print("\n=== Tool Calling Example ===")
response = juggler.juggle(
    messages=[{
        "role": "user",
        "content": "Extract the name and age from: 'John is 25 years old'"
    }],
    capabilities=[Capabilities.TOOL_CALLING, Capabilities.JSON_SCHEMA]
)
print(response)

# Example 4: Browser search (only on Groq GPT-OSS)
print("\n=== Browser Search Example ===")
response = juggler.juggle(
    messages=[{
        "role": "user",
        "content": "What's the latest news about AI?"
    }],
    capabilities=[Capabilities.BROWSER_SEARCH],
    preferred_provider="groq"
)
print(response)

# Example 5: Multilingual
print("\n=== Multilingual Example ===")
response = juggler.juggle(
    messages=[{
        "role": "user",
        "content": "Translate 'Hello, how are you?' to Spanish, French, and German"
    }],
    capabilities=[Capabilities.MULTILINGUAL]
)
print(response)
