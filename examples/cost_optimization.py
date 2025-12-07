"""Cost optimization example - maximizing free tiers."""

from juggler import Juggler

# Strategy: Use multiple free-tier keys to maximize usage
juggler = Juggler(
    # Multiple Cerebras keys (free tier: 1M tokens/day per key)
    cerebras_keys=[
        "csk_key1",
        "csk_key2",
        "csk_key3",
    ],
    # Multiple Groq keys (free tier: 14,400 requests/day per key)
    groq_keys=[
        "gsk_key1",
        "gsk_key2",
    ],
    # Multiple NVIDIA keys (free tier)
    nvidia_keys=[
        "nvapi_key1",
        "nvapi_key2",
    ],
    # Paid providers as fallback
    mistral_keys=["mistral_key1"],
    cohere_keys=["cohere_key1"]
)

# The juggler will:
# 1. Try Cerebras first (free, fastest)
# 2. If rate limited, try next Cerebras key
# 3. If all Cerebras exhausted, try Groq
# 4. If all Groq exhausted, try NVIDIA
# 5. If all free tiers exhausted, try Mistral/Cohere (paid)

# Make many requests - juggler handles key rotation
for i in range(100):
    response = juggler.chat(f"Request {i}: Tell me a fact")
    print(f"Request {i}: {response[:50]}...")
    print(f"  Used: {response.models_used[0]['provider']}/{response.models_used[0]['model']}")

# Tips for maximizing free tier:
# 1. Use multiple API keys per provider
# 2. Let juggler handle rate limits automatically
# 3. Set paid providers (Mistral, Cohere) as fallback only
# 4. Use "regular" power for most requests (faster, cheaper)
# 5. Monitor your usage via provider dashboards
