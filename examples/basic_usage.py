"""Basic usage example for Juggler."""

from juggler import Juggler

# Initialize with API keys (or auto-load from .env)
juggler = Juggler(
    cerebras_keys=["csk_..."],  # Replace with your keys
    groq_keys=["gsk_..."],
    nvidia_keys=["nvapi_..."],
    mistral_keys=["mistral_..."],  # Optional paid fallback
    cohere_keys=["cohere_..."]     # Optional paid fallback
)

# Or just auto-load from .env
juggler = Juggler()

# Simple request
response = juggler.chat("Hello, world! Tell me a joke.")
print("Response:", response)
print("Used:", response.models_used[0]['provider'], "/", response.models_used[0]['model'])

# Multi-turn conversation
response = juggler.chat([
    {"role": "user", "content": "What is the capital of France?"}
])
print("\nFirst response:", response)

# Continue conversation
response = juggler.chat([
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": response},
    {"role": "user", "content": "What's the population?"}
])
print("Second response:", response)
