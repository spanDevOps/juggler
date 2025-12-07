"""Multi-key rotation strategy example."""

from jugglerr import LLMJugglerr
import time

# Setup with multiple keys per provider
jugglerr = LLMJugglerr(
    groq_keys=[
        "gsk_key1",
        "gsk_key2",
        "gsk_key3",
        "gsk_key4",
        "gsk_key5"
    ],
    cerebras_keys=[
        "csk_key1",
        "csk_key2",
        "csk_key3"
    ]
)

# The jugglerr automatically:
# 1. Parses rate limit headers from each response
# 2. Tracks remaining requests/tokens per key
# 3. Selects the key with most capacity
# 4. Rotates to next key when one is exhausted

print("Making 50 requests with automatic key rotation...")
start_time = time.time()

for i in range(50):
    try:
        response = jugglerr.juggle([
            {"role": "user", "content": f"Request {i}: Quick fact"}
        ])
        print(f"✅ Request {i}: Success")
    except Exception as e:
        print(f"❌ Request {i}: Failed - {e}")

elapsed = time.time() - start_time
print(f"\nCompleted 50 requests in {elapsed:.2f} seconds")
print(f"Average: {elapsed/50:.2f} seconds per request")

# Key rotation benefits:
# 1. Maximize free tier usage (5 Groq keys = 72,000 requests/day)
# 2. Avoid rate limits (jugglerr knows which keys have capacity)
# 3. No manual key management needed
# 4. Automatic failover if one key fails

# Check key state (internal tracking)
print("\n=== Key State ===")
for key, state in jugglerr.key_state.items():
    print(f"{key}: {state.get('remaining_requests', 'unknown')} requests remaining")
