# Models Used Tracking

**Transparency and debugging with automatic fallback tracking**

---

## Overview

Every Juggler response now includes a `models_used` attribute that tracks the complete fallback chain. This provides transparency into which providers and models were tried, which succeeded, and why others failed.

---

## Why This Matters

1. **Debugging**: See exactly which provider failed and why
2. **Transparency**: Know which model generated your response
3. **Monitoring**: Track provider reliability over time
4. **Cost tracking**: Know which paid providers were used
5. **Performance**: See response times for each attempt

---

## Response Objects

All Juggler methods return objects that:
- **Behave like their simple types** (string, list) for backward compatibility
- **Include `models_used` attribute** for tracking

### ChatResponse

```python
response = juggler.chat([{"role": "user", "content": "Hello"}])

# Behaves like a string
print(response)  # "Hello! How can I help you?"
print(len(response))  # 25
print(response.upper())  # "HELLO! HOW CAN I HELP YOU?"

# But also has models_used
print(response.models_used)
# [ModelAttempt(provider='cerebras', model='qwen-3-32b', success=True, ...)]
```

### EmbeddingResponse

```python
embeddings = juggler.embed(["Hello", "World"])

# Behaves like a list
print(len(embeddings))  # 2
print(len(embeddings[0]))  # 4096
for emb in embeddings:
    print(len(emb))

# But also has models_used and dimensions
print(embeddings.models_used)
print(embeddings.dimensions)  # 4096
```

### RerankResponse

```python
top_docs = juggler.rerank("query", ["doc1", "doc2", "doc3"], top_k=2)

# Behaves like a list
print(len(top_docs))  # 2
for doc in top_docs:
    print(doc)

# But also has models_used and scores
print(top_docs.models_used)
print(top_docs.scores)  # [0.95, 0.87]
```

### TranscriptionResponse

```python
text = juggler.transcribe("audio.mp3")

# Behaves like a string
print(text)
print(len(text))

# But also has models_used, language, duration
print(text.models_used)
print(text.language)  # "en"
print(text.duration)  # 45.2
```

### SpeechResponse

```python
audio = juggler.speak("Hello world")

# Has audio_data and write_to_file()
audio.write_to_file("output.mp3")
print(len(audio))  # Size in bytes

# Also has models_used
print(audio.models_used)
```

---

## ModelAttempt Structure

Each attempt in `models_used` contains:

```python
{
    "provider": "cerebras",           # Provider name
    "model": "qwen-3-32b",            # Model ID
    "success": True,                  # Whether this attempt succeeded
    "attempt": 1,                     # Attempt number (1, 2, 3, ...)
    "status_code": 200,               # HTTP status code
    "error": None,                    # Error message (if failed)
    "error_type": None,               # Error type (if failed)
    "elapsed_ms": 1250.5              # Response time in milliseconds
}
```

---

## Examples

### Example 1: Successful First Attempt

```python
response = juggler.chat([{"role": "user", "content": "Hello"}])

print(response.models_used)
# [
#     {
#         "provider": "cerebras",
#         "model": "qwen-3-32b",
#         "success": True,
#         "attempt": 1,
#         "status_code": 200,
#         "elapsed_ms": 1250.5
#     }
# ]
```

### Example 2: Fallback After Rate Limit

```python
response = juggler.chat([{"role": "user", "content": "Hello"}])

print(response.models_used)
# [
#     {
#         "provider": "cerebras",
#         "model": "qwen-3-32b",
#         "success": False,
#         "attempt": 1,
#         "status_code": 429,
#         "error": "Rate limited",
#         "error_type": "RateLimitError",
#         "elapsed_ms": 150.2
#     },
#     {
#         "provider": "groq",
#         "model": "llama-3.1-8b-instant",
#         "success": True,
#         "attempt": 2,
#         "status_code": 200,
#         "elapsed_ms": 890.3
#     }
# ]
```

### Example 3: Multiple Failures Before Success

```python
response = juggler.chat([{"role": "user", "content": "Hello"}])

print(response.models_used)
# [
#     {
#         "provider": "cerebras",
#         "model": "qwen-3-32b",
#         "success": False,
#         "attempt": 1,
#         "status_code": 429,
#         "error": "Rate limited"
#     },
#     {
#         "provider": "groq",
#         "model": "llama-3.1-8b-instant",
#         "success": False,
#         "attempt": 2,
#         "status_code": 500,
#         "error": "Internal server error"
#     },
#     {
#         "provider": "nvidia",
#         "model": "meta-llama/llama-3.1-8b-instruct",
#         "success": True,
#         "attempt": 3,
#         "status_code": 200,
#         "elapsed_ms": 2100.7
#     }
# ]
```

---

## Use Cases

### 1. Debugging Failed Requests

```python
try:
    response = juggler.chat([{"role": "user", "content": "Hello"}])
except Exception as e:
    print(f"All providers failed: {e}")
    # Check what was tried (if available)
```

### 2. Monitoring Provider Reliability

```python
responses = []
for i in range(100):
    response = juggler.chat([{"role": "user", "content": f"Test {i}"}])
    responses.append(response)

# Analyze which providers were used
from collections import Counter
providers_used = Counter(
    r.models_used[0].provider 
    for r in responses 
    if r.models_used and r.models_used[0].success
)
print(providers_used)
# Counter({'cerebras': 75, 'groq': 20, 'nvidia': 5})
```

### 3. Cost Tracking

```python
# Track which paid providers were used
paid_providers = {'mistral', 'cohere', 'openai'}

responses = []
for query in queries:
    response = juggler.chat([{"role": "user", "content": query}])
    responses.append(response)

# Count paid vs free
paid_count = sum(
    1 for r in responses
    if r.models_used and r.models_used[0].provider in paid_providers
)
free_count = len(responses) - paid_count

print(f"Free: {free_count}, Paid: {paid_count}")
```

### 4. Performance Analysis

```python
response = juggler.chat([{"role": "user", "content": "Hello"}])

# Check response time
if response.models_used:
    attempt = response.models_used[0]
    if attempt.success and attempt.elapsed_ms:
        print(f"Response time: {attempt.elapsed_ms:.0f}ms")
        print(f"Provider: {attempt.provider}")
```

### 5. Logging for Analytics

```python
import json
import logging

response = juggler.chat([{"role": "user", "content": "Hello"}])

# Log the complete trace
log_data = {
    "timestamp": "2025-12-07T12:00:00Z",
    "query": "Hello",
    "response": str(response),
    "models_used": [dict(attempt) for attempt in response.models_used]
}

logging.info(json.dumps(log_data))
```

---

## Backward Compatibility

The response objects are fully backward compatible:

```python
# Old code still works
response = juggler.chat([{"role": "user", "content": "Hello"}])
print(response)  # Works like a string

embeddings = juggler.embed(["text"])
print(len(embeddings))  # Works like a list

# New code can access models_used
print(response.models_used)
print(embeddings.models_used)
```

---

## Serialization

Convert to JSON for storage or API responses:

```python
import json

response = juggler.chat([{"role": "user", "content": "Hello"}])

# Serialize to JSON
data = {
    "response": str(response),
    "models_used": [dict(attempt) for attempt in response.models_used]
}

json_str = json.dumps(data, indent=2)
print(json_str)
```

---

## Best Practices

1. **Always check models_used for debugging**
   ```python
   if not response.models_used[0].success:
       print(f"Failed: {response.models_used[0].error}")
   ```

2. **Log models_used for production monitoring**
   ```python
   logger.info(f"Used: {response.models_used[0].provider}/{response.models_used[0].model}")
   ```

3. **Track fallback patterns**
   ```python
   if len(response.models_used) > 1:
       print(f"Fallback occurred: {len(response.models_used)} attempts")
   ```

4. **Monitor response times**
   ```python
   if response.models_used[0].elapsed_ms > 5000:
       logger.warning(f"Slow response: {response.models_used[0].elapsed_ms}ms")
   ```

5. **Alert on paid provider usage**
   ```python
   if response.models_used[0].provider in ['mistral', 'cohere']:
       alert_team(f"Using paid provider: {response.models_used[0].provider}")
   ```

---

## Summary

- ✅ All responses include `models_used` tracking
- ✅ Fully backward compatible (behaves like string/list)
- ✅ Detailed information about each attempt
- ✅ Easy to serialize to JSON
- ✅ Useful for debugging, monitoring, and cost tracking
