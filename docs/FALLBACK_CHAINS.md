# Juggler Fallback Chains

**Complete mapping of provider fallback logic for all request types**

---

## Overview

Juggler automatically tries multiple providers in order until one succeeds. This document maps out the exact fallback chain for each type of request.

---

## 1. Chat Completion

### Default Priority (No Specific Requirements)

```
1. Cerebras (free)
   ├─ Models: qwen-3-32b, llama3.1-8b, qwen-3-235b, gpt-oss-120b, llama-3.3-70b, zai-glm-4.6
   └─ Fallback reason: Rate limit, API error, no keys

2. Groq (free)
   ├─ Models: llama-3.1-8b-instant, qwen/qwen3-32b, llama-3.3-70b-versatile, etc.
   └─ Fallback reason: Rate limit, API error, no keys

3. NVIDIA (free)
   ├─ Models: 56+ models including specialized ones
   └─ Fallback reason: Rate limit, API error, no keys

4. Mistral (paid)
   ├─ Models: mistral-small-latest, ministral-8b-latest, mistral-large-latest, etc.
   └─ Fallback reason: Rate limit, API error, no keys

5. Cohere (paid)
   ├─ Models: command-r, command-r-plus, command-r7b-12-2024
   └─ Fallback reason: Rate limit, API error, no keys

6. OpenAI (paid) - if configured
   ├─ Models: gpt-4, gpt-3.5-turbo, etc.
   └─ Final fallback
```

### With Capabilities Filter

#### Vision Capability

```
1. Groq (free)
   ├─ Models: llama-4-maverick-17b-128e-instruct, llama-4-scout-17b-16e-instruct
   └─ Native multimodal support

2. Mistral (paid)
   ├─ Models: mistral-medium-latest, pixtral-large-latest
   └─ Frontier vision models

3. NVIDIA (free)
   ├─ Models: microsoft/phi-3.5-vision-instruct, minimaxai/minimax-m2, marin/marin-8b-instruct
   └─ Specialized vision models
```

#### Reasoning Capability

```
1. Cerebras (free)
   ├─ Models: zai-glm-4.6, qwen-3-235b, qwen-3-32b, gpt-oss-120b
   └─ Dual-mode reasoning, thinking models

2. Groq (free)
   ├─ Models: openai/gpt-oss-120b, openai/gpt-oss-20b, groq/compound, groq/compound-mini
   └─ Reasoning + built-in tools

3. Mistral (paid)
   ├─ Models: magistral-medium-latest, magistral-small-latest, mistral-medium-latest
   └─ Transparent reasoning
```

#### Tool Calling Capability

```
1. Cerebras (free)
   ├─ Most models support tool calling
   └─ Note: llama-3.3-70b lacks multi-turn tool calling

2. Groq (free)
   ├─ Most models support tool calling
   └─ Some with parallel tool calling

3. NVIDIA (free)
   ├─ Many models support tool calling
   └─ Varies by model

4. Mistral (paid)
   ├─ Most models support tool calling
   └─ Some with structured outputs

5. Cohere (paid)
   ├─ All command models support tool calling
   └─ Advanced tool use
```

### With Power Filter

#### Power: "super" (70B+ models)

```
1. Cerebras (free)
   ├─ zai-glm-4.6 (357B total, 32B active)
   ├─ qwen-3-235b-a22b-instruct-2507 (235B)
   ├─ gpt-oss-120b (120B)
   └─ llama-3.3-70b (70B)

2. Groq (free)
   ├─ moonshotai/kimi-k2-instruct (trillion-param MoE, 32B active)
   ├─ openai/gpt-oss-120b (120B)
   └─ llama-3.3-70b-versatile (70B)

3. NVIDIA (free)
   ├─ deepseek-ai/deepseek-v3.1 (large)
   ├─ stockmark/stockmark-2-100b-instruct (100B)
   └─ abacusai/dracarys-llama-3.1-70b-instruct (70B)

4. Mistral (paid)
   ├─ mistral-large-latest (123B)
   ├─ mistral-medium-latest (~100B)
   ├─ pixtral-large-latest (~100B)
   ├─ magistral-medium-latest (70B+)
   └─ devstral-medium-latest (~70B)

5. Cohere (paid)
   └─ command-r-plus (large)
```

#### Power: "regular" (7B-32B models)

```
1. Cerebras (free)
   ├─ qwen-3-32b (32B)
   └─ llama3.1-8b (8B)

2. Groq (free)
   ├─ qwen/qwen3-32b (32B)
   ├─ llama-4-maverick-17b-128e-instruct (17B)
   ├─ llama-4-scout-17b-16e-instruct (17B)
   ├─ openai/gpt-oss-20b (20B)
   └─ llama-3.1-8b-instant (8B)

3. NVIDIA (free)
   ├─ 40+ models in 7B-32B range
   └─ Various specialized models

4. Mistral (paid)
   ├─ mistral-small-latest (24B)
   ├─ codestral-latest (32B)
   ├─ magistral-small-latest (24B)
   ├─ ministral-14b-latest (14B)
   ├─ ministral-8b-latest (8B)
   └─ ministral-3b-latest (3B)

5. Cohere (paid)
   └─ command-r (regular), command-r7b-12-2024 (7B)
```

---

## 2. Embeddings

### Default Fallback Chain

```
1. NVIDIA (free)
   ├─ Default: nvidia/nv-embed-v1 (4096 dims)
   ├─ Alternative: nvidia/bge-m3 (1024 dims, multilingual)
   ├─ Code: nvidia/nv-embedcode-7b-v1
   ├─ Retrieval: nvidia/nemoretriever-embedding-v1
   ├─ Vision: nvidia/nv-dinov2
   └─ Protein: nvidia/esm2-650m

2. Cohere (paid)
   ├─ Default: embed-v4.0 (128K context, text+images)
   ├─ English: embed-english-v3.0
   ├─ Multilingual: embed-multilingual-v3.0
   ├─ Fast English: embed-english-light-v3.0
   └─ Fast Multilingual: embed-multilingual-light-v3.0

3. Mistral (paid)
   ├─ Text: mistral-embed
   └─ Code: codestral-embed
```

### By Use Case

#### General Text Embeddings
```
NVIDIA (nv-embed-v1) → Cohere (embed-v4.0) → Mistral (mistral-embed)
```

#### Code Embeddings
```
NVIDIA (nv-embedcode-7b-v1) → Mistral (codestral-embed) → Cohere (embed-v4.0)
```

#### Multilingual Embeddings
```
NVIDIA (bge-m3) → Cohere (embed-multilingual-v3.0) → Mistral (mistral-embed)
```

#### Retrieval-Optimized
```
NVIDIA (nemoretriever-embedding-v1) → Cohere (embed-v4.0 with input_type) → Mistral (mistral-embed)
```

---

## 3. Reranking

### Default Fallback Chain

```
1. NVIDIA (free)
   ├─ Model: nvidia/llama-3.2-nv-rerankqa-1b-v2
   ├─ Features: Multilingual, 24% accuracy improvement
   └─ Fallback reason: Rate limit, API error, no keys

2. Cohere (paid)
   ├─ Default: rerank-v3.5 (multilingual, latest)
   ├─ Alternative: rerank-v3.0
   ├─ English: rerank-english-v3.0
   └─ Multilingual: rerank-multilingual-v3.0
```

### By Use Case

#### Multilingual Reranking
```
NVIDIA (llama-3.2-nv-rerankqa-1b-v2) → Cohere (rerank-v3.5)
```

#### English-Only Reranking
```
NVIDIA (llama-3.2-nv-rerankqa-1b-v2) → Cohere (rerank-english-v3.0)
```

---

## 4. Speech-to-Text (Transcription)

### Default Fallback Chain

```
1. Groq (free)
   ├─ Default: whisper-large-v3 (high quality)
   ├─ Fast: whisper-large-v3-turbo (faster inference)
   └─ Fallback reason: Rate limit, API error, no keys

(Currently only Groq supported - future: add more providers)
```



---

## 5. Text-to-Speech

### Default Fallback Chain

```
1. Groq (free)
   ├─ English: playai-tts
   │   └─ Voices: Aria, Clyde, Deedee, Finn, Freya, Kai, Liam, Mia, Nova, Orion, River, Sky
   ├─ Arabic: playai-tts-arabic
   │   └─ Voices: Laila, Majed
   └─ Fallback reason: Rate limit, API error, no keys

2. NVIDIA (free) - PLANNED
   ├─ Models: Various TTS models available
   └─ To be implemented

(Currently only Groq supported - NVIDIA TTS exists but not yet integrated)
```



---

## 6. Special Cases

### Specific Model Requested

When a user requests a specific model via `preferred_model`:

```
NO FALLBACK - Only tries the provider that owns that model

Example:
juggler.chat(preferred_model="llama-3.3-70b")
└─ Only tries Cerebras (no fallback to Groq/Mistral/etc.)

Reason: User explicitly wants THAT model, not a similar one
```

### Preferred Provider

When a user specifies `preferred_provider`:

```
Tries preferred provider first, then falls back to normal priority

Example:
juggler.chat(preferred_provider="groq")
└─ Priority: [groq, cerebras, nvidia, mistral, cohere, openai]
   (groq moved to front)
```

---

## 7. Fallback Triggers

### What Causes Fallback?

1. **Rate Limit (429)**
   - Provider returns 429 status
   - Juggler tries next provider immediately

2. **API Error (4xx, 5xx)**
   - Provider returns error status
   - Juggler tries next provider

3. **No API Keys**
   - Provider has no configured keys
   - Juggler skips to next provider

4. **Timeout**
   - Request takes too long
   - Juggler tries next provider

5. **No Matching Model**
   - Provider has no model matching requirements
   - Juggler skips to next provider

### What Does NOT Cause Fallback?

1. **Specific Model Requested**
   - User wants `preferred_model="llama-3.3-70b"`
   - Only tries Cerebras, no fallback

2. **Successful Response**
   - Provider returns 200 with valid response
   - No fallback needed

---

## 8. Response Tracking

### models_used Field

Every Juggler response includes `models_used` tracking:

```python
{
    "content": "The actual response",
    "models_used": [
        {
            "provider": "cerebras",
            "model": "qwen-3-32b",
            "success": False,
            "error": "Rate limited",
            "status_code": 429,
            "attempt": 1
        },
        {
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "success": True,
            "status_code": 200,
            "attempt": 2
        }
    ]
}
```

This helps users:
- Debug issues
- Understand which provider was used
- See the fallback path taken
- Monitor provider reliability

---

## Summary Table

| Request Type | Primary (Free) | Secondary (Free) | Tertiary (Paid) | Fallback Count |
|--------------|----------------|------------------|-----------------|----------------|
| **Chat** | Cerebras | Groq, NVIDIA | Mistral, Cohere | 5+ providers |
| **Embeddings** | NVIDIA | - | Cohere, Mistral | 3 providers |
| **Reranking** | NVIDIA | - | Cohere | 2 providers |
| **Transcription** | Groq | - | - | 1 provider |
| **TTS** | Groq | NVIDIA (planned) | - | 1-2 providers |

---

## Best Practices

1. **Let Juggler Choose**: Don't specify `preferred_model` unless you need that exact model
2. **Use Capabilities**: Specify `capabilities=["vision"]` instead of picking a model
3. **Use Power Levels**: Use `power="super"` for complex tasks, `power="regular"` for simple ones
4. **Check models_used**: Review the fallback path to understand provider behavior
5. **Configure Multiple Keys**: Add multiple API keys per provider for better reliability
