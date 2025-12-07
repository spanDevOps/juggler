# Jugglerr User Guide

**Complete guide to using Jugglerr for chat, embeddings, reranking, TTS, and STT**

---

## Quick Start

```python
from jugglerr import Jugglerr

# Initialize (auto-loads API keys from .env)
jugglerr = Jugglerr()

# Chat
response = jugglerr.chat([
    {"role": "user", "content": "What is Python?"}
])
print(response)

# Embeddings
embeddings = jugglerr.embed(["Hello world", "Python is great"])

# Reranking
top_docs = jugglerr.rerank(
    query="What is AI?",
    documents=["Doc 1", "Doc 2", "Doc 3"],
    top_k=2
)

# Speech-to-Text
text = jugglerr.transcribe("audio.mp3")

# Text-to-Speech
audio = jugglerr.speak("Hello world", voice="Aria")
audio.write_to_file("hello.mp3")
```

---

## Installation & Setup

### 1. Install Jugglerr

```bash
pip install jugglerrr
```

### 2. Configure API Keys

Create a `.env` file in your project root:

```env
# Free providers (try these first!)
CEREBRAS_API_KEYS=csk-xxx,csk-yyy
GROQ_API_KEYS=gsk-xxx,gsk-yyy
NVIDIA_API_KEYS=nvapi-xxx

# Paid providers
MISTRAL_API_KEYS=mistral-xxx
COHERE_API_KEYS=co-xxx
```

**Note**: You can provide multiple comma-separated keys for automatic rotation.

### 3. Initialize Jugglerr

```python
from jugglerr import Jugglerr

# Auto-loads from .env
jugglerr = Jugglerr()

# Or pass keys explicitly
jugglerr = Jugglerr(
    cerebras_keys=["csk-xxx"],
    groq_keys=["gsk-xxx"],
    nvidia_keys=["nvapi-xxx"]
)
```

---

## 1. Chat Completion

### Basic Chat

```python
# Simple question
response = jugglerr.chat([
    {"role": "user", "content": "What is 2+2?"}
])
print(response)  # "4"

# Multi-turn conversation
response = jugglerr.chat([
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"}
])
print(response)  # "Your name is Alice"
```

### Power-Based Routing

```python
# Use small, fast models (default)
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Hello"}],
    power="regular"  # 7B-32B models
)

# Use large, smart models
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    power="super"  # 70B+ models
)
```

### Capability-Based Routing

```python
# Need vision?
response = jugglerr.chat(
    messages=[{
        "role": "user",
        "content": "What's in this image?",
        "images": ["base64_image_data"]
    }],
    capabilities=["vision"]
)

# Need tool calling?
response = jugglerr.chat(
    messages=[{"role": "user", "content": "What's the weather?"}],
    capabilities=["tool_calling"]
)

# Need reasoning?
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Solve this complex problem"}],
    capabilities=["reasoning"]
)
```

### Specific Model

```python
# Use a specific model (no fallback to other providers)
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Hello"}],
    preferred_model="llama-3.3-70b"  # Only uses Cerebras
)
```

### Streaming

```python
# Stream tokens in real-time
for chunk in jugglerr.chat_stream(
    messages=[{"role": "user", "content": "Tell me a story"}]
):
    print(chunk, end='', flush=True)
```

### Advanced Options

```python
response = jugglerr.chat(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,      # Creativity (0.0-2.0)
    max_tokens=500,       # Max response length
    power="super",        # Model size
    capabilities=["streaming", "tool_calling"],
    context_window="large",  # large/medium/small
    preferred_provider="groq"  # Try this provider first
)
```

---

## 2. Embeddings

Convert text to numerical vectors for semantic search, clustering, or similarity.

### Basic Embeddings

```python
# Single text
embeddings = jugglerr.embed("Hello world")
print(len(embeddings[0]))  # 4096 dimensions

# Multiple texts
embeddings = jugglerr.embed([
    "Python is great",
    "JavaScript is popular",
    "Rust is fast"
])
print(len(embeddings))  # 3 embeddings
```

### Search/Retrieval Optimized

```python
# Embed a search query
query_embedding = jugglerr.embed_query("What is Python?")

# Embed documents/passages
doc_embeddings = jugglerr.embed_documents([
    "Python is a programming language",
    "JavaScript runs in browsers",
    "Rust is a systems language"
])

# Now use these for similarity search in your vector database
```

### Specific Model

```python
# Use a specific embedding model
embeddings = jugglerr.embed(
    texts=["Hello world"],
    model="nvidia/nv-embed-v1"  # 4096 dims
)

# Or use Cohere/Mistral models
embeddings = jugglerr.embed(
    texts=["Hello world"],
    model="embed-v4.0"  # Cohere
)
```

### Provider Fallback

Jugglerr automatically tries providers in order:
1. **NVIDIA** (free) - `nv-embed-v1` (4096 dims)
2. **Cohere** (paid) - `embed-v4.0` (128K context)
3. **Mistral** (paid) - `mistral-embed`

---

## 3. Reranking

Rerank documents by relevance to improve RAG accuracy (~24% improvement).

### Basic Reranking

```python
# Rerank documents
documents = [
    "Python is a programming language",
    "The sky is blue",
    "Machine learning is a subset of AI"
]

top_docs = jugglerr.rerank(
    query="What is machine learning?",
    documents=documents,
    top_k=2  # Return top 2
)

print(top_docs)
# ["Machine learning is a subset of AI", "Python is a programming language"]
```

### Full RAG Pipeline

```python
# 1. Embed query
query = "What is machine learning?"
query_emb = jugglerr.embed_query(query)

# 2. Search vector database (pseudo-code)
# candidate_docs = vector_db.search(query_emb, top_k=20)

# 3. Rerank for better accuracy
candidate_docs = ["Doc 1", "Doc 2", "Doc 3", ...]  # From vector search
top_docs = jugglerr.rerank(
    query=query,
    documents=candidate_docs,
    top_k=3
)

# 4. Use in RAG
context = "\n".join(top_docs)
response = jugglerr.chat([
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
])
```

### Provider Fallback

Jugglerr automatically tries providers in order:
1. **NVIDIA** (free) - `llama-3.2-nv-rerankqa-1b-v2` (multilingual)
2. **Cohere** (paid) - `rerank-v3.5` (multilingual)

---

## 4. Speech-to-Text (Transcription)

Convert audio files to text.

### Basic Transcription

```python
# Transcribe audio file
text = jugglerr.transcribe("audio.mp3")
print(text)

# From file object
with open("audio.mp3", "rb") as f:
    text = jugglerr.transcribe(f)
```

### Advanced Options

```python
# Force specific language
text = jugglerr.transcribe(
    file="audio.mp3",
    language="en"  # en, es, fr, etc.
)

# Use faster model
text = jugglerr.transcribe(
    file="audio.mp3",
    model="whisper-large-v3-turbo"  # Faster than default
)

# More deterministic output
text = jugglerr.transcribe(
    file="audio.mp3",
    temperature=0.0  # Less random
)
```

### Provider Fallback

Currently only supports:
- **Groq** (free) - `whisper-large-v3`, `whisper-large-v3-turbo`

---

## 5. Text-to-Speech

Convert text to spoken audio.

### Basic TTS

```python
# Generate speech
audio = jugglerr.speak("Hello, how are you?")
audio.write_to_file("hello.mp3")

# Or save directly
jugglerr.speak(
    text="Hello world",
    output_file="hello.mp3"
)
```

### Different Voices

```python
# Female voices
audio = jugglerr.speak("Hello", voice="Aria")
audio = jugglerr.speak("Hello", voice="Freya")
audio = jugglerr.speak("Hello", voice="Mia")

# Male voices
audio = jugglerr.speak("Hello", voice="Clyde")
audio = jugglerr.speak("Hello", voice="Liam")
audio = jugglerr.speak("Hello", voice="Orion")

# Neutral voices
audio = jugglerr.speak("Hello", voice="River")
audio = jugglerr.speak("Hello", voice="Sky")
```

**Available voices**: Aria, Clyde, Deedee, Finn, Freya, Kai, Liam, Mia, Nova, Orion, River, Sky

### Arabic TTS

```python
# Arabic text-to-speech
audio = jugglerr.speak(
    text="مرحبا",
    model="playai-tts-arabic",
    voice="Laila"  # or "Majed"
)
audio.write_to_file("arabic.mp3")
```

### Provider Fallback

Currently only supports:
- **Groq** (free) - `playai-tts`, `playai-tts-arabic`

---

## Best Practices

### 1. Use .env for API Keys

Never hardcode API keys. Use a `.env` file:

```env
CEREBRAS_API_KEYS=csk-xxx,csk-yyy
GROQ_API_KEYS=gsk-xxx
```

### 2. Start with Free Providers

Jugglerr automatically tries free providers first (Cerebras, Groq, NVIDIA).

### 3. Use Power Levels Wisely

- `power="regular"` - Fast, cheap, good for simple tasks
- `power="super"` - Slower, more expensive, better for complex tasks

### 4. Specify Capabilities When Needed

```python
# Only use models that support vision
response = jugglerr.chat(
    messages=[...],
    capabilities=["vision"]
)
```

### 5. Use Streaming for Long Responses

```python
# Better UX for long responses
for chunk in jugglerr.chat_stream(messages=[...]):
    print(chunk, end='', flush=True)
```

### 6. Rerank for Better RAG

Always rerank after vector search for ~24% better accuracy.

---

## API Reference Summary

| Method | Purpose | Providers | Fallback |
|--------|---------|-----------|----------|
| `chat()` | Chat completion | All | ✅ Yes |
| `chat_stream()` | Streaming chat | All | ✅ Yes |
| `embed()` | Text embeddings | NVIDIA, Cohere, Mistral | ✅ Yes |
| `rerank()` | Document reranking | NVIDIA, Cohere | ✅ Yes |
| `transcribe()` | Speech-to-text | Groq | ✅ Yes |
| `speak()` | Text-to-speech | Groq | ✅ Yes |
