#!/usr/bin/env python3
"""
Quick start example for Juggler.

Shows how to use chat, embeddings, reranking, TTS, and STT.
"""

from juggler import Juggler

# Initialize (auto-loads from .env)
juggler = Juggler()

print("=" * 70)
print("JUGGLER QUICK START")
print("=" * 70)

# 1. Chat
print("\n1. CHAT")
print("-" * 70)
response = juggler.chat([
    {"role": "user", "content": "What is 2+2? Answer in one sentence."}
])
print(f"Response: {response}")

# 2. Streaming Chat
print("\n2. STREAMING CHAT")
print("-" * 70)
print("Response: ", end='', flush=True)
for chunk in juggler.chat_stream([
    {"role": "user", "content": "Count from 1 to 5, separated by commas."}
]):
    print(chunk, end='', flush=True)
print()

# 3. Embeddings
print("\n3. EMBEDDINGS")
print("-" * 70)
embeddings = juggler.embed(["Hello world", "Python is great"])
print(f"Generated {len(embeddings)} embeddings")
print(f"Dimensions: {len(embeddings[0])}")

# 4. Reranking
print("\n4. RERANKING")
print("-" * 70)
documents = [
    "Python is a programming language",
    "The sky is blue",
    "Machine learning is a subset of AI",
    "JavaScript runs in browsers"
]
query = "What is machine learning?"
print(f"Query: {query}")
print(f"Documents: {len(documents)}")

top_docs = juggler.rerank(
    query=query,
    documents=documents,
    top_k=2
)
print(f"Top 2 results:")
for i, doc in enumerate(top_docs, 1):
    print(f"  {i}. {doc}")

# 5. Speech-to-Text (if you have an audio file)
print("\n5. SPEECH-TO-TEXT")
print("-" * 70)
print("Skipped (requires audio file)")
# text = juggler.transcribe("audio.mp3")
# print(f"Transcription: {text}")

# 6. Text-to-Speech
print("\n6. TEXT-TO-SPEECH")
print("-" * 70)
print("Generating speech...")
audio = juggler.speak(
    text="Hello, this is a test of text to speech.",
    voice="Aria"
)
print(f"Generated {len(audio.audio_data)} bytes of audio")
# audio.write_to_file("output.mp3")
print("(Audio not saved - uncomment to save)")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
