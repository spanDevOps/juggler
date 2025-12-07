"""
Embedding example using Juggler.
"""

from juggler import LLMJuggler

# Initialize Juggler with NVIDIA API key
juggler = LLMJuggler(
    nvidia_keys=["your-nvidia-api-key"]  # Or load from .env
)

# Example 1: Simple text embeddings
print("Example 1: Simple text embeddings")
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing deals with text"
]

embeddings = juggler.embed(texts)
print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
print()

# Example 2: Query embedding for search
print("Example 2: Query embedding for search")
query = "What is machine learning?"
query_embedding = juggler.embed_query(query)
print(f"Query embedding dimension: {len(query_embedding)}")
print()

# Example 3: Document embeddings for indexing
print("Example 3: Document embeddings for indexing")
documents = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "SQL is used for databases"
]

doc_embeddings = juggler.embed_documents(documents)
print(f"Generated {len(doc_embeddings)} document embeddings")
print()

# Example 4: Code embeddings
print("Example 4: Code embeddings")
code_snippet = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

code_embedding = juggler.embed(
    code_snippet,
    model="nvidia/nv-embedcode-7b-v1",
    input_type="query"
)
print(f"Code embedding dimension: {len(code_embedding[0])}")
print()

# Example 5: Multilingual embeddings
print("Example 5: Multilingual embeddings")
multilingual_texts = [
    "Hello, world!",
    "Bonjour le monde!",
    "Hola mundo!",
    "こんにちは世界",
    "Привет мир"
]

multilingual_embeddings = juggler.embed(
    multilingual_texts,
    model="baai/bge-m3"
)
print(f"Generated {len(multilingual_embeddings)} multilingual embeddings")
print(f"Embedding dimension: {len(multilingual_embeddings[0])}")
print()

print("✅ All examples complete!")
