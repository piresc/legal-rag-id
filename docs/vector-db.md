# Vector Database Service

## Overview

The Vector Database service provides efficient document storage and retrieval capabilities using both traditional keyword search (TF-IDF) and modern semantic search. It is specifically optimized for Indonesian legal documents.

**Service Port**: 8003
**Container Name**: `legal-rag-vector-db`

## Features

### Hybrid Search Approach

The service combines two search methods:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Keyword-based search
   - Optimized for Indonesian language with custom stop words
   - N-gram support (1-3 word phrases)
   - Fast and interpretable results

2. **Semantic Search**
   - Uses `sentence-transformers` with multilingual model
   - Model: `paraphrase-multilingual-MiniLM-L12-v2`
   - Understands meaning beyond exact keywords
   - Better for conceptual queries

### Combined Scoring

Search results use weighted combination:
- **60%** Semantic similarity (when available)
- **40%** TF-IDF score

This ensures both relevance and keyword matching are considered.

## Architecture

### Data Structures

```python
@dataclass
class DocumentChunk:
    chunk_id: str              # Unique identifier
    content: str               # Document text
    metadata: Dict[str, Any]   # Document metadata
    embedding: List[float]     # Semantic embedding (optional)
    doc_id: str                # Source document ID
    level: str                 # Hierarchy level
```

### Metadata Structure

Each document chunk includes:
- `doc_type`: Type of legal document (UU, PP, Perpres, etc.)
- `doc_number`: Document number
- `doc_year`: Year of enactment
- `article_number`: Article/pasal number
- `title`: Document title
- `url`: Source URL (if applicable)

## API Endpoints

### Health Check

```http
GET /health
```

**Response**: `200 OK` if service is healthy

### Search

```http
GET /search?query={query}&top_k={top_k}
```

**Parameters**:
- `query` (string, required): Search query in Indonesian
- `top_k` (integer, optional): Number of results (default: 5)
- `filters` (object, optional): Metadata filters

**Example Request**:
```bash
curl "http://localhost:8003/search?query=pasal%20korupsi&top_k=5"
```

**Example Response**:
```json
{
  "results": [
    {
      "chunk": {
        "chunk_id": "chunk_12345",
        "doc_id": "doc_001",
        "content": "Setiap orang yang... tindak pidana korupsi...",
        "metadata": {
          "doc_type": "UU",
          "doc_number": "31",
          "doc_year": "1999",
          "article_number": "2"
        }
      },
      "score": 0.892,
      "rank": 1
    }
  ]
}
```

### Statistics

```http
GET /stats
```

**Response**:
```json
{
  "total_chunks": 15000,
  "total_documents": 45,
  "vocabulary_size": 8500,
  "index_type": "tf-idf + semantic",
  "last_updated": "2024-01-15T10:30:00Z",
  "memory_usage_mb": 256.5
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `/app/data/vectors/vector_database.pkl` | Path to vector database file |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `USE_SEMANTIC_SEARCH` | `true` | Enable semantic search |
| `REDIS_URL` | - | Redis connection URL |

### Indonesian Stop Words

The TF-IDF vectorizer includes common Indonesian stop words:

```
dan, di, ke, dari, yang, untuk, dalam, pada,
adalah, ini, itu, dengan, akan, bagi, serta,
atau, tetapi, melainkan, yaitu, ialah, yakni
```

## Usage Examples

### Adding Documents

```python
from shared.src.interfaces.vector_db_interface import (
    EnhancedVectorDatabase,
    DocumentChunk
)

# Initialize database
db = EnhancedVectorDatabase(use_semantic=True)
await db.load("data/vectors/vector_database.pkl")

# Add document chunks
chunks = [
    DocumentChunk(
        chunk_id="chunk_001",
        content="Tindak pidana korupsi...",
        metadata={
            "doc_type": "UU",
            "doc_number": "31",
            "doc_year": "1999"
        }
    )
]

await db.add_chunks(chunks)
await db.save("data/vectors/vector_database.pkl")
```

### Searching Documents

```python
# Search for relevant documents
results = await db.search(
    query="apa itu tindak pidana korupsi?",
    top_k=5,
    filters={"doc_type": "UU"}  # Optional filter
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.chunk.content[:100]}...")
    print(f"Source: {result.chunk.metadata}")
```

### Using the Client

```python
from shared.src.interfaces.vector_db_interface import VectorDatabaseClient

async with VectorDatabaseClient("http://localhost:8003") as client:
    results = await client.search(
        query="pasal korupsi",
        top_k=5
    )
```

## Performance Considerations

### Memory Usage

- TF-IDF Matrix: ~100-500 MB (depends on corpus size)
- Semantic Embeddings: ~200-1000 MB (depends on corpus size)
- Total: ~300-1500 MB for typical legal corpus

### Search Speed

- TF-IDF Search: ~10-50ms
- Semantic Search: ~50-200ms
- Combined: ~60-250ms

### Scaling

For larger deployments (>100K documents):
1. Consider dedicated vector database (Weaviate, Milvus, Pinecone)
2. Implement pagination for search results
3. Add query result caching
4. Use GPU for semantic search acceleration

## Data Persistence

The database is persisted as a pickle file containing:
- Document chunks
- Chunk index
- TF-IDF vectorizer
- TF-IDF matrix
- Semantic embeddings (if enabled)

### Backup

```bash
# Backup database
cp data/vectors/vector_database.pkl backups/vector_db_$(date +%Y%m%d).pkl
```

### Restore

```python
db = EnhancedVectorDatabase()
await db.load("backups/vector_db_20240115.pkl")
```

## Troubleshooting

### High Memory Usage

**Symptom**: Service using >2GB memory

**Solutions**:
1. Disable semantic search: `USE_SEMANTIC_SEARCH=false`
2. Reduce TF-IDF features: Set `max_features=5000`
3. Increase chunk size: Fewer, larger chunks

### Slow Search Performance

**Symptom**: Search takes >500ms

**Solutions**:
1. Reduce `top_k` parameter
2. Add Redis caching
3. Use search filters to reduce search space
4. Consider semantic search only (skip TF-IDF)

### "Database Not Loaded" Error

**Symptom**: Searches return empty results

**Solutions**:
1. Check database file exists at `DATABASE_PATH`
2. Verify file permissions
3. Check logs for loading errors
4. Ensure `vector_database.pkl` is properly formatted

## Future Enhancements

- [ ] Support for ANN (Approximate Nearest Neighbor) indexing
- [ ] Multi-language support beyond Indonesian
- [ ] Real-time document updates
- [ ] Distributed search across multiple instances
- [ ] Hybrid search with customizable weight ratios
- [ ] Query expansion and spell correction

## References

- **Implementation**: [`shared/src/interfaces/vector_db_interface.py`](../ai-agent/shared/src/interfaces/vector_db_interface.py)
- **Dockerfile**: [`docker/Dockerfile.vector_db`](../ai-agent/docker/Dockerfile.vector_db)
- **Service Config**: [`docker-compose.yml`](../ai-agent/docker-compose.yml)
