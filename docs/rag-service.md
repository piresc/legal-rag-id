# RAG Service

## Overview

The RAG (Retrieval-Augmented Generation) Service is the core orchestration layer that combines document retrieval with AI generation. It coordinates between the Vector Database, LLM Service, and Cache Service to provide intelligent responses to legal queries.

**Service Port**: 8001
**Container Name**: `legal-rag-rag-service`

## What is RAG?

RAG enhances AI responses by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the AI prompt with retrieved context
3. **Generating** responses grounded in actual documents

This approach provides:
- ✅ Accurate, factual responses
- ✅ Source citations and references
- ✅ Up-to-date information (depends on your corpus)
- ✅ Reduced hallucinations (AI making things up)

## Architecture

### Query Processing Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ 1. Validate     │
│    Query        │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────────┐
│2. Cache│ │3. Retrieve │
│  Check │ │  Documents │
└───┬────┘ └─────┬──────┘
    │            │
    │    ┌───────┴────────┐
    │    │                │
    ▼    ▼                ▼
  Hit?   Vector DB    Semantic Search
    │    │    + TF-IDF
    │    └────────┬───────┘
    │             │
    │    ┌────────┴────────┐
    │    │                 │
    │    ▼                 ▼
    │  4. Prepare    5. Generate
    │     Context      Response
    │    │    │           │
    │    └────┴───────────┘
    │              │
    ▼              ▼
  Return     6. Cache
  Result      Result
```

## Query Modes

### 1. Basic Mode

**Description**: Returns raw search results without LLM generation

**Use Cases**:
- Document discovery
- Source verification
- Quick reference lookup

**Example**:
```python
request = QueryRequest(
    question="pasal korupsi",
    mode=QueryMode.BASIC,
    top_k=5
)
```

**Response**:
```json
{
  "answer": "Saya menemukan beberapa dokumen yang relevan:\n\n1. UU Nomor 31 Tahun 1999...",
  "sources": [...],
  "confidence": 0.85
}
```

### 2. API Mode (Default)

**Description**: Uses LLM to generate natural language response with retrieved context

**Use Cases**:
- Natural language questions
- Explanations and summaries
- Complex legal analysis

**Example**:
```python
request = QueryRequest(
    question="apa perbedaan antara korupsi dan kolusi?",
    mode=QueryMode.API,
    top_k=5,
    provider="anthropic"
)
```

**Response**:
```json
{
  "answer": "Berdasarkan dokumen hukum yang relevan, perbedaan antara korupsi dan kolusi adalah...",
  "sources": [...],
  "confidence": 0.92,
  "provider": "anthropic",
  "model": "claude-3-sonnet"
}
```

### 3. Hybrid Mode

**Description**: Combines basic search results with LLM-generated analysis

**Use Cases**:
- Research and analysis
- Comprehensive answers
- Detailed legal explanations

**Example**:
```python
request = QueryRequest(
    question="jelaskan tindak pidana pencucian uang",
    mode=QueryMode.HYBRID,
    top_k=10
)
```

## API Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "rag_service": true,
  "vector_db": true,
  "llm_manager": true,
  "cache": true,
  "overall": true
}
```

### Process Query

```http
POST /query
Content-Type: application/json
```

**Request Body**:
```json
{
  "question": "apa hukuman pidana korupsi?",
  "mode": "api",
  "top_k": 5,
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "filters": {
    "doc_type": "UU",
    "doc_year": "1999"
  },
  "use_cache": true,
  "session_id": "session_12345"
}
```

**Response**:
```json
{
  "answer": "Berdasarkan UU Nomor 31 Tahun 1999...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "doc_id": "doc_uu31_1999",
      "title": "UU Nomor 31 Tahun 1999",
      "content": "Setiap orang yang...",
      "score": 0.95,
      "metadata": {
        "doc_type": "UU",
        "doc_number": "31",
        "doc_year": "1999",
        "article_number": "2"
      }
    }
  ],
  "confidence": 0.92,
  "processing_time": 1.53,
  "query_id": "query_1705312345_abc12345",
  "mode": "api",
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "cached": false,
  "metadata": {
    "tokens_used": 1250,
    "context_length": 3500
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DB_URL` | `http://vector-db:8000` | Vector database service URL |
| `REDIS_URL` | `redis://:password@cache-service:6379/1` | Redis connection string |
| `DEFAULT_PROVIDER` | `deepseek` | Default LLM provider |
| `DEFAULT_MODEL` | `deepseek-reasoner` | Default LLM model |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `TOP_K_DEFAULT` | `5` | Default number of search results |

### Query Parameters

| Parameter | Type | Default | Limits | Description |
|-----------|------|---------|--------|-------------|
| `question` | string | - | 3-1000 chars | User's question |
| `mode` | string | `api` | basic/api/hybrid | Query processing mode |
| `top_k` | integer | `5` | 1-20 | Number of documents to retrieve |
| `provider` | string | null | - | LLM provider override |
| `model` | string | null | - | LLM model override |
| `filters` | object | null | - | Metadata filters |
| `use_cache` | boolean | `true` | - | Enable/disable caching |
| `session_id` | string | null | - | Session identifier for tracking |

## Document Retrieval

### Search Strategy

The service uses hybrid search:

```python
async def _retrieve_relevant_documents(self, request: QueryRequest):
    # 1. Vector database search
    results = await self.vector_db_client.search(
        query=request.question,
        top_k=request.top_k,
        filters=request.filters
    )

    # Results include:
    # - TF-IDF score (keyword matching)
    # - Semantic score (meaning similarity)
    # - Combined score (60% semantic + 40% TF-IDF)

    return results
```

### Filtering

Filter by metadata:

```python
# Only get results from specific document type
filters = {
    "doc_type": "UU",
    "doc_year": "1999"
}

# Only get specific articles
filters = {
    "article_number": "2"
}
```

## Response Generation

### Context Preparation

The service formats retrieved documents into context:

```python
def _prepare_llm_context(self, relevant_docs):
    context = """Berikut adalah dokumen hukum Indonesia yang relevan:

Dokumen 1: UU Nomor 31 Tahun 1999

Setiap orang yang secara melawan hukum...

(Pasal 2)

[Relevansi: 0.950]

...
"""
    return context
```

### Confidence Scoring

Combined confidence calculation:

```python
confidence = 0.6 * search_confidence + 0.4 * llm_confidence

# Where:
# - search_confidence = max score from retrieved documents
# - llm_confidence = confidence from LLM response
```

## Caching Strategy

### Cache Key

```python
key = f"rag_query:{md5(question + mode + top_k + provider + model + filters)}"
```

### Cache Workflow

1. **Check Cache**: Look for identical previous queries
2. **Cache Hit**: Return cached response immediately
3. **Cache Miss**: Process full query
4. **Store Result**: Cache response for future use

### Cache Invalidation

Manual cache clearing:

```python
await rag_service.clear_cache()  # Clear all RAG cache
```

## Usage Examples

### Python Client

```python
import aiohttp

async def query_legal(question: str):
    async with aiohttp.ClientSession() as session:
        payload = {
            "question": question,
            "mode": "api",
            "top_k": 5,
            "use_cache": True
        }

        async with session.post(
            "http://localhost:8001/query",
            json=payload
        ) as response:
            result = await response.json()
            return result

# Use
result = await query_legal("apa itu korupsi?")
print(result['answer'])
print(f"Sources: {len(result['sources'])}")
print(f"Confidence: {result['confidence']}")
```

### cURL

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "apa hukuman pidana korupsi?",
    "mode": "api",
    "top_k": 5
  }'
```

### With Filters

```python
request = {
    "question": "jelaskan pasal 2",
    "mode": "api",
    "top_k": 5,
    "filters": {
        "doc_type": "UU",
        "doc_number": "31",
        "doc_year": "1999"
    }
}
```

## Performance Optimization

### Improving Response Time

1. **Use Cache**
   ```python
   use_cache=True  # Cache hits return in ~10ms
   ```

2. **Reduce top_k**
   ```python
   top_k=3  # Fewer documents = faster processing
   ```

3. **Use Faster Models**
   ```python
   provider="groq",  # Very fast inference
   model="llama-3.3-70b-versatile"
   ```

4. **Basic Mode for Simple Queries**
   ```python
   mode=QueryMode.BASIC  # No LLM generation
   ```

### Monitoring Performance

Track these metrics:

```python
stats = await rag_service.get_service_stats()
print(f"Average response time: {stats['rag_stats']['average_response_time']:.2f}s")
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")
print(f"Queries by mode: {stats['rag_stats']['queries_by_mode']}")
```

## Error Handling

### Common Errors

#### No Relevant Documents Found

```json
{
  "answer": "Maaf, saya tidak dapat menemukan dokumen hukum yang relevan untuk pertanyaan Anda.",
  "sources": [],
  "confidence": 0.0
}
```

**Solution**: Try rephrasing the question or use broader terms.

#### Service Unavailable

```json
{
  "detail": "RAG Service Unavailable"
}
```

**Solution**: Check service health and dependencies.

#### Invalid Query

```json
{
  "detail": "Invalid query format"
}
```

**Solution**: Ensure question is 3-1000 characters.

## Best Practices

### DO ✅

- Use specific, well-formed questions
- Enable caching for repeated queries
- Use appropriate `top_k` values (3-10)
- Filter by document type when possible
- Check confidence scores
- Review source citations

### DON'T ❌

- Ask very broad questions ("apa itu hukum?")
- Set `top_k` too high (>15)
- Disable cache without reason
- Ignore low confidence scores
- Skip source verification

## Troubleshooting

### Slow Responses (>3s)

**Check**:
1. Vector DB performance
2. LLM API response time
3. Network latency
4. Cache hit rate

**Solutions**:
- Use faster LLM provider
- Enable caching
- Reduce `top_k`
- Check network connectivity

### Low Confidence Scores (<0.5)

**Check**:
1. Query relevance to corpus
2. Document quality and coverage
3. Search filters

**Solutions**:
- Rephrase question
- Remove restrictive filters
- Expand document corpus

### Cache Not Working

**Check**:
1. Redis connection
2. Cache key generation
3. TTL settings

**Solutions**:
- Verify Redis is running
- Check cache stats
- Review cache configuration

## Statistics and Monitoring

### Service Statistics

```python
{
  'total_queries': 1523,
  'successful_queries': 1498,
  'cache_hits': 892,
  'average_response_time': 1.23,
  'queries_by_mode': {
    'api': 1200,
    'basic': 250,
    'hybrid': 73
  },
  'queries_by_provider': {
    'anthropic': 800,
    'deepseek': 500,
    'groq': 223
  }
}
```

### Key Metrics to Track

- **Average Response Time**: Target <2s
- **Cache Hit Rate**: Target >70%
- **Confidence Score**: Target >0.6
- **Success Rate**: Target >95%

## References

- **Implementation**: [`src/rag/rag_service.py`](../ai-agent/src/rag/rag_service.py)
- **Service API**: [`src/rag/service.py`](../ai-agent/src/rag/service.py)
- **Dockerfile**: [`docker/Dockerfile.rag`](../ai-agent/docker/Dockerfile.rag)
