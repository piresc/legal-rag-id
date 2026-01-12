# Cache Service

## Overview

The Cache Service provides high-performance response caching using Redis with automatic fallback to in-memory storage. It significantly reduces latency and API costs by serving repeated queries from cache.

**Service Port**: 6381 (external), 6379 (internal)
**Container Name**: `legal-rag-ai-cache`
**Technology**: Redis 7 Alpine

## Why Caching Matters

### Benefits

1. **Reduced Latency**
   - Cached responses: ~5-10ms
   - Full query processing: ~500-2000ms
   - **100-200x faster** for cached queries

2. **Lower API Costs**
   - Avoids repeated LLM API calls
   - Reduces token usage by ~60-80% for typical workloads

3. **Improved User Experience**
   - Near-instant responses for common questions
   - Reduced load on backend services

4. **Rate Limit Protection**
   - Fewer API calls to external services
   - Better resilience under high load

## Architecture

### Hybrid Caching Strategy

```
┌─────────────────┐
│  Cache Manager  │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Primary │    │
    │  Redis  │    │ Fallback
    │  Cache  │◄───┤ In-Memory
    └─────────┘    │ Cache
                   └──────────┘
```

### Cache Hierarchy

1. **Redis (Primary)**
   - Persistent across service restarts
   - Shared across all service instances
   - Supports TTL (Time To Live)
   - Advanced features (patterns, stats)

2. **In-Memory (Fallback)**
   - Automatic fallback if Redis unavailable
   - Per-instance caching
   - No persistence
   - Graceful degradation

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://:redis_password@localhost:6379/1` | Redis connection string |
| `REDIS_PASSWORD` | `redis_password` | Redis password |
| `CACHE_TTL` | `3600` | Default TTL in seconds (1 hour) |

### Docker Compose

```yaml
cache-service:
  image: redis:7-alpine
  command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
  volumes:
    - ai_redis_data:/data
  ports:
    - "6381:6379"
```

## Cache Manager API

### Initialization

```python
from src.cache.cache_manager import CacheManager

cache = CacheManager(
    redis_url="redis://:password@localhost:6379/1",
    default_ttl=3600  # 1 hour
)

await cache.initialize()
```

### Basic Operations

#### Set Cache

```python
# Set with default TTL
await cache.set("query:123", response_data)

# Set with custom TTL (5 minutes)
await cache.set("query:123", response_data, ttl=300)
```

#### Get Cache

```python
# Retrieve from cache
cached_data = await cache.get("query:123")

if cached_data:
    print("Cache hit!")
else:
    print("Cache miss")
```

#### Delete Cache

```python
# Delete specific key
await cache.delete("query:123")

# Clear all keys matching pattern
await cache.clear_pattern("rag_query:*")
```

### Advanced Operations

#### Check Existence

```python
exists = await cache.exists("query:123")
# Returns: True or False
```

#### Get TTL

```python
ttl = await cache.get_ttl("query:123")
# Returns: Remaining seconds or None
```

#### Health Check

```python
health = await cache.health_check()
# Returns: {
#   'healthy': True,
#   'cache_type': 'redis',
#   'connected': True
# }
```

#### Statistics

```python
stats = await cache.get_stats()
# Returns: {
#   'cache_type': 'redis',
#   'hits': 1523,
#   'misses': 312,
#   'hit_rate': 0.83,
#   'sets': 892,
#   'total_keys': 892,
#   'memory_used': '45.2M',
#   'connected_clients': 5
# }
```

## Usage in RAG Service

### Cache Key Generation

```python
def _generate_cache_key(request: QueryRequest) -> str:
    key_data = {
        'question': request.question,
        'mode': request.mode.value,
        'top_k': request.top_k,
        'provider': request.provider,
        'model': request.model,
        'filters': request.filters
    }

    key_string = json.dumps(key_data, sort_keys=True)
    return f"rag_query:{hashlib.md5(key_string.encode()).hexdigest()}"
```

### Cache Workflow

```python
# 1. Check cache
cached_response = await self._get_cached_response(request)
if cached_response:
    return cached_response  # Cache hit!

# 2. Process query
response = await self._process_query(request)

# 3. Cache result
await self._cache_response(request, response)

# 4. Return response
return response
```

## Cache Strategy

### TTL (Time To Live) Guidelines

| Data Type | Recommended TTL | Rationale |
|-----------|-----------------|-----------|
| Legal query responses | 1-24 hours | Legal documents rarely change |
| Vector search results | 1-6 hours | Database updates infrequent |
| LLM responses | 1-12 hours | Same query → same answer |
| Session data | 30-60 minutes | User session duration |

### Cache Invalidation

#### Manual Invalidation

```python
# Clear all RAG query cache
await cache.clear_pattern("rag_query:*")

# Clear specific session
await cache.clear_pattern(f"session:{session_id}:*")
```

#### Automatic Invalidation

- TTL expiration (automatic)
- Service restart clears in-memory cache
- Redis restart (unless AOF enabled)

### Cache Warming

Pre-populate cache with common queries:

```python
common_queries = [
    "apa itu korupsi?",
    "jelaskan pasal 12 UU 31/1999",
    "hukuman pidana korupsi"
]

for query in common_queries:
    request = QueryRequest(question=query)
    await process_query(request)  # Results will be cached
```

## Performance Optimization

### Hit Rate Optimization

1. **Normalize Queries**
   ```python
   # Bad
   "Apa itu korupsi?"
   "apa itu korupsi?"  # Different cache key

   # Good
   query = query.lower().strip()
   "apa itu korupsi?"  # Same cache key
   ```

2. **Use Consistent Parameters**
   ```python
   # Bad
   QueryRequest(question="...", top_k=5)
   QueryRequest(question="...", top_k=10)  # Different key

   # Good
   QueryRequest(question="...", top_k=5)  # Use default
   ```

3. **Cache Similar Queries**
   ```python
   # Group similar queries
   key = f"query_category:{category}:{hash(query)}"
   ```

### Memory Management

#### Redis Memory Limits

```bash
# Set max memory (in redis.conf)
maxmemory 256mb
maxmemory-policy allkeys-lru  # Evict least recently used
```

#### Monitor Memory Usage

```python
stats = await cache.get_stats()
print(f"Memory used: {stats['memory_used']}")

# Clear old entries if needed
if stats['total_keys'] > 10000:
    await cache.clear_pattern("rag_query:*")
```

## Monitoring

### Key Metrics

Track these metrics for cache health:

1. **Hit Rate**
   - Target: >70%
   - Warning: <50%
   - Critical: <30%

2. **Memory Usage**
   - Monitor: `stats['memory_used']`
   - Alert: >500MB

3. **Key Count**
   - Monitor: `stats['total_keys']`
   - Alert: >50,000 keys

### Metrics Export

```python
from shared.src.utils.metrics import increment_counter, record_timing

# In cache operations
if cached_value:
    increment_counter("cache_hits_total")
else:
    increment_counter("cache_misses_total")

# Record operation timing
with track_performance("cache_get"):
    value = await cache.get(key)
```

## Troubleshooting

### Low Hit Rate (<50%)

**Causes**:
1. Too short TTL
2. High query variance
3. Cache not being used
4. Keys expiring too fast

**Solutions**:
1. Increase TTL
2. Normalize queries better
3. Check cache is actually being called
4. Review cache key generation

### Redis Connection Failures

**Symptoms**:
- Logs show "Failed to connect to Redis"
- All caches miss
- Falls back to in-memory

**Solutions**:
1. Check Redis is running: `docker ps | grep redis`
2. Verify connection string
3. Check password is correct
4. Test connection: `redis-cli -h localhost -p 6381 -a password ping`

### High Memory Usage

**Symptoms**:
- Redis using >1GB memory
- OOM warnings in logs

**Solutions**:
1. Reduce TTL
2. Set max memory limit
3. Enable LRU eviction
4. Clear old cache entries
5. Reduce cache key count

## Best Practices

### DO ✅

- Use sensible TTL values (1-24 hours)
- Monitor hit rates regularly
- Set memory limits
- Use cache patterns for bulk operations
- Handle cache failures gracefully
- Log cache performance metrics

### DON'T ❌

- Cache sensitive data without encryption
- Use extremely long TTL (>7 days)
- Ignore cache failures (should fallback)
- Cache real-time data
- Cache very large responses (>1MB)
- Forget to implement cache invalidation

## Example Implementation

Complete example showing cache integration:

```python
from src.cache.cache_manager import CacheManager
from src.rag.rag_service import QueryRequest, QueryResponse

class CachedRAGService:
    def __init__(self):
        self.cache = CacheManager(
            redis_url="redis://:password@cache-service:6379/1",
            default_ttl=3600
        )

    async def initialize(self):
        await self.cache.initialize()

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        # Check cache
        cache_key = self._make_key(request)
        cached = await self.cache.get(cache_key)

        if cached:
            return QueryResponse(**cached)

        # Process query
        response = await self._process(request)

        # Cache result
        await self.cache.set(cache_key, response.__dict__)

        return response

    def _make_key(self, request: QueryRequest) -> str:
        import hashlib
        import json

        data = {
            'q': request.question,
            'mode': request.mode.value,
            'k': request.top_k
        }
        return f"rag:{hashlib.md5(json.dumps(data).encode()).hexdigest()}"
```

## References

- **Implementation**: [`src/cache/cache_manager.py`](../ai-agent/src/cache/cache_manager.py)
- **Docker Compose**: [`docker-compose.yml`](../ai-agent/docker-compose.yml)
- **Redis Documentation**: https://redis.io/docs/
