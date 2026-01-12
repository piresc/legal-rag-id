# API Gateway

## Overview

The API Gateway serves as the unified entry point for all client requests to the Indonesian Legal RAG System. It provides a clean, RESTful API interface while handling routing, CORS, authentication, and request validation internally.

**Service Port**: 8000
**Container Name**: `legal-rag-api-gateway`

## Why API Gateway?

An API Gateway provides several key benefits:

### 1. **Unified Interface**
- Single endpoint for all clients
- Hides internal service complexity
- Simplifies client integration

### 2. **Request Routing**
- Routes requests to appropriate backend services
- Load balancing across service instances
- Service discovery integration

### 3. **Cross-Cutting Concerns**
- Authentication and authorization
- Rate limiting
- Request/response validation
- Logging and monitoring

### 4. **Protocol Translation**
- REST → Internal service protocols
- Response format standardization
- Error handling and translation

## Architecture

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   Port 8000     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐ ┌──────────────┐ ┌─────────────┐
    │ RAG Service │ │ LLM Service  │ │ Vector DB   │
    │   Port 8001 │ │  Port 8002   │ │  Port 8003  │
    └─────────────┘ └──────────────┘ └─────────────┘
```

## API Endpoints

### Health Check

Check gateway and backend service health.

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "api-gateway"
}
```

### Chat Query

Submit a legal query for processing.

```http
POST /chat/query
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

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | Yes | - | User's legal question (3-1000 chars) |
| `mode` | string | No | `api` | Query mode: `basic`, `api`, or `hybrid` |
| `top_k` | integer | No | `5` | Number of documents to retrieve (1-20) |
| `provider` | string | No | null | LLM provider override |
| `model` | string | No | null | LLM model override |
| `filters` | object | No | null | Metadata filters for search |
| `use_cache` | boolean | No | `true` | Enable response caching |
| `session_id` | string | No | null | Session identifier for tracking |

**Success Response** (200 OK):
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

**Error Responses**:

**503 Service Unavailable** (RAG Service Down):
```json
{
  "detail": "RAG Service Unavailable"
}
```

**400 Bad Request** (Invalid Input):
```json
{
  "detail": "Invalid input: question must be between 3 and 1000 characters"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API Gateway host binding |
| `API_PORT` | `8000` | API Gateway port |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_PERIOD` | `60` | Rate limit window (seconds) |
| `JWT_SECRET_KEY` | - | JWT signing key (if auth enabled) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT token expiration |

### Service URLs

The gateway routes to internal services:

| Service | Internal URL | External URL |
|---------|--------------|--------------|
| RAG Service | `http://rag-service:8001` | `localhost:8001` |
| LLM Service | `http://llm-service:8002` | `localhost:8002` |
| Vector DB | `http://vector-db:8000` | `localhost:8003` |

## Features

### CORS (Cross-Origin Resource Sharing)

The gateway handles CORS automatically:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Configuration**:
```python
allow_origins=[
    "https://your-frontend.com",
    "https://app.your-domain.com"
]
```

### Request Validation

All requests are validated before routing:

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: str = Field("api", regex="^(basic|api|hybrid)$")
    top_k: int = Field(5, ge=1, le=20)
    # ... additional fields
```

### Error Handling

The gateway provides consistent error responses:

```python
try:
    async with session.post(f"{RAG_SERVICE_URL}/query", json=request.dict()) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=resp.status, detail="RAG Service Error")
        return await resp.json()
except aiohttp.ClientError as e:
    logger.error(f"Connection error: {e}")
    raise HTTPException(status_code=503, detail="RAG Service Unavailable")
```

## Usage Examples

### cURL

```bash
# Basic query
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "apa itu tindak pidana korupsi?"
  }'

# Query with parameters
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "jelaskan pasal 2 UU 31/1999",
    "mode": "api",
    "top_k": 5,
    "provider": "anthropic",
    "use_cache": true
  }'

# Query with filters
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "hukuman pidana korupsi",
    "filters": {
      "doc_type": "UU",
      "doc_year": "1999"
    }
  }'
```

### Python

```python
import requests

# Basic query
response = requests.post(
    "http://localhost:8000/chat/query",
    json={
        "question": "apa itu korupsi?"
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])}")

# Query with all options
response = requests.post(
    "http://localhost:8000/chat/query",
    json={
        "question": "jelaskan perbedaan korupsi dan kolusi",
        "mode": "api",
        "top_k": 10,
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "use_cache": True,
        "session_id": "user_123_session"
    }
)

result = response.json()
```

### JavaScript/TypeScript

```typescript
// Basic query
const response = await fetch('http://localhost:8000/chat/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'apa itu korupsi?'
  })
});

const result = await response.json();
console.log('Answer:', result.answer);
console.log('Confidence:', result.confidence);

// Query with parameters
const response = await fetch('http://localhost:8000/chat/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'jelaskan pasal 2',
    mode: 'api',
    top_k: 5,
    provider: 'anthropic',
    filters: {
      doc_type: 'UU',
      doc_number: '31',
      doc_year: '1999'
    }
  })
});
```

### Java

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import org.json.JSONObject;

HttpClient client = HttpClient.newHttpClient();

JSONObject requestBody = new JSONObject();
requestBody.put("question", "apa itu korupsi?");
requestBody.put("mode", "api");
requestBody.put("top_k", 5);

HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://localhost:8000/chat/query"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
    .build();

HttpResponse<String> response = client.send(request,
    HttpResponse.BodyHandlers.ofString());

JSONObject result = new JSONObject(response.body());
System.out.println("Answer: " + result.getString("answer"));
```

## Request Flow

### Typical Query Flow

```
1. Client Request
   │
   ▼
2. API Gateway
   │  ├── Validate request
   │  ├── Check authentication (if enabled)
   │  └── Apply rate limiting
   │
   ▼
3. Route to RAG Service
   │  ├── Check cache
   │  ├── Retrieve documents (Vector DB)
   │  └── Generate response (LLM Service)
   │
   ▼
4. Response to Gateway
   │  └── Format/validate response
   │
   ▼
5. Return to Client
```

## Rate Limiting

### Implementation

Rate limiting protects backend services:

```python
# Configuration
RATE_LIMIT_REQUESTS = 100  # requests
RATE_LIMIT_PERIOD = 60     # seconds

# This allows 100 requests per minute per IP
```

### Best Practices

1. **Set Appropriate Limits**
   - Development: 1000 req/min
   - Production: 100 req/min
   - Paid tiers: higher limits

2. **Implement Backoff**
   ```python
   import time

   for attempt in range(3):
       try:
           response = make_request()
           break
       except HTTPException as e:
           if e.status_code == 429:
               time.sleep(2 ** attempt)  # Exponential backoff
   ```

3. **Use Session IDs**
   - Track usage per user
   - Implement fair queuing

## Authentication (Future Enhancement)

### JWT Token-Based Auth

```python
# Login endpoint
@app.post("/auth/login")
async def login(credentials: Credentials):
    # Validate credentials
    # Generate JWT token
    token = create_jwt_token(user_id)
    return {"access_token": token}

# Protected endpoint
@app.post("/chat/query")
async def chat_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    # Process request
    pass
```

### API Key Auth

```python
# Verify API key
@app.api_key_header("X-API-Key")
async def verify_api_key(api_key: str):
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Protected endpoint
@app.post("/chat/query", dependencies=[Depends(verify_api_key)])
async def chat_query(request: QueryRequest):
    # Process request
    pass
```

## Monitoring and Logging

### Request Logging

```python
logger.info(f"Incoming request: {request.question[:100]}...")
logger.info(f"Processing time: {response.processing_time:.2f}s")
logger.info(f"Provider used: {response.provider}")
```

### Metrics to Track

1. **Request Volume**
   - Requests per second
   - Requests per endpoint
   - Requests per user

2. **Response Times**
   - P50, P95, P99 latencies
   - Backend service times
   - Total request time

3. **Error Rates**
   - 4xx errors (client errors)
   - 5xx errors (server errors)
   - Timeouts

4. **Resource Usage**
   - CPU, memory
   - Network I/O
   - Connection counts

## Performance Optimization

### Connection Pooling

```python
# Reuse HTTP connections
connector = aiohttp.TCPConnector(
    limit=100,              # Max connections
    limit_per_host=10,      # Max per host
    ttl_dns_cache=300       # DNS cache TTL
)

session = aiohttp.ClientSession(connector=connector)
```

### Response Compression

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Caching Headers

```python
@app.get("/health")
async def health_check():
    response = {"status": "healthy"}
    return JSONResponse(
        content=response,
        headers={"Cache-Control": "no-cache"}
    )
```

## Deployment

### Docker Compose

```yaml
api-gateway:
  build:
    context: .
    dockerfile: docker/Dockerfile.api_gateway
  container_name: legal-rag-api-gateway
  environment:
    - API_HOST=0.0.0.0
    - API_PORT=8000
    - CORS_ORIGINS=*
    - RATE_LIMIT_REQUESTS=100
    - RATE_LIMIT_PERIOD=60
  ports:
    - "8000:8000"
  depends_on:
    - rag-service
    - llm-service
    - vector-db
  networks:
    - legal-rag-ai-network
  restart: unless-stopped
```

### Production Considerations

1. **Use Nginx/Traefik**
   - SSL termination
   - Load balancing
   - Static file serving

2. **Enable Rate Limiting**
   - Prevent abuse
   - Fair resource allocation

3. **Implement Authentication**
   - JWT or API keys
   - User management

4. **Monitor and Alert**
   - Error rates
   - Response times
   - Resource usage

5. **Log Aggregation**
   - Centralized logging
   - Structured logs (JSON)

## Troubleshooting

### Connection Refused

**Symptom**: `Connection refused` error

**Solutions**:
1. Check gateway is running: `docker ps | grep api-gateway`
2. Verify port 8000 is not in use
3. Check network configuration
4. Review logs: `docker logs legal-rag-api-gateway`

### CORS Errors

**Symptom**: Browser shows CORS error

**Solutions**:
1. Verify `CORS_ORIGINS` includes your domain
2. Check preflight requests are handled
3. Ensure credentials are allowed if needed

### Slow Responses

**Symptom**: Requests taking >3 seconds

**Solutions**:
1. Check backend service health
2. Enable response caching
3. Implement rate limiting
4. Scale backend services

### 503 Service Unavailable

**Symptom**: `RAG Service Unavailable` error

**Solutions**:
1. Check RAG service is running
2. Verify internal service URLs
3. Check network connectivity
4. Review service dependencies

## Best Practices

### DO ✅

1. **Use HTTPS in production**
2. **Implement rate limiting**
3. **Validate all inputs**
4. **Log all requests**
5. **Monitor performance**
6. **Handle errors gracefully**
7. **Use appropriate HTTP status codes**
8. **Implement authentication**
9. **Set reasonable timeouts**
10. **Cache responses when appropriate**

### DON'T ❌

1. **Don't bypass validation**
2. **Don't expose internal URLs**
3. **Don't ignore rate limits**
4. **Don't log sensitive data**
5. **Don't allow unlimited requests**
6. **Don't return raw errors to clients**
7. **Don't skip authentication**
8. **Don't hardcode configuration**
9. **Don't ignore CORS in production**
10. **Don't forget to monitor**

## References

- **Implementation**: [`src/api/main.py`](../ai-agent/src/api/main.py)
- **Dockerfile**: [`docker/Dockerfile.api_gateway`](../ai-agent/docker/Dockerfile.api_gateway)
- **Docker Compose**: [`docker-compose.yml`](../ai-agent/docker-compose.yml)

## Related Documentation

- [RAG Service](./rag-service.md) - Backend query processing
- [LLM Service](./llm-service.md) - AI provider integration
- [Vector DB](./vector-db.md) - Document storage and search
- [Infrastructure](./infrastructure.md) - Nginx, Consul, RabbitMQ
