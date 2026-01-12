# LLM Service

## Overview

The LLM (Large Language Model) Service provides a unified interface for multiple AI providers, enabling flexible and cost-effective natural language generation. It supports OpenAI, Anthropic, DeepSeek, and Groq with automatic fallback and load balancing.

**Service Port**: 8002
**Container Name**: `legal-rag-llm-service`

## Supported Providers

| Provider | Models | Best For | Cost | Speed |
|----------|--------|----------|------|-------|
| **OpenAI** | GPT-3.5, GPT-4 | General purpose, complex reasoning | $$$ | Medium |
| **Anthropic** | Claude 3 Sonnet/Opus | nuanced, long-form responses | $$$$ | Medium |
| **DeepSeek** | DeepSeek Reasoner | Code, reasoning, cost-effective | $ | Slow |
| **Groq** | Llama 3, Mixtral | Fast inference | Free | Fastest |

## Architecture

### Provider Selection Flow

```
Request
    │
    ▼
┌──────────────────┐
│ Provider Request │
│  (optional)      │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  Specified  Default
  Provider?  Provider
    │         │
    └────┬────┘
         │
    ┌────▼─────────┐
    │ Try Provider │
    └────┬─────────┘
         │
    ┌────▼────┐
    │ Success? │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
    ▼         ▼
  Return  Try Fallback
  Result  Providers
```

### Features

- ✅ **Multiple Providers**: Switch between providers without code changes
- ✅ **Automatic Fallback**: If one provider fails, try others automatically
- ✅ **Health Monitoring**: Track provider availability and performance
- ✅ **Cost Optimization**: Use cheaper providers when appropriate
- ✅ **Token Tracking**: Monitor usage and costs across providers
- ✅ **Indonesian Optimization**: Prompts optimized for Indonesian legal text

## API Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "stats": {
    "total_requests": 1523,
    "requests_by_provider": {
      "anthropic": 800,
      "deepseek": 500,
      "groq": 223
    },
    "average_response_time": 1.23,
    "total_tokens_used": 1250000
  }
}
```

### Generate Response

```http
POST /generate
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "apa hukuman pidana korupsi?",
  "context": "Dokumen hukum Indonesia...\n\nUU Nomor 31 Tahun 1999...",
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229"
}
```

**Response**:
```json
{
  "content": "Berdasarkan UU Nomor 31 Tahun 1999 tentang Pemberantasan Tindak Pidana Korupsi...",
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229",
  "tokens_used": 1250,
  "processing_time": 1.53,
  "confidence": 0.85,
  "metadata": {
    "stop_reason": "end_turn",
    "input_tokens": 950,
    "output_tokens": 300
  }
}
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `DEEPSEEK_API_KEY` | No | - | DeepSeek API key |
| `GROQ_API_KEY` | No | - | Groq API key |
| `DEFAULT_PROVIDER` | No | `deepseek` | Default LLM provider |
| `DEFAULT_MODEL` | No | `deepseek-reasoner` | Default model name |
| `MAX_TOKENS` | No | `1000` | Maximum tokens per response |
| `TEMPERATURE` | No | `0.7` | Response randomness (0-1) |
| `TIMEOUT` | No | `30` | Request timeout in seconds |

### Provider Configuration

Each provider can be configured independently:

```python
LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
    max_tokens=1500,
    temperature=0.7,
    timeout=30,
    enabled=True
)
```

## Usage Examples

### Python Client

```python
import aiohttp

async def generate_legal_response(query: str, context: str):
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": query,
            "context": context,
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229"
        }

        async with session.post(
            "http://localhost:8002/generate",
            json=payload
        ) as response:
            result = await response.json()
            return result

# Use
response = await generate_legal_response(
    query="apa itu tindak pidana korupsi?",
    context="UU Nomor 31 Tahun 1999..."
)

print(f"Response: {response['content']}")
print(f"Tokens used: {response['tokens_used']}")
print(f"Processing time: {response['processing_time']:.2f}s")
```

### Using LLM Manager Directly

```python
from src.llm.llm_manager import LLMManager

# Initialize
llm_manager = LLMManager()
await llm_manager.initialize()

# Generate response
response = await llm_manager.generate_response(
    query="apa perbedaan korupsi dan kolusi?",
    context=context_text,
    provider="anthropic"  # Optional, uses default if not specified
)

print(f"Content: {response.content}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Tokens: {response.tokens_used}")
```

### Specifying Provider and Model

```python
# Use OpenAI GPT-4
response = await llm_manager.generate_response(
    query="complex legal question",
    context=context,
    provider="openai",
    model="gpt-4"
)

# Use Groq for fast inference
response = await llm_manager.generate_response(
    query="simple question",
    context=context,
    provider="groq",
    model="llama3-70b-8192"
)
```

## Prompt Engineering

### Indonesian Legal Assistant Prompt

The service uses optimized prompts for Indonesian legal queries:

```python
system_prompt = """Anda adalah asisten hukum Indonesia yang profesional dan berpengetahuan luas.
Tugas Anda adalah menjawab pertanyaan hukum berdasarkan dokumen hukum Indonesia yang diberikan.

Pedoman jawaban:
1. Jawab dalam bahasa Indonesia yang jelas dan mudah dipahami
2. Berikan jawaban yang akurat berdasarkan informasi dari dokumen
3. Sertakan referensi ke pasal atau dokumen yang relevan
4. Jika informasi tidak cukup, jelaskan secara jelas
5. Berikan jawaban yang ringkas namun komprehensif
6. Hindari memberikan nasihat hukum formal, sebutkan bahwa ini adalah informasi umum

Format jawaban:
- Jawaban langsung pertanyaan
- Dasar hukum (pasal/dokumen)
- Penjelasan singkat jika diperlukan"""
```

### User Prompt Template

```python
user_prompt = f"""Pertanyaan: {query}

{context}

Berdasarkan dokumen hukum di atas, jawab pertanyaan tersebut secara akurat dan profesional."""
```

## Provider Comparison

### OpenAI (GPT-3.5/GPT-4)

**Strengths**:
- Best for complex reasoning
- Excellent understanding of nuance
- Strong multilingual support

**Weaknesses**:
- Higher cost (especially GPT-4)
- Rate limits can be restrictive
- API can be slow during high demand

**Best For**:
- Complex legal analysis
- Detailed explanations
- Multi-step reasoning

**Cost**: ~$0.002-0.12 per 1K tokens

### Anthropic (Claude 3)

**Strengths**:
- Excellent for long-form content
- Strong ethical guidelines
- Good at following instructions
- Large context window (200K tokens)

**Weaknesses**:
- Higher cost than some alternatives
- Slightly slower than Groq

**Best For**:
- Detailed legal explanations
- Document summarization
- Complex, nuanced questions

**Cost**: ~$0.003-0.015 per 1K tokens

### DeepSeek

**Strengths**:
- Very cost-effective
- Good at code and reasoning
- Strong performance for price

**Weaknesses**:
- Slower inference
- Less optimized for non-code tasks
- May be less reliable

**Best For**:
- Cost-sensitive applications
- Reasoning tasks
- Background processing

**Cost**: ~$0.001-0.002 per 1K tokens

### Groq (Llama 3, Mixtral)

**Strengths**:
- Extremely fast inference
- Free tier available
- Good quality open-source models
- Low latency

**Weaknesses**:
- Models may be less capable than GPT-4/Claude
- Limited context window
- May struggle with very complex tasks

**Best For**:
- Real-time responses
- High-volume queries
- Cost optimization

**Cost**: Free (as of writing)

## Cost Optimization

### Strategy 1: Provider Tiers

```python
# Simple queries → Fast, cheap provider
if complexity_score < 0.3:
    provider = "groq"  # Free and fast

# Medium complexity → Mid-tier provider
elif complexity_score < 0.7:
    provider = "deepseek"  # Cost-effective

# Complex queries → Best provider
else:
    provider = "anthropic"  # Best quality
```

### Strategy 2: Caching

```python
# Check cache first
cached = await cache.get(query_hash)
if cached:
    return cached  # Avoid API call entirely

# Generate and cache
response = await llm_manager.generate_response(...)
await cache.set(query_hash, response)
```

### Strategy 3: Token Limits

```python
# Adjust max_tokens based on query type
if is_simple_question(query):
    max_tokens = 500  # Shorter responses
else:
    max_tokens = 1500  # Detailed responses
```

## Monitoring and Metrics

### Provider Statistics

```python
stats = await llm_manager.get_stats()
# Returns:
{
    'llm_stats': {
        'total_requests': 1523,
        'requests_by_provider': {
            'anthropic': 800,
            'deepseek': 500,
            'groq': 223
        },
        'average_response_time': 1.23,
        'total_tokens_used': 1250000
    },
    'provider_count': 3,
    'providers': ['anthropic', 'deepseek', 'groq'],
    'default_provider': 'anthropic'
}
```

### Provider Health Status

```python
status = await llm_manager.get_provider_status()
# Returns:
{
    'anthropic': {
        'healthy': True,
        'model_info': {
            'provider': 'anthropic',
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 1000,
            'temperature': 0.7
        }
    },
    'deepseek': {
        'healthy': True,
        'model_info': {...}
    },
    'groq': {
        'healthy': False,
        'error': 'Connection timeout'
    }
}
```

### Key Metrics to Track

1. **Request Volume**: Track requests per provider
2. **Response Time**: Monitor average latency
3. **Token Usage**: Track costs across providers
4. **Error Rate**: Monitor provider failures
5. **Fallback Rate**: How often primary provider fails

## Error Handling

### Automatic Fallback

If a provider fails, the service automatically tries other providers:

```python
try:
    # Try primary provider
    response = await llm_manager.generate_response(
        query="...",
        context="...",
        provider="anthropic"
    )
except Exception as e:
    # Automatic fallback to other providers
    # Tries: deepseek → groq → openai
    logger.error(f"Primary provider failed: {e}")
```

### Common Errors

#### API Key Invalid

```json
{
  "error": "OpenAI API error: 401 - Incorrect API key provided"
}
```

**Solution**: Verify API key in environment variables

#### Rate Limit Exceeded

```json
{
  "error": "Anthropic API error: 429 - Rate limit exceeded"
}
```

**Solution**:
1. Implement exponential backoff
2. Use multiple providers
3. Add request queuing

#### Provider Timeout

```json
{
  "error": "Groq API error: timeout - Request took longer than 30s"
}
```

**Solution**:
1. Increase timeout value
2. Try simpler prompts
3. Use fallback provider

## Best Practices

### DO ✅

1. **Use appropriate provider for task**
   - Simple queries → Groq (fast, free)
   - Complex analysis → Anthropic/OpenAI (best quality)

2. **Implement caching**
   - Cache responses to repeated queries
   - Set appropriate TTL

3. **Monitor token usage**
   - Track costs per provider
   - Set budget limits

4. **Handle errors gracefully**
   - Always have fallback
   - Log failures for analysis

5. **Optimize prompts**
   - Be specific and clear
   - Include relevant context
   - Use appropriate language (Indonesian)

### DON'T ❌

1. **Don't rely on single provider**
   - Always configure multiple providers
   - Test fallback mechanism

2. **Don't ignore rate limits**
   - Implement backoff
   - Distribute load across providers

3. **Don't use excessive tokens**
   - Set reasonable `max_tokens`
   - Truncate long context if needed

4. **Don't forget to track costs**
   - Monitor token usage
   - Set up alerts

5. **Don't expose API keys**
   - Use environment variables
   - Never commit keys to git

## Performance Tuning

### Latency Optimization

```python
# 1. Use faster provider for time-sensitive queries
response = await llm_manager.generate_response(
    query="...",
    context="...",
    provider="groq"  # Fastest
)

# 2. Reduce max_tokens for faster responses
response = await llm_manager.generate_response(
    query="...",
    context="...",
    max_tokens=500  # Shorter = faster
)

# 3. Lower temperature for more deterministic (faster) responses
response = await llm_manager.generate_response(
    query="...",
    context="...",
    temperature=0.3  # Lower = faster
)
```

### Quality Optimization

```python
# 1. Use best provider for quality
response = await llm_manager.generate_response(
    query="...",
    context="...",
    provider="anthropic"  # Best quality
)

# 2. Increase max_tokens for detailed responses
response = await llm_manager.generate_response(
    query="...",
    context="...",
    max_tokens=2000
)

# 3. Provide comprehensive context
response = await llm_manager.generate_response(
    query="...",
    context=detailed_legal_documents,  # More context = better
)
```

## Troubleshooting

### All Providers Failing

**Symptoms**: All providers returning errors

**Solutions**:
1. Check API keys are valid
2. Verify network connectivity
3. Check provider status pages
4. Review rate limits
5. Check request format

### High Token Usage

**Symptoms**: Unexpectedly high token counts

**Solutions**:
1. Truncate context if too long
2. Reduce `max_tokens` setting
3. Use more concise prompts
4. Check for token counting bugs

### Slow Responses

**Symptoms**: Responses taking >5 seconds

**Solutions**:
1. Switch to faster provider (Groq)
2. Reduce context length
3. Lower `max_tokens`
4. Check network latency
5. Implement caching

## References

- **Implementation**: [`src/llm/llm_manager.py`](../ai-agent/src/llm/llm_manager.py)
- **Service API**: [`src/llm/service.py`](../ai-agent/src/llm/service.py)
- **Dockerfile**: [`docker/Dockerfile.llm`](../ai-agent/docker/Dockerfile.llm)

## Provider Documentation

- [OpenAI API](https://platform.openai.com/docs)
- [Anthropic Claude](https://docs.anthropic.com/claude/docs)
- [DeepSeek API](https://platform.deepseek.com/api-docs/)
- [Groq API](https://console.groq.com/docs)
