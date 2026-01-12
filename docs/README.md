# Indonesian Legal RAG System - Documentation

Welcome to the documentation for the Indonesian Legal RAG (Retrieval-Augmented Generation) System. This system provides intelligent search and question-answering capabilities for Indonesian legal documents using advanced AI techniques.

## Overview

The system is built as a microservices architecture with the following core components:

- **Vector Database** - Document storage and semantic search
- **Cache Service** - Redis-based response caching
- **RAG Service** - Retrieval-augmented generation orchestration
- **LLM Service** - Multi-provider LLM integration
- **API Gateway** - Unified REST API entry point

## Documentation

### Core Services

| Service | Description | Documentation |
|---------|-------------|---------------|
| **Vector DB** | Enhanced vector database with TF-IDF and semantic search | [vector-db.md](./vector-db.md) |
| **Cache Service** | Redis-based caching with in-memory fallback | [cache-service.md](./cache-service.md) |
| **RAG Service** | Main orchestration service for query processing | [rag-service.md](./rag-service.md) |
| **LLM Service** | Multi-provider LLM integration (OpenAI, Anthropic, DeepSeek, Groq) | [llm-service.md](./llm-service.md) |
| **API Gateway** | FastAPI-based unified entry point | [api-gateway.md](./api-gateway.md) |

### Infrastructure

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **Nginx** | Load balancer and reverse proxy for production deployments | [infrastructure.md](./infrastructure.md#nginx) |
| **Consul** | Service discovery and health monitoring | [infrastructure.md](./infrastructure.md#consul) |
| **RabbitMQ** | Message queue for asynchronous processing | [infrastructure.md](./infrastructure.md#rabbitmq) |

## Quick Start

```bash
# Deploy all services
cd scripts
./deploy.sh deploy

# Check service health
./deploy.sh health

# View service URLs
./deploy.sh urls
```

## System Architecture

```
┌─────────────┐
│ API Gateway │ ◄───── User Requests (Port 8000)
└──────┬──────┘
       │
       ├───► RAG Service (Port 8001)
       │      │
       │      ├───► Vector DB (Port 8003)
       │      │      └─── TF-IDF Search
       │      │      └─── Semantic Search
       │      │
       │      ├───► LLM Service (Port 8002)
       │      │      └─── OpenAI
       │      │      └─── Anthropic
       │      │      └─── DeepSeek
       │      │      └─── Groq
       │      │
       │      └───► Cache Service (Redis 6381)
       │             └─── Response Caching
       │
       └───► Infrastructure
              ├─── Nginx (Production Load Balancer)
              ├─── Consul (Service Discovery)
              └─── RabbitMQ (Async Processing)
```

## Query Flow

1. **User Request** → API Gateway receives query
2. **Cache Check** → RAG Service checks cache for identical queries
3. **Document Retrieval** → Vector DB searches for relevant legal documents
4. **Context Preparation** → Retrieved documents formatted as context
5. **LLM Generation** → LLM Service generates response using context
6. **Response Caching** → Result cached for future queries
7. **Return Result** → Formatted response with sources returned to user

## Features

- **Hybrid Search**: Combines TF-IDF keyword matching with semantic similarity
- **Multi-LLM Support**: Works with OpenAI, Anthropic, DeepSeek, and Groq
- **Intelligent Caching**: Reduces latency and API costs for repeated queries
- **Indonesian Legal Focus**: Optimized for Indonesian legal documents and terminology
- **Confidence Scoring**: Provides relevance scores for all responses
- **Source Citations**: All responses include source document references

## Development

For local development:

```bash
# Build and start services
cd ai-agent
docker-compose up --build

# View logs
docker-compose logs -f rag-service

# Run tests
pytest tests/
```

## Deployment

See [deploy.sh](../scripts/deploy.sh) for deployment options.

## Support

For issues or questions, please refer to the individual service documentation linked above.
