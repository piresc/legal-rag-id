# Indonesian Legal RAG System

Retrieval-Augmented Generation (RAG) system for Indonesian legal documents with TF-IDF search.

## Quick Start

```bash
cd ai-agent
docker-compose up -d
```

## Services

| Service | Port | Description |
|----------|-------|-------------|
| API Gateway | 8000 | Query endpoint |
| Vector DB | 8003 | Document search |
| Redis Cache | 6381 | Response caching |
| LLM Service | - | LLM integration |
| RAG Service | 8001 | RAG processing |

## API Usage

### Query (Basic Mode - No API Key Required)

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa sanksi untuk tidak menggunakan helm?",
    "mode": "basic",
    "top_k": 3
  }'
```

### Query (API Mode - Requires API Key)

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa sanksi untuk tidak menggunakan helm?",
    "mode": "api",
    "provider": "deepseek",
    "top_k": 3
  }'
```

## Configuration

Edit `ai-agent/.env` to configure:

```bash
# API Keys (for API mode)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GROQ_API_KEY=your_key

# Search Settings
USE_SEMANTIC_SEARCH=false
TOP_K_DEFAULT=5
CACHE_TTL=3600
```

## Health Checks

```bash
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8003/health  # Vector DB
```

## Project Structure

```
ai-agent/
├── src/
│   ├── api/              # API gateway
│   ├── rag/              # RAG processing
│   ├── llm/              # LLM integration
│   └── cache/            # Response caching
├── shared/               # Shared utilities
│   └── src/
│       ├── events/         # Event system
│       ├── interfaces/     # Vector DB interface
│       └── utils/          # Config, logging, metrics
├── docker/               # Docker files
├── docker-compose.yml     # Service orchestration
└── .env                  # Environment variables

data/
├── vectors/              # Vector database
└── raw/                 # Raw documents
```

## Data

Vector database contains 35 Indonesian legal documents including:
- UU No. 22 Tahun 2009 (Lalu Lintas dan Angkutan Jalan)
- UU No. 1 Tahun 2009
- Various Peraturan Pemerintah
- Other legal documents

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Restart specific service
docker-compose restart [service-name]
```

## License

MIT License