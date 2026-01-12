
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
from pydantic import BaseModel

from shared.src.utils.logging import setup_logging, get_logger
from shared.src.utils.config import get_config

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Indonesian Legal RAG API Gateway")
config = get_config()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_SERVICE_URL = "http://rag-service:8001"

class QueryRequest(BaseModel):
    question: str
    mode: str = "api"
    top_k: int = 5
    provider: Optional[str] = None
    model: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    session_id: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-gateway"}

@app.post("/chat/query")
async def chat_query(request: QueryRequest):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{RAG_SERVICE_URL}/query", json=request.dict()) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"RAG Service Error: {error_detail}")
                return await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Connection error to RAG service: {e}")
            raise HTTPException(status_code=503, detail="RAG Service Unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
