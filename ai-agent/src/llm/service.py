
import asyncio
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

from shared.src.utils.logging import setup_logging, get_logger
from shared.src.utils.config import get_config
from ..llm.llm_manager import LLMManager

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="LLM Service")
config = get_config()
llm_manager = LLMManager(config)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing LLM manager...")
    await llm_manager.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "stats": await llm_manager.get_stats()}

class GenerationRequest(BaseModel):
    query: str
    context: str
    provider: Optional[str] = None
    model: Optional[str] = None

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        response = await llm_manager.generate_response(
            query=request.query,
            context=request.context,
            provider=request.provider,
            model=request.model
        )
        return response
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
