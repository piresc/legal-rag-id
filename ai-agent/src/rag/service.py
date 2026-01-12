
import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional

from shared.src.utils.logging import setup_logging, get_logger
from shared.src.utils.config import get_config
from ..rag.rag_service import LegalRAGService, QueryRequest, QueryResponse, QueryRequestModel, QueryResponseModel

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="RAG Service")
config = get_config()
rag_service = LegalRAGService(config)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing RAG service...")
    await rag_service.initialize()

@app.get("/health")
async def health_check():
    health = await rag_service.health_check()
    if not health['overall']:
        raise HTTPException(status_code=503, detail="Service unhealthy")
    return health

@app.post("/query", response_model=QueryResponseModel)
async def process_query(request: QueryRequestModel):
    try:
        # Import QueryMode enum
        from ..rag.rag_service import QueryMode

        # Convert mode string to enum
        mode_map = {
            "basic": QueryMode.BASIC,
            "api": QueryMode.API,
            "hybrid": QueryMode.HYBRID
        }
        mode_enum = mode_map.get(request.mode.lower(), QueryMode.API)

        rag_request = QueryRequest(
            question=request.question,
            mode=mode_enum,
            top_k=request.top_k,
            provider=request.provider,
            model=request.model,
            filters=request.filters,
            use_cache=request.use_cache,
            session_id=request.session_id
        )
        response = await rag_service.process_query(rag_request)
        return response
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query processing failed: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
