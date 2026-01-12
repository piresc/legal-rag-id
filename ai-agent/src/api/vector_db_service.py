
import os
import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from shared.src.interfaces.vector_db_interface import EnhancedVectorDatabase, SearchResult, DocumentChunk
from shared.src.utils.logging import setup_logging, get_logger
from shared.src.utils.config import load_config_from_env

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Vector Database Service")
config = load_config_from_env()

# Initialize Vector DB
vector_db = EnhancedVectorDatabase(use_semantic=config.processing.use_semantic_search)
db_path = config.database.vector_db_path

@app.on_event("startup")
async def startup_event():
    logger.info(f"Loading vector database from {db_path}...")
    try:
        if await vector_db.load(db_path):
            logger.info("Vector database loaded successfully")
        else:
            logger.warning("Could not load vector database (might be new).")
    except Exception as e:
        logger.error(f"Error loading vector DB: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "loaded": vector_db.is_loaded, "stats": await vector_db.get_stats()}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

@app.get("/search")
async def search(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
    try:
        results = await vector_db.search(query, top_k, filters)
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
