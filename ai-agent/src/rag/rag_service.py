"""
RAG Service for Indonesian Legal RAG System AI Agent Layer
"""

import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time

from fastapi import HTTPException, Depends
from pydantic import BaseModel, Field

from shared.src.interfaces.vector_db_interface import (
    VectorDatabaseInterface, DocumentChunk, SearchResult, VectorDatabaseClient
)
from shared.src.events.event_system import event_bus, EventType, publish_agent_query_processed
from shared.src.utils.config import Config, get_config
from shared.src.utils.logging import get_logger, log_async_performance
from shared.src.utils.metrics import track_performance, PerformanceTracker, increment_counter, record_timing
from shared.src.utils.validation import validate_query, is_valid_query

# Import LLM providers
from ..llm.llm_manager import LLMManager, LLMProvider
from ..cache.cache_manager import CacheManager

logger = get_logger(__name__)


class QueryMode(Enum):
    """Query processing modes"""
    BASIC = "basic"  # TF-IDF search only
    API = "api"    # API-based LLM response
    HYBRID = "hybrid"  # Combination of both


@dataclass
class QueryRequest:
    """User query request"""
    question: str
    mode: QueryMode = QueryMode.API
    top_k: int = 5
    provider: Optional[str] = None
    model: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    session_id: Optional[str] = None


@dataclass
class QueryResponse:
    """Query response"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    query_id: str
    mode: QueryMode
    provider: Optional[str] = None
    model: Optional[str] = None
    cached: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class SourceReference:
    """Source reference for citations"""
    chunk_id: str
    doc_id: Optional[str]
    title: Optional[str]
    content: str
    score: float
    metadata: Dict[str, Any]
    url: Optional[str] = None


class LegalRAGService:
    """Main RAG service for Indonesian legal document queries"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Get vector DB URL from environment or config
        vector_db_url = os.getenv("VECTOR_DB_URL", getattr(self.config.database, 'vector_db_url', 'http://localhost:8000'))
        
        # Initialize components
        self.vector_db_client = VectorDatabaseClient(
            base_url=vector_db_url
        )
        self.llm_manager = LLMManager(self.config)
        self.cache_manager = CacheManager(
            redis_url=self.config.database.redis_url,
            default_ttl=3600  # 1 hour
        )
        
        # Service state
        self.is_initialized = False
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'queries_by_mode': {},
            'queries_by_provider': {}
        }
        
        logger.info("RAG service initialized")
    
    async def initialize(self):
        """Initialize the RAG service"""
        if self.is_initialized:
            return
        
        try:
            # Initialize LLM manager
            await self.llm_manager.initialize()
            
            # Test vector database connection
            async with self.vector_db_client as client:
                if not await client.health_check():
                    logger.warning("Vector database health check failed")
            
            self.is_initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}", exc_info=True)
            raise
    
    @log_async_performance("rag_query_processing")
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process user query with RAG"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        query_id = self._generate_query_id(request)
        
        logger.info(f"Processing query: {request.question[:100]}... (ID: {query_id})")
        
        try:
            # Validate query
            if not is_valid_query(request.question):
                raise ValueError("Invalid query format")
            
            # Check cache first
            cached_response = None
            if request.use_cache:
                cached_response = await self._get_cached_response(request)
                if cached_response:
                    logger.info(f"Cache hit for query {query_id}")
                    return cached_response
            
            # Retrieve relevant documents
            relevant_docs = await self._retrieve_relevant_documents(request)
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query: {request.question}")
                return QueryResponse(
                    answer="Maaf, saya tidak dapat menemukan dokumen hukum yang relevan untuk pertanyaan Anda.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    query_id=query_id,
                    mode=request.mode,
                    cached=False
                )
            
            # Generate response based on mode
            if request.mode == QueryMode.BASIC:
                response = await self._generate_basic_response(request, relevant_docs)
            else:
                response = await self._generate_api_response(request, relevant_docs)
            
            # Update processing time
            response.processing_time = time.time() - start_time
            
            # Cache response if enabled
            if request.use_cache:
                await self._cache_response(request, response)
            
            # Update statistics
            self._update_stats(request, response)
            
            # Publish event
            await publish_agent_query_processed(
                "rag_service",
                {
                    "query_id": query_id,
                    "question": request.question,
                    "mode": request.mode.value,
                    "processing_time": response.processing_time,
                    "confidence": response.confidence,
                    "sources_count": len(response.sources)
                }
            )
            
            logger.info(f"Query processed successfully in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing query {query_id}: {str(e)}", exc_info=True)
            
            # Return error response
            return QueryResponse(
                answer=f"Maaf, terjadi kesalahan saat memproses pertanyaan Anda: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                query_id=query_id,
                mode=request.mode,
                cached=False
            )
    
    @track_performance("document_retrieval")
    async def _retrieve_relevant_documents(self, request: QueryRequest) -> List[SearchResult]:
        """Retrieve relevant documents from vector database"""
        try:
            async with self.vector_db_client as client:
                search_results = await client.search(
                    query=request.question,
                    top_k=request.top_k,
                    filters=request.filters
                )
                
                logger.debug(f"Retrieved {len(search_results)} relevant documents")
                return search_results
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def _generate_basic_response(self, request: QueryRequest, relevant_docs: List[SearchResult]) -> QueryResponse:
        """Generate basic search-based response"""
        if not relevant_docs:
            return QueryResponse(
                answer="Tidak ditemukan dokumen yang relevan.",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                query_id="",
                mode=request.mode
            )
        
        # Format sources
        sources = []
        for result in relevant_docs:
            source = SourceReference(
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                title=result.chunk.metadata.get('title'),
                content=result.chunk.content,
                score=result.score,
                metadata=result.chunk.metadata,
                url=result.chunk.metadata.get('url')
            )
            sources.append(asdict(source))
        
        # Create basic response
        answer_parts = ["Saya menemukan beberapa dokumen yang relevan:"]
        
        for i, result in enumerate(relevant_docs[:3]):  # Show top 3 results
            chunk = result.chunk
            metadata = chunk.metadata
            
            source_info = f"\n{i+1}. "
            if metadata.get('doc_type') and metadata.get('doc_number'):
                source_info += f"{metadata['doc_type']} Nomor {metadata['doc_number']}"
            if metadata.get('doc_year'):
                source_info += f" Tahun {metadata['doc_year']}"
            
            if metadata.get('article_number'):
                source_info += f", Pasal {metadata['article_number']}"
            
            source_info += f"\n   Relevansi: {result.score:.3f}"
            source_info += f"\n   Isi: {chunk.content[:200]}..."
            
            answer_parts.append(source_info)
        
        answer_parts.append("\n\nUntuk informasi lebih lengkap, silakan merujuk ke dokumen aslinya.")
        
        return QueryResponse(
            answer="".join(answer_parts),
            sources=sources,
            confidence=max(result.score for result in relevant_docs) if relevant_docs else 0.0,
            processing_time=0.0,
            query_id="",
            mode=request.mode,
            cached=False,
            metadata={}
        )
    
    @track_performance("api_response_generation")
    async def _generate_api_response(self, request: QueryRequest, relevant_docs: List[SearchResult]) -> QueryResponse:
        """Generate API-based LLM response"""
        if not relevant_docs:
            return QueryResponse(
                answer="Maaf, saya tidak dapat menemukan dokumen hukum yang relevan untuk pertanyaan Anda.",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                query_id="",
                mode=request.mode,
                cached=False
            )
        
        # Prepare context for LLM
        context = self._prepare_llm_context(relevant_docs)
        
        # Generate response using LLM
        llm_response = await self.llm_manager.generate_response(
            query=request.question,
            context=context,
            provider=request.provider,
            model=request.model
        )
        
        # Format sources
        sources = []
        for result in relevant_docs:
            source = SourceReference(
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                title=result.chunk.metadata.get('title'),
                content=result.chunk.content,
                score=result.score,
                metadata=result.chunk.metadata,
                url=result.chunk.metadata.get('url')
            )
            sources.append(asdict(source))
        
        # Calculate confidence
        confidence = self._calculate_confidence(llm_response, relevant_docs)
        
        return QueryResponse(
            answer=llm_response.content,
            sources=sources,
            confidence=confidence,
            processing_time=llm_response.processing_time,
            query_id="",
            mode=request.mode,
            provider=llm_response.provider,
            model=llm_response.model,
            cached=False,
            metadata={
                'tokens_used': llm_response.tokens_used,
                'context_length': len(context)
            }
        )
    
    def _prepare_llm_context(self, relevant_docs: List[SearchResult]) -> str:
        """Prepare context for LLM from search results"""
        context_parts = ["Berikut adalah dokumen hukum Indonesia yang relevan:"]
        
        for i, result in enumerate(relevant_docs):
            chunk = result.chunk
            metadata = chunk.metadata
            
            # Add document header
            doc_header = f"\nDokumen {i+1}:"
            if metadata.get('doc_type'):
                doc_header += f" {metadata['doc_type']}"
            if metadata.get('doc_number'):
                doc_header += f" Nomor {metadata['doc_number']}"
            if metadata.get('doc_year'):
                doc_header += f" Tahun {metadata['doc_year']}"
            
            context_parts.append(doc_header)
            
            # Add content
            context_parts.append(f"\n{chunk.content}")
            
            # Add metadata if available
            if metadata.get('article_number'):
                context_parts.append(f"\n(Pasal {metadata['article_number']})")
            
            context_parts.append(f"\n[Relevansi: {result.score:.3f}]")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, llm_response, relevant_docs: List[SearchResult]) -> float:
        """Calculate confidence score for response"""
        # Base confidence from search results
        search_confidence = max(result.score for result in relevant_docs) if relevant_docs else 0.0
        
        # Adjust based on LLM response quality
        if hasattr(llm_response, 'confidence'):
            llm_confidence = llm_response.confidence
        else:
            llm_confidence = 0.8  # Default confidence for API responses
        
        # Weighted combination
        confidence = 0.6 * search_confidence + 0.4 * llm_confidence
        
        return min(confidence, 1.0)
    
    async def _get_cached_response(self, request: QueryRequest) -> Optional[QueryResponse]:
        """Get cached response if available"""
        cache_key = self._generate_cache_key(request)
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            increment_counter("cache_hits_total")
            response = QueryResponse(**cached_data)
            response.cached = True
            return response
        
        return None
    
    async def _cache_response(self, request: QueryRequest, response: QueryResponse):
        """Cache response for future use"""
        cache_key = self._generate_cache_key(request)
        
        # Convert response to dict for caching
        response_dict = asdict(response)
        response_dict['cached'] = True
        
        await self.cache_manager.set(cache_key, response_dict)
    
    def _generate_cache_key(self, request: QueryRequest) -> str:
        """Generate cache key for request"""
        mode_value = request.mode.value if hasattr(request.mode, 'value') else request.mode
        
        key_data = {
            'question': request.question,
            'mode': mode_value,
            'top_k': request.top_k,
            'provider': request.provider,
            'model': request.model,
            'filters': request.filters
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"rag_query:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _generate_query_id(self, request: QueryRequest) -> str:
        """Generate unique query ID"""
        timestamp = int(time.time())
        question_hash = hashlib.md5(request.question.encode()).hexdigest()[:8]
        return f"query_{timestamp}_{question_hash}"
    
    def _update_stats(self, request: QueryRequest, response: QueryResponse):
        """Update service statistics"""
        increment_counter("search_requests_total")
        
        self.stats['total_queries'] += 1
        self.stats['successful_queries'] += 1
        
        # Update mode statistics
        mode_key = request.mode.value
        self.stats['queries_by_mode'][mode_key] = self.stats['queries_by_mode'].get(mode_key, 0) + 1
        
        # Update provider statistics
        if response.provider:
            provider_key = response.provider
            self.stats['queries_by_provider'][provider_key] = self.stats['queries_by_provider'].get(provider_key, 0) + 1
        
        # Update average response time
        total_time = self.stats['average_response_time'] * (self.stats['total_queries'] - 1)
        self.stats['average_response_time'] = (total_time + response.processing_time) / self.stats['total_queries']
        
        # Record timing
        record_timing("search_duration_seconds", response.processing_time)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            'rag_service': True,
            'vector_db': False,
            'llm_manager': False,
            'cache': False,
            'overall': False
        }

        try:
            # Check vector database
            try:
                async with self.vector_db_client as client:
                    health_status['vector_db'] = await client.health_check()
            except:
                health_status['vector_db'] = False

            # Check LLM manager
            try:
                health_status['llm_manager'] = await self.llm_manager.health_check()
            except:
                health_status['llm_manager'] = False

            # Check cache
            try:
                health_status['cache'] = await self.cache_manager.health_check()
            except:
                health_status['cache'] = False

            # Overall is True if rag_service is True and at least one check passed
            health_status['overall'] = health_status['rag_service'] and any([
                health_status['vector_db'],
                health_status['llm_manager'],
                health_status['cache']
            ])

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_status['rag_service'] = False
            health_status['overall'] = False

        return health_status
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'rag_stats': self.stats,
            'llm_stats': await self.llm_manager.get_stats(),
            'cache_stats': await self.cache_manager.get_stats(),
            'is_initialized': self.is_initialized
        }
    
    async def clear_cache(self):
        """Clear response cache"""
        await self.cache_manager.clear_pattern("rag_query:*")
        logger.info("RAG service cache cleared")
    
    async def reset_stats(self):
        """Reset service statistics"""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'queries_by_mode': {},
            'queries_by_provider': {}
        }
        logger.info("RAG service statistics reset")


# FastAPI models for API endpoints
class QueryRequestModel(BaseModel):
    """FastAPI model for query requests"""
    question: str = Field(..., description="User's legal question", min_length=3, max_length=1000)
    mode: str = Field("api", description="Query mode: basic, api, or hybrid")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    provider: Optional[str] = Field(None, description="LLM provider")
    model: Optional[str] = Field(None, description="LLM model")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    use_cache: bool = Field(True, description="Use response caching")
    session_id: Optional[str] = Field(None, description="Session identifier")


class QueryResponseModel(BaseModel):
    """FastAPI model for query responses"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    query_id: str
    mode: str
    provider: Optional[str]
    model: Optional[str]
    cached: bool
    metadata: Optional[Dict[str, Any]] = None


# Global service instance
_rag_service: Optional[LegalRAGService] = None


def get_rag_service(config: Optional[Config] = None) -> LegalRAGService:
    """Get global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = LegalRAGService(config)
    return _rag_service