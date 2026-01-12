"""
Shared interface for vector database operations
"""

import asyncio
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Document chunk data structure"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    level: Optional[str] = None  # document, chapter, article, paragraph


@dataclass
class SearchResult:
    """Search result data structure"""
    chunk: DocumentChunk
    score: float
    rank: int
    explanation: Optional[str] = None


@dataclass
class DatabaseStats:
    """Database statistics"""
    total_chunks: int
    total_documents: int
    vocabulary_size: int
    index_type: str
    last_updated: str
    memory_usage_mb: float


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to database"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get specific chunk by ID"""
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk by ID"""
        pass
    
    @abstractmethod
    async def update_chunk(self, chunk_id: str, chunk: DocumentChunk) -> bool:
        """Update existing chunk"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> DatabaseStats:
        """Get database statistics"""
        pass
    
    @abstractmethod
    async def save(self, file_path: str) -> bool:
        """Save database to file"""
        pass
    
    @abstractmethod
    async def load(self, file_path: str) -> bool:
        """Load database from file"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health"""
        pass


class EnhancedVectorDatabase(VectorDatabaseInterface):
    """Enhanced vector database with TF-IDF and semantic search"""
    
    def __init__(self, use_semantic: bool = True):
        self.use_semantic = use_semantic
        self.chunks: List[DocumentChunk] = []
        self.chunk_index: Dict[str, int] = {}  # chunk_id -> index
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_model = None
        self.semantic_embeddings = None
        self.is_loaded = False
        
        # Try to import required libraries
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            self.TfidfVectorizer = TfidfVectorizer
            self.cosine_similarity = cosine_similarity
            self.np = np
            
            # Initialize TF-IDF vectorizer for Indonesian text
            self.tfidf_vectorizer = self.TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words=[
                    'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'dalam', 'pada', 
                    'adalah', 'ini', 'itu', 'dengan', 'akan', 'bagi', 'serta', 
                    'atau', 'tetapi', 'melainkan', 'yaitu', 'ialah', 'yakni'
                ],
                lowercase=True
            )
            
        except ImportError as e:
            logger.warning(f"Could not import required libraries: {e}")
        
        # Try to load semantic model if requested
        if use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Loaded semantic embedding model")
            except ImportError:
                logger.warning("Could not load sentence-transformers, semantic search disabled")
                self.use_semantic = False
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to database"""
        try:
            for chunk in chunks:
                if chunk.chunk_id not in self.chunk_index:
                    self.chunk_index[chunk.chunk_id] = len(self.chunks)
                    self.chunks.append(chunk)
                else:
                    # Update existing chunk
                    index = self.chunk_index[chunk.chunk_id]
                    self.chunks[index] = chunk
            
            # Rebuild indices
            await self._build_indices()
            self.is_loaded = True
            logger.info(f"Added {len(chunks)} chunks to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return False
    
    async def search(self, query: str, top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks"""
        if not self.is_loaded or not self.chunks:
            return []
        
        try:
            # Apply filters if provided
            filtered_chunks = self.chunks
            if filters:
                filtered_chunks = self._apply_filters(self.chunks, filters)
            
            if not filtered_chunks:
                return []
            
            # TF-IDF search
            tfidf_scores = await self._tfidf_search(query, filtered_chunks)
            
            # Semantic search if available
            semantic_scores = None
            if self.use_semantic and self.semantic_embeddings is not None:
                semantic_scores = await self._semantic_search(query, filtered_chunks)
            
            # Combine scores
            results = await self._combine_scores(
                query, filtered_chunks, tfidf_scores, semantic_scores
            )
            
            # Sort and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get specific chunk by ID"""
        if chunk_id in self.chunk_index:
            index = self.chunk_index[chunk_id]
            return self.chunks[index]
        return None
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk by ID"""
        if chunk_id not in self.chunk_index:
            return False
        
        try:
            index = self.chunk_index[chunk_id]
            del self.chunks[index]
            del self.chunk_index[chunk_id]
            
            # Rebuild index
            for i, chunk in enumerate(self.chunks):
                self.chunk_index[chunk.chunk_id] = i
            
            # Rebuild indices
            await self._build_indices()
            logger.info(f"Deleted chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunk: {e}")
            return False
    
    async def update_chunk(self, chunk_id: str, chunk: DocumentChunk) -> bool:
        """Update existing chunk"""
        if chunk_id not in self.chunk_index:
            return False
        
        try:
            index = self.chunk_index[chunk_id]
            self.chunks[index] = chunk
            
            # Rebuild indices
            await self._build_indices()
            logger.info(f"Updated chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chunk: {e}")
            return False
    
    async def get_stats(self) -> DatabaseStats:
        """Get database statistics"""
        import psutil
        import os
        
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        vocabulary_size = 0
        if self.tfidf_vectorizer and hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            vocabulary_size = len(self.tfidf_vectorizer.vocabulary_)
        
        return DatabaseStats(
            total_chunks=len(self.chunks),
            total_documents=len(set(chunk.doc_id for chunk in self.chunks if chunk.doc_id)),
            vocabulary_size=vocabulary_size,
            index_type="tf-idf" + (" + semantic" if self.use_semantic else ""),
            last_updated="unknown",
            memory_usage_mb=memory_mb
        )
    
    async def save(self, file_path: str) -> bool:
        """Save database to file"""
        try:
            db_data = {
                'chunks': [self._chunk_to_dict(chunk) for chunk in self.chunks],
                'chunk_index': self.chunk_index,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'semantic_embeddings': self.semantic_embeddings,
                'use_semantic': self.use_semantic,
                'is_loaded': self.is_loaded
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(db_data, f)
            
            logger.info(f"Saved database to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False
    
    async def load(self, file_path: str) -> bool:
        """Load database from file"""
        try:
            with open(file_path, 'rb') as f:
                db_data = pickle.load(f)

            # Check for old format (robust_vector_database.pkl format)
            if 'documents' in db_data and 'vectors' in db_data:
                logger.info("Detected old format vector database, converting...")
                return await self._load_old_format(db_data, file_path)

            # New format
            self.chunks = [self._dict_to_chunk(chunk_dict) for chunk_dict in db_data['chunks']]
            self.chunk_index = db_data['chunk_index']
            self.tfidf_vectorizer = db_data['tfidf_vectorizer']
            self.tfidf_matrix = db_data['tfidf_matrix']
            self.semantic_embeddings = db_data.get('semantic_embeddings')
            self.use_semantic = db_data.get('use_semantic', False)
            self.is_loaded = db_data.get('is_loaded', True)

            logger.info(f"Loaded database from {file_path} with {len(self.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

    async def _load_old_format(self, old_db: dict, file_path: str) -> bool:
        """Load old format vector database"""
        try:
            import hashlib

            documents = old_db.get('documents', [])
            document_metadata = old_db.get('document_metadata', [])

            # Create chunks from documents
            self.chunks = []
            self.chunk_index = {}

            for i, (doc, metadata) in enumerate(zip(documents, document_metadata)):
                chunk_id = f"chunk_{i}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=metadata.get('doc_id', f"doc_{i}") if isinstance(metadata, dict) else f"doc_{i}",
                    content=str(doc),
                    metadata=metadata if isinstance(metadata, dict) else {'title': str(doc)}
                )
                self.chunks.append(chunk)
                self.chunk_index[chunk_id] = i

            # Initialize TF-IDF vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(norm='l2')
            texts = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Set other properties
            self.semantic_embeddings = None
            self.use_semantic = False
            self.is_loaded = True

            logger.info(f"Loaded old format database from {file_path} with {len(self.chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error loading old format database: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check database health"""
        return self.is_loaded and len(self.chunks) > 0
    
    async def _build_indices(self):
        """Build search indices"""
        if not self.chunks:
            return
        
        # Build TF-IDF matrix
        if self.tfidf_vectorizer:
            texts = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Build semantic embeddings
        if self.use_semantic and self.semantic_model:
            texts = [chunk.content for chunk in self.chunks]
            self.semantic_embeddings = self.semantic_model.encode(texts)
    
    async def _tfidf_search(self, query: str, chunks: List[DocumentChunk]) -> List[float]:
        """Perform TF-IDF search"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return [0.0] * len(chunks)
        
        # Get chunk indices
        chunk_indices = [self.chunk_index[chunk.chunk_id] for chunk in chunks]
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = self.cosine_similarity(query_vector, self.tfidf_matrix[chunk_indices])[0]
        return similarities.tolist()
    
    async def _semantic_search(self, query: str, chunks: List[DocumentChunk]) -> List[float]:
        """Perform semantic search"""
        if not self.semantic_model or self.semantic_embeddings is None:
            return [0.0] * len(chunks)
        
        # Get chunk indices
        chunk_indices = [self.chunk_index[chunk.chunk_id] for chunk in chunks]
        
        # Encode query
        query_embedding = self.semantic_model.encode([query])
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.semantic_embeddings[chunk_indices])[0]
        return similarities.tolist()
    
    async def _combine_scores(self, query: str, chunks: List[DocumentChunk], 
                             tfidf_scores: List[float], 
                             semantic_scores: Optional[List[float]]) -> List[SearchResult]:
        """Combine TF-IDF and semantic scores"""
        results = []
        
        for i, chunk in enumerate(chunks):
            tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.0
            
            if semantic_scores and i < len(semantic_scores):
                semantic_score = semantic_scores[i]
                # Weighted combination
                combined_score = 0.6 * semantic_score + 0.4 * tfidf_score
            else:
                combined_score = tfidf_score
            
            results.append(SearchResult(
                chunk=chunk,
                score=combined_score,
                rank=0  # Will be set after sorting
            ))
        
        return results
    
    def _apply_filters(self, chunks: List[DocumentChunk], 
                      filters: Dict[str, Any]) -> List[DocumentChunk]:
        """Apply filters to chunks"""
        filtered = chunks
        
        for key, value in filters.items():
            if key == 'doc_type':
                filtered = [c for c in filtered if c.metadata.get('doc_type') == value]
            elif key == 'doc_year':
                filtered = [c for c in filtered if c.metadata.get('doc_year') == value]
            elif key == 'level':
                filtered = [c for c in filtered if c.level == value]
        
        return filtered
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization"""
        return {
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'embedding': chunk.embedding,
            'doc_id': chunk.doc_id,
            'level': chunk.level
        }
    
    def _dict_to_chunk(self, chunk_dict: Dict[str, Any]) -> DocumentChunk:
        """Convert dictionary to chunk object"""
        return DocumentChunk(
            chunk_id=chunk_dict['chunk_id'],
            content=chunk_dict['content'],
            metadata=chunk_dict['metadata'],
            embedding=chunk_dict.get('embedding'),
            doc_id=chunk_dict.get('doc_id'),
            level=chunk_dict.get('level')
        )


class VectorDatabaseClient:
    """Client for interacting with vector database service"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks via API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        params = {'query': query, 'top_k': top_k}
        if filters:
            params['filters'] = filters
        
        async with self.session.get(f"{self.base_url}/search", params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []
                for result in data:
                    # Convert chunk dict to DocumentChunk object
                    chunk_data = result.get('chunk', {})
                    if isinstance(chunk_data, dict):
                        chunk = DocumentChunk(
                            chunk_id=chunk_data.get('chunk_id', ''),
                            doc_id=chunk_data.get('doc_id'),
                            content=chunk_data.get('content', ''),
                            metadata=chunk_data.get('metadata', {})
                        )
                        result['chunk'] = chunk
                    results.append(SearchResult(**result))
                return results
            else:
                raise Exception(f"Search failed: {response.status}")
    
    async def health_check(self) -> bool:
        """Check database health via API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        async with self.session.get(f"{self.base_url}/health") as response:
            return response.status == 200