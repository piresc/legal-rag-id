"""
Cache Manager for Indonesian Legal RAG System
Manages Redis-based caching for RAG responses
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("aioredis not available, cache will use in-memory fallback")

from shared.src.utils.config import Config, get_config
from shared.src.utils.metrics import track_performance, increment_counter

logger = logging.getLogger(__name__)


class InMemoryCache:
    """In-memory fallback cache when Redis is not available"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ttls: Dict[str, datetime] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.ttls:
            if datetime.now() > self.ttls[key]:
                # Expired, remove it
                del self.cache[key]
                del self.ttls[key]
                return None
        
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        self.cache[key] = value
        self.ttls[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def delete(self, key: str):
        """Delete key from cache"""
        self.cache.pop(key, None)
        self.ttls.pop(key, None)
    
    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_delete:
            await self.delete(key)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_type': 'in_memory',
            'total_keys': len(self.cache),
            'memory_usage': len(json.dumps(self.cache))
        }


class CacheManager:
    """Manages caching using Redis with in-memory fallback"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis: Optional[aioredis.Redis] = None
        self.in_memory_cache: Optional[InMemoryCache] = None
        self.use_redis = False
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'pattern_clears': 0
        }
    
    async def initialize(self):
        """Initialize cache connection"""
        if REDIS_AVAILABLE:
            try:
                # Parse Redis URL
                # Format: redis://[:password@]host:port/db
                if self.redis_url.startswith('redis://'):
                    self.redis = await aioredis.from_url(self.redis_url)
                    # Test connection
                    await self.redis.ping()
                    self.use_redis = True
                    logger.info("Cache manager connected to Redis")
                else:
                    raise ValueError("Invalid Redis URL format")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using in-memory cache")
                self.use_redis = False
        else:
            logger.warning("Redis not available, using in-memory cache")
            self.use_redis = False
        
        if not self.use_redis:
            self.in_memory_cache = InMemoryCache()
            logger.info("Cache manager using in-memory cache")
    
    @track_performance("cache_get")
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.use_redis and self.redis:
                value = await self.redis.get(key)
                if value:
                    self.stats['hits'] += 1
                    increment_counter("cache_hits_total")
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value.decode('utf-8')
                else:
                    self.stats['misses'] += 1
                    increment_counter("cache_misses_total")
                    return None
            elif self.in_memory_cache:
                value = await self.in_memory_cache.get(key)
                if value:
                    self.stats['hits'] += 1
                    increment_counter("cache_hits_total")
                else:
                    self.stats['misses'] += 1
                    increment_counter("cache_misses_total")
                return value
            else:
                # Not initialized
                await self.initialize()
                return await self.get(key)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.stats['misses'] += 1
            return None
    
    @track_performance("cache_set")
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            self.stats['sets'] += 1
            increment_counter("cache_sets_total")
            
            if self.use_redis and self.redis:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                await self.redis.setex(key, ttl, value)
            elif self.in_memory_cache:
                await self.in_memory_cache.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.stats['deletes'] += 1
            increment_counter("cache_deletes_total")
            
            if self.use_redis and self.redis:
                await self.redis.delete(key)
            elif self.in_memory_cache:
                await self.in_memory_cache.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
    
    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        try:
            self.stats['pattern_clears'] += 1
            
            if self.use_redis and self.redis:
                keys = []
                async for key in self.redis.iscan(match=pattern):
                    keys.append(key)
                if keys:
                    await self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching pattern: {pattern}")
            elif self.in_memory_cache:
                await self.in_memory_cache.clear_pattern(pattern)
                logger.info(f"Cleared keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Error clearing pattern: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.use_redis and self.redis:
                return await self.redis.exists(key) > 0
            elif self.in_memory_cache:
                return await self.in_memory_cache.get(key) is not None
            return False
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        try:
            if self.use_redis and self.redis:
                return await self.redis.ttl(key)
            return None
        except Exception as e:
            logger.error(f"Error getting TTL: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'healthy': False,
            'cache_type': 'none',
            'connected': False
        }
        
        try:
            if self.use_redis and self.redis:
                await self.redis.ping()
                health.update({
                    'healthy': True,
                    'cache_type': 'redis',
                    'connected': True
                })
            elif self.in_memory_cache:
                health.update({
                    'healthy': True,
                    'cache_type': 'in_memory',
                    'connected': True
                })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health['error'] = str(e)
        
        return health
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        base_stats = {
            'cache_type': 'redis' if self.use_redis else 'in_memory',
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'pattern_clears': self.stats['pattern_clears'],
            'hit_rate': self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
        }
        
        if self.use_redis and self.redis:
            try:
                info = await self.redis.info()
                base_stats.update({
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'memory_used': info.get('used_memory_human', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        elif self.in_memory_cache:
            mem_stats = await self.in_memory_cache.get_stats()
            base_stats.update(mem_stats)
        
        return base_stats
    
    async def clear_all(self):
        """Clear all cached data"""
        try:
            if self.use_redis and self.redis:
                await self.redis.flushdb()
            elif self.in_memory_cache:
                await self.clear_pattern("*")
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def close(self):
        """Close cache connection"""
        if self.redis:
            await self.redis.close()
        logger.info("Cache manager closed")