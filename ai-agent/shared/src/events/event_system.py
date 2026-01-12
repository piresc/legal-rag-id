"""
Event system for inter-layer communication in Indonesian Legal RAG System
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for system communication"""
    DOCUMENTS_UPDATED = "documents_updated"
    DATABASE_REFRESHED = "database_refreshed"
    QUALITY_ISSUES_DETECTED = "quality_issues_detected"
    PIPELINE_COMPLETED = "pipeline_completed"
    AGENT_QUERY_PROCESSED = "agent_query_processed"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK = "health_check"


@dataclass
class Event:
    """Event data structure"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            metadata=data.get('metadata')
        )


class EventBus:
    """Event bus for system-wide communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
        
    async def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to events of specific type"""
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.info(f"Subscribed to {event_type.value} events")
    
    async def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Unsubscribe from events"""
        async with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.info(f"Unsubscribed from {event_type.value} events")
                except ValueError:
                    logger.warning(f"Callback not found for {event_type.value}")
    
    async def publish(self, event_type: EventType, source: str, data: Dict[str, Any], 
                     metadata: Optional[Dict[str, Any]] = None):
        """Publish event to all subscribers"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            source=source,
            data=data,
            metadata=metadata
        )
        
        async with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            # Notify subscribers
            if event_type in self._subscribers:
                tasks = []
                for callback in self._subscribers[event_type]:
                    try:
                        task = asyncio.create_task(self._safe_callback(callback, event))
                        tasks.append(task)
                    except Exception as e:
                        logger.error(f"Error creating task for callback: {e}")
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Published {event_type.value} event from {source}")
        return event
    
    async def _safe_callback(self, callback: Callable[[Event], None], event: Event):
        """Safely execute callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error in event callback: {e}")
    
    async def get_events(self, event_type: Optional[EventType] = None, 
                        source: Optional[str] = None, 
                        limit: int = 100) -> List[Event]:
        """Get events from history with optional filtering"""
        async with self._lock:
            events = self._event_history.copy()
        
        # Filter by event type
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Filter by source
        if source:
            events = [e for e in events if e.source == source]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get statistics about events"""
        async with self._lock:
            total_events = len(self._event_history)
            event_counts = {}
            source_counts = {}
            
            for event in self._event_history:
                event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
                source_counts[event.source] = source_counts.get(event.source, 0) + 1
        
        return {
            'total_events': total_events,
            'event_types': event_counts,
            'sources': source_counts,
            'subscribers': {event_type.value: len(callbacks) for event_type, callbacks in self._subscribers.items()}
        }


class EventLogger:
    """Event logging utility"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = logging.getLogger("event_logger")
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def log_event(self, event: Event):
        """Log event to file"""
        log_data = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'source': event.source,
            'event_id': event.event_id,
            'data': event.data
        }
        
        self.logger.info(json.dumps(log_data, ensure_ascii=False))


# Global event bus instance
event_bus = EventBus()


# Convenience functions for common events
async def publish_documents_updated(source: str, document_count: int, document_ids: List[str]):
    """Publish documents updated event"""
    await event_bus.publish(
        EventType.DOCUMENTS_UPDATED,
        source,
        {
            'document_count': document_count,
            'document_ids': document_ids,
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_database_refreshed(source: str, chunk_count: int, db_path: str):
    """Publish database refreshed event"""
    await event_bus.publish(
        EventType.DATABASE_REFRESHED,
        source,
        {
            'chunk_count': chunk_count,
            'database_path': db_path,
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_quality_issues_detected(source: str, issues: List[Dict[str, Any]]):
    """Publish quality issues detected event"""
    await event_bus.publish(
        EventType.QUALITY_ISSUES_DETECTED,
        source,
        {
            'issues': issues,
            'issue_count': len(issues),
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_pipeline_completed(source: str, stats: Dict[str, Any]):
    """Publish pipeline completed event"""
    await event_bus.publish(
        EventType.PIPELINE_COMPLETED,
        source,
        {
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_system_error(source: str, error: str, context: Optional[Dict[str, Any]] = None):
    """Publish system error event"""
    await event_bus.publish(
        EventType.SYSTEM_ERROR,
        source,
        {
            'error': error,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_health_check(source: str, status: str, metrics: Dict[str, Any]):
    """Publish health check event"""
    await event_bus.publish(
        EventType.HEALTH_CHECK,
        source,
        {
            'status': status,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    )


async def publish_agent_query_processed(source: str, query_data: Dict[str, Any]):
    """Publish agent query processed event"""
    await event_bus.publish(
        EventType.HEALTH_CHECK,
        source,
        {
            'query_data': query_data,
            'timestamp': datetime.utcnow().isoformat()
        }
    )