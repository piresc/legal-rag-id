"""
Metrics collection utilities for Indonesian Legal RAG System
"""

import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    count: int
    sum: float
    min: float
    max: float
    avg: float
    last_value: float
    last_updated: datetime


class MetricsCollector:
    """Thread-safe metrics collector"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Predefined metric names
        self.METRIC_NAMES = {
            'search_requests_total': 'Total number of search requests',
            'search_duration_seconds': 'Search request duration in seconds',
            'search_results_count': 'Number of search results returned',
            'api_requests_total': 'Total number of API requests',
            'api_duration_seconds': 'API request duration in seconds',
            'api_errors_total': 'Total number of API errors',
            'documents_processed_total': 'Total number of documents processed',
            'document_processing_duration_seconds': 'Document processing duration',
            'vector_db_size': 'Vector database size in bytes',
            'vector_db_chunks_total': 'Total number of chunks in vector database',
            'cache_hits_total': 'Total number of cache hits',
            'cache_misses_total': 'Total number of cache misses',
            'system_memory_usage_bytes': 'System memory usage in bytes',
            'system_cpu_usage_percent': 'System CPU usage percentage'
        }
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            self._counters[name] += value
            self._record_metric(name, value, labels, 'counter')
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        with self._lock:
            self._gauges[name] = value
            self._record_metric(name, value, labels, 'gauge')
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram value"""
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 values per histogram
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
            self._record_metric(name, value, labels, 'histogram')
    
    def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record timing (convenience method for histogram)"""
        self.observe_histogram(name, duration, labels)
    
    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]], metric_type: str):
        """Record metric value with timestamp"""
        metric = MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            labels={**(labels or {}), 'metric_type': metric_type}
        )
        self._metrics[name].append(metric)
    
    def get_counter(self, name: str) -> float:
        """Get counter value"""
        with self._lock:
            return self._counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self._lock:
            return self._gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Optional[MetricSummary]:
        """Get histogram statistics"""
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return None
            
            return MetricSummary(
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values),
                last_value=values[-1] if values else 0.0,
                last_updated=datetime.utcnow()
            )
    
    def get_metric_history(self, name: str, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get metric history"""
        with self._lock:
            history = list(self._metrics.get(name, []))
            
            if since:
                history = [m for m in history if m.timestamp >= since]
            
            return history
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: self.get_histogram_stats(name).__dict__ 
                    for name in self._histograms
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            return summary
    
    def reset_metric(self, name: str):
        """Reset a specific metric"""
        with self._lock:
            if name in self._counters:
                self._counters[name] = 0.0
            if name in self._gauges:
                self._gauges[name] = 0.0
            if name in self._histograms:
                self._histograms[name].clear()
            if name in self._metrics:
                self._metrics[name].clear()
    
    def reset_all_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._metrics.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_performance(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function performance"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing
                collector.record_timing(metric_name, duration, labels)
                
                # Increment success counter
                success_metric = f"{metric_name}_success_total"
                collector.increment_counter(success_metric, 1.0, labels)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record timing even for failures
                collector.record_timing(metric_name, duration, labels)
                
                # Increment error counter
                error_metric = f"{metric_name}_error_total"
                collector.increment_counter(error_metric, 1.0, labels)
                
                logger.error(f"Function {func.__name__} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator


def track_async_performance(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track async function performance"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing
                collector.record_timing(metric_name, duration, labels)
                
                # Increment success counter
                success_metric = f"{metric_name}_success_total"
                collector.increment_counter(success_metric, 1.0, labels)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record timing even for failures
                collector.record_timing(metric_name, duration, labels)
                
                # Increment error counter
                error_metric = f"{metric_name}_error_total"
                collector.increment_counter(error_metric, 1.0, labels)
                
                logger.error(f"Async function {func.__name__} failed: {str(e)}")
                raise
        
        return wrapper
    return decorator


class SystemMetrics:
    """System-level metrics collection"""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or get_metrics_collector()
        self._running = False
        self._thread = None
    
    def start_collection(self, interval: int = 30):
        """Start system metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, args=(interval,))
        self._thread.daemon = True
        self._thread.start()
        logger.info("Started system metrics collection")
    
    def stop_collection(self):
        """Stop system metrics collection"""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Stopped system metrics collection")
    
    def _collect_loop(self, interval: int):
        """Collection loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            import psutil
            import os
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.collector.set_gauge('system_memory_usage_bytes', memory.used)
            self.collector.set_gauge('system_memory_available_bytes', memory.available)
            self.collector.set_gauge('system_memory_usage_percent', memory.percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.collector.set_gauge('system_cpu_usage_percent', cpu_percent)
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            self.collector.set_gauge('process_memory_usage_bytes', process.memory_info().rss)
            self.collector.set_gauge('process_cpu_usage_percent', process.cpu_percent())
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.collector.set_gauge('system_disk_usage_bytes', disk.used)
            self.collector.set_gauge('system_disk_available_bytes', disk.free)
            self.collector.set_gauge('system_disk_usage_percent', (disk.used / disk.total) * 100)
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


# Context manager for performance tracking
class PerformanceTracker:
    """Context manager for performance tracking"""
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.labels = labels
        self.collector = get_metrics_collector()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timing(self.metric_name, duration, self.labels)
            
            if exc_type is None:
                success_metric = f"{self.metric_name}_success_total"
                self.collector.increment_counter(success_metric, 1.0, self.labels)
            else:
                error_metric = f"{self.metric_name}_error_total"
                self.collector.increment_counter(error_metric, 1.0, self.labels)


# Convenience functions
def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
    """Increment counter metric"""
    get_metrics_collector().increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Set gauge metric"""
    get_metrics_collector().set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Observe histogram metric"""
    get_metrics_collector().observe_histogram(name, value, labels)


def record_timing(name: str, duration: float, labels: Optional[Dict[str, str]] = None):
    """Record timing metric"""
    get_metrics_collector().record_timing(name, duration, labels)


# Initialize system metrics
_system_metrics = SystemMetrics()


def start_system_metrics_collection(interval: int = 30):
    """Start system metrics collection"""
    _system_metrics.start_collection(interval)


def stop_system_metrics_collection():
    """Stop system metrics collection"""
    _system_metrics.stop_collection()