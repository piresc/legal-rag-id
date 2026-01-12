"""
Logging utilities for Indonesian Legal RAG System
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self, context: dict = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def update_context(self, **kwargs):
        """Update context"""
        self.context.update(kwargs)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    context: Optional[dict] = None
) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        for handler in root_logger.handlers:
            handler.addFilter(context_filter)
    
    return root_logger


def get_logger(name: str, context: Optional[dict] = None) -> logging.Logger:
    """Get logger with optional context"""
    logger = logging.getLogger(name)
    
    if context:
        # Add context filter
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with dynamic context"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add extra context to log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs
    
    def with_context(self, **kwargs):
        """Create new adapter with additional context"""
        new_extra = self.extra.copy()
        new_extra.update(kwargs)
        return LoggerAdapter(self.logger, new_extra)


def configure_structured_logging(config):
    """Configure structured logging based on config"""
    return setup_logging(
        level=config.monitoring.log_level,
        log_file=config.monitoring.log_file,
        json_format=config.is_production(),
        context={
            'service': 'indonesian-legal-rag',
            'environment': config.environment,
            'version': '1.0.0'
        }
    )


# Performance logging decorator
def log_performance(logger: Optional[logging.Logger] = None):
    """Decorator to log function performance"""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_logger = logger or logging.getLogger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                function_logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'success': True
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                function_logger.error(
                    f"Function {func.__name__} failed with error: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'success': False,
                        'error': str(e)
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


def log_async_performance(logger: Optional[logging.Logger] = None):
    """Decorator to log async function performance"""
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            # Get logger - use module logger if parameter is None or not a logger
            if logger is None:
                function_logger = logging.getLogger(func.__module__)
            elif isinstance(logger, str):
                function_logger = logging.getLogger(logger)
            else:
                function_logger = logger

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                function_logger.info(
                    f"Async function {func.__name__} completed successfully",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'success': True,
                        'async': True
                    }
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                function_logger.error(
                    f"Async function {func.__name__} failed with error: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'success': False,
                        'error': str(e),
                        'async': True
                    },
                    exc_info=True
                )

                raise

        return wrapper
    return decorator


# Request logging middleware for FastAPI
def create_request_logging_middleware(logger: Optional[logging.Logger] = None):
    """Create FastAPI middleware for request logging"""
    from fastapi import Request, Response
    import time
    
    request_logger = logger or logging.getLogger("request")
    
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        request_logger.info(
            f"Request started: {request.method} {request.url}",
            extra={
                'method': request.method,
                'url': str(request.url),
                'client_ip': request.client.host if request.client else None,
                'user_agent': request.headers.get("user-agent"),
                'request_id': request.headers.get("x-request-id")
            }
        )
        
        try:
            response = await call_next(request)
            execution_time = time.time() - start_time
            
            # Log response
            request_logger.info(
                f"Request completed: {request.method} {request.url} - {response.status_code}",
                extra={
                    'method': request.method,
                    'url': str(request.url),
                    'status_code': response.status_code,
                    'execution_time': execution_time,
                    'success': response.status_code < 400
                }
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            request_logger.error(
                f"Request failed: {request.method} {request.url} - {str(e)}",
                extra={
                    'method': request.method,
                    'url': str(request.url),
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                },
                exc_info=True
            )
            
            raise
    
    return log_requests


# Error logging utility
def log_error(logger: logging.Logger, error: Exception, context: Optional[dict] = None):
    """Log error with context"""
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {}
    }
    
    logger.error(
        f"Error occurred: {error_data['error_type']}: {error_data['error_message']}",
        extra=error_data,
        exc_info=True
    )


# Audit logging
def log_audit_event(logger: logging.Logger, event_type: str, user_id: Optional[str] = None,
                   resource: Optional[str] = None, action: Optional[str] = None,
                   details: Optional[dict] = None):
    """Log audit event"""
    audit_data = {
        'event_type': event_type,
        'user_id': user_id,
        'resource': resource,
        'action': action,
        'details': details or {},
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(
        f"Audit event: {event_type}",
        extra=audit_data
    )


# Security logging
def log_security_event(logger: logging.Logger, event_type: str, severity: str,
                      source_ip: Optional[str] = None, details: Optional[dict] = None):
    """Log security event"""
    security_data = {
        'security_event': True,
        'event_type': event_type,
        'severity': severity,
        'source_ip': source_ip,
        'details': details or {},
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if severity.upper() in ['HIGH', 'CRITICAL']:
        logger.error(
            f"Security event ({severity}): {event_type}",
            extra=security_data
        )
    else:
        logger.warning(
            f"Security event ({severity}): {event_type}",
            extra=security_data
        )