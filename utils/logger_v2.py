"""
Structured Logging System (JSON Format)

Provides comprehensive logging for:
- Query-level operations
- System-level events
- Error tracking
- Performance metrics
"""

import logging
import json
import time
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
import uuid


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exc()
            }

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


class QueryLogger:
    """
    Specialized logger for query-level operations

    Tracks full query lifecycle with performance metrics
    """

    def __init__(self, base_logger: logging.Logger):
        self.base_logger = base_logger

    def log_query(self,
                  query_id: str,
                  query_text: str,
                  parsed_query: Dict[str, Any],
                  results: list,
                  latency_ms: float,
                  components_used: list,
                  fallbacks_triggered: list = None,
                  user_id: str = None,
                  additional_info: Dict = None):
        """
        Log complete query information

        Args:
            query_id: Unique query identifier
            query_text: Raw query string
            parsed_query: Parsed query structure
            results: List of results
            latency_ms: Total query latency
            components_used: List of components used
            fallbacks_triggered: List of fallbacks used
            user_id: Optional user identifier
            additional_info: Additional fields
        """
        log_data = {
            'event_type': 'query',
            'query_id': query_id,
            'query_text': query_text,
            'parsed_query': {
                'entities': parsed_query.get('entities', []),
                'num_entities': len(parsed_query.get('entities', [])),
                'domain': parsed_query.get('domain'),
                'time_period': parsed_query.get('time_period')
            },
            'results': {
                'num_results': len(results),
                'top_score': max([r.get('score', 0) for r in results]) if results else 0,
                'avg_score': sum([r.get('score', 0) for r in results]) / len(results) if results else 0
            },
            'performance': {
                'latency_ms': latency_ms,
                'latency_category': self._categorize_latency(latency_ms)
            },
            'components_used': components_used,
            'fallbacks_triggered': fallbacks_triggered or [],
            'user_id': user_id
        }

        if additional_info:
            log_data.update(additional_info)

        self.base_logger.info(
            f"Query processed: {query_text[:50]}...",
            extra={'extra_fields': log_data}
        )

    def log_query_failure(self,
                         query_id: str,
                         query_text: str,
                         error_type: str,
                         error_message: str,
                         stack_trace: str = None):
        """Log query failure"""
        log_data = {
            'event_type': 'query_failure',
            'query_id': query_id,
            'query_text': query_text,
            'error': {
                'type': error_type,
                'message': error_message,
                'stack_trace': stack_trace
            }
        }

        self.base_logger.error(
            f"Query failed: {error_message}",
            extra={'extra_fields': log_data}
        )

    def _categorize_latency(self, latency_ms: float) -> str:
        """Categorize latency for analysis"""
        if latency_ms < 50:
            return 'excellent'
        elif latency_ms < 100:
            return 'good'
        elif latency_ms < 200:
            return 'acceptable'
        elif latency_ms < 500:
            return 'slow'
        else:
            return 'very_slow'


class SystemLogger:
    """Logger for system-level events"""

    def __init__(self, base_logger: logging.Logger):
        self.base_logger = base_logger

    def log_component_init(self,
                          component: str,
                          success: bool,
                          load_time_ms: float = None,
                          error: str = None):
        """Log component initialization"""
        log_data = {
            'event_type': 'component_init',
            'component': component,
            'success': success,
            'load_time_ms': load_time_ms,
            'error': error
        }

        if success:
            self.base_logger.info(
                f"Component initialized: {component}",
                extra={'extra_fields': log_data}
            )
        else:
            self.base_logger.error(
                f"Component initialization failed: {component}",
                extra={'extra_fields': log_data}
            )

    def log_cache_stats(self,
                       cache_name: str,
                       hit_rate: float,
                       size: int,
                       max_size: int):
        """Log cache statistics"""
        log_data = {
            'event_type': 'cache_stats',
            'cache_name': cache_name,
            'hit_rate': hit_rate,
            'size': size,
            'max_size': max_size,
            'utilization': size / max_size if max_size > 0 else 0
        }

        self.base_logger.info(
            f"Cache stats: {cache_name}",
            extra={'extra_fields': log_data}
        )

    def log_resource_usage(self,
                          memory_mb: float,
                          cpu_percent: float,
                          disk_io_mb: float = None):
        """Log system resource usage"""
        log_data = {
            'event_type': 'resource_usage',
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'disk_io_mb': disk_io_mb
        }

        self.base_logger.debug(
            "Resource usage snapshot",
            extra={'extra_fields': log_data}
        )


class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self, base_logger: logging.Logger):
        self.base_logger = base_logger
        self.latency_samples = []
        self.max_samples = 1000

    def log_latency(self,
                   operation: str,
                   latency_ms: float,
                   metadata: Dict = None):
        """Log operation latency"""
        self.latency_samples.append({
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })

        if len(self.latency_samples) > self.max_samples:
            self.latency_samples.pop(0)

        log_data = {
            'event_type': 'performance',
            'operation': operation,
            'latency_ms': latency_ms,
            'metadata': metadata or {}
        }

        self.base_logger.debug(
            f"Performance: {operation}",
            extra={'extra_fields': log_data}
        )

    def get_percentiles(self, operation: str = None) -> Dict[str, float]:
        """Calculate latency percentiles"""
        import numpy as np

        # Filter by operation if specified
        if operation:
            samples = [s['latency_ms'] for s in self.latency_samples
                      if s['operation'] == operation]
        else:
            samples = [s['latency_ms'] for s in self.latency_samples]

        if not samples:
            return {}

        return {
            'p50': float(np.percentile(samples, 50)),
            'p95': float(np.percentile(samples, 95)),
            'p99': float(np.percentile(samples, 99)),
            'mean': float(np.mean(samples)),
            'max': float(np.max(samples)),
            'count': len(samples)
        }


class HeritageLogger:
    """
    Main logger for Heritage Document Recommender

    Provides structured logging with multiple specialized loggers
    """

    def __init__(self,
                 name: str = 'heritage_recommender',
                 log_file: str = None,
                 log_level: str = 'INFO',
                 console_output: bool = True):
        """
        Initialize logger

        Args:
            name: Logger name
            log_file: Path to log file (optional)
            log_level: Logging level
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # JSON formatter
        json_formatter = JSONFormatter()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(json_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)

        # Specialized loggers
        self.query_logger = QueryLogger(self.logger)
        self.system_logger = SystemLogger(self.logger)
        self.performance_logger = PerformanceLogger(self.logger)

        # Log initialization
        self.logger.info(
            "Logger initialized",
            extra={'extra_fields': {
                'event_type': 'logger_init',
                'log_file': log_file,
                'log_level': log_level
            }}
        )

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra={'extra_fields': kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra={'extra_fields': kwargs})

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra={'extra_fields': kwargs})

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra={'extra_fields': kwargs})

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra={'extra_fields': kwargs})

    def log_query(self, **kwargs):
        """Delegate to query logger"""
        self.query_logger.log_query(**kwargs)

    def log_component_init(self, **kwargs):
        """Delegate to system logger"""
        self.system_logger.log_component_init(**kwargs)

    def log_performance(self, operation: str, latency_ms: float, **kwargs):
        """Delegate to performance logger"""
        self.performance_logger.log_latency(operation, latency_ms, kwargs)

    def get_performance_stats(self, operation: str = None) -> Dict:
        """Get performance statistics"""
        return self.performance_logger.get_percentiles(operation)


class LogContext:
    """
    Context manager for timing operations and automatic logging

    Usage:
        with LogContext(logger, 'faiss_search') as ctx:
            results = search_faiss(query)
            ctx.add_metadata({'num_results': len(results)})
    """

    def __init__(self,
                 logger: HeritageLogger,
                 operation: str,
                 log_level: str = 'DEBUG'):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None
        self.metadata = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            # Success
            self.logger.log_performance(
                self.operation,
                elapsed_ms,
                **self.metadata
            )
        else:
            # Error
            self.logger.error(
                f"Operation failed: {self.operation}",
                operation=self.operation,
                elapsed_ms=elapsed_ms,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.metadata
            )

    def add_metadata(self, metadata: Dict):
        """Add metadata to log entry"""
        self.metadata.update(metadata)


# Utility functions

def generate_query_id() -> str:
    """Generate unique query ID"""
    return f"query_{uuid.uuid4().hex[:12]}"


def format_query_log(query_id: str,
                    query_text: str,
                    num_results: int,
                    latency_ms: float,
                    **kwargs) -> Dict:
    """Format query log entry"""
    return {
        'query_id': query_id,
        'query_text': query_text,
        'num_results': num_results,
        'latency_ms': latency_ms,
        **kwargs
    }


if __name__ == '__main__':
    # Demo logging system
    print("="*60)
    print("STRUCTURED LOGGING SYSTEM DEMO")
    print("="*60)

    # Initialize logger
    logger = HeritageLogger(
        name='heritage_demo',
        log_file='logs/heritage_recommender.log',
        log_level='INFO',
        console_output=True
    )

    print("\n1. Query Logging:")
    logger.log_query(
        query_id='test-query-1',
        query_text='Mughal architecture monuments',
        parsed_query={'entities': ['Mughal', 'architecture'], 'domain': 'islamic'},
        results=[
            {'doc_id': 'doc1', 'score': 0.95},
            {'doc_id': 'doc2', 'score': 0.88}
        ],
        latency_ms=45.2,
        components_used=['faiss', 'simrank', 'horn'],
        fallbacks_triggered=[],
        user_id='user-123'
    )

    print("\n2. Component Initialization:")
    logger.log_component_init(
        component='knowledge_graph',
        success=True,
        load_time_ms=234.5
    )

    print("\n3. Performance Logging:")
    with LogContext(logger, 'simrank_lookup') as ctx:
        time.sleep(0.05)  # Simulate work
        ctx.add_metadata({'num_nodes': 100, 'cache_hit': True})

    print("\n4. Performance Statistics:")
    stats = logger.get_performance_stats('simrank_lookup')
    print(f"   P95 latency: {stats.get('p95', 0):.2f} ms")

    print("\n5. Error Logging:")
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error(
            "Test error occurred",
            component='test',
            error_type=type(e).__name__
        )

    print("\n✓ Logging system initialized")
    print(f"   Log file: logs/heritage_recommender.log")


_global_logger = None

def get_logger(name: str = None, log_file: str = None, log_level: str = 'INFO') -> HeritageLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name (defaults to 'heritage_recommender')
        log_file: Path to log file (optional)
        log_level: Logging level (default: 'INFO')
    
    Returns:
        HeritageLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = HeritageLogger(
            name=name or 'heritage_recommender',
            log_file=log_file,
            log_level=log_level,
            console_output=True
        )
    
    return _global_logger