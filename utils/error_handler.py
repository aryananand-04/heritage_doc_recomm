"""
Comprehensive Error Handling System

Categorizes errors by severity and implements appropriate responses:
- CRITICAL: System cannot continue (503 Service Unavailable)
- HIGH: Component failure with fallback
- MEDIUM: Data quality issues with reduced functionality
- LOW: Expected edge cases with graceful handling
"""

import sys
import traceback
from typing import Optional, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass
import time


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 4  # System cannot continue
    HIGH = 3      # Component failure, use fallback
    MEDIUM = 2    # Data quality issue, reduced functionality
    LOW = 1       # Expected edge case, graceful message


class ErrorCategory(Enum):
    """Error categories for classification"""
    # Critical errors
    MISSING_KNOWLEDGE_GRAPH = ("Knowledge graph file not found", ErrorSeverity.CRITICAL)
    CORRUPTED_SIMRANK = ("SimRank matrix corrupted or invalid", ErrorSeverity.CRITICAL)
    MISSING_FAISS_INDEX = ("FAISS index file missing", ErrorSeverity.CRITICAL)
    DATABASE_CONNECTION = ("Cannot connect to database", ErrorSeverity.CRITICAL)

    # High severity errors
    MISSING_HORN_WEIGHTS = ("Horn's Index weights missing", ErrorSeverity.HIGH)
    EMBEDDING_NOT_FOUND = ("Embedding missing for document", ErrorSeverity.HIGH)
    ENTITY_EXTRACTION_FAILED = ("Entity extraction service failed", ErrorSeverity.HIGH)
    MODEL_LOAD_FAILED = ("Model loading failed", ErrorSeverity.HIGH)

    # Medium severity errors
    NO_ENTITIES_DETECTED = ("Query has no detected entities", ErrorSeverity.MEDIUM)
    INCOMPLETE_METADATA = ("Document has incomplete metadata", ErrorSeverity.MEDIUM)
    INVALID_QUERY_FORMAT = ("Query format is invalid", ErrorSeverity.MEDIUM)
    CLUSTER_NOT_FOUND = ("Document cluster not found", ErrorSeverity.MEDIUM)

    # Low severity errors
    EMPTY_QUERY = ("Empty query provided", ErrorSeverity.LOW)
    UNSUPPORTED_LANGUAGE = ("Query in unsupported language", ErrorSeverity.LOW)
    NO_RESULTS_FOUND = ("No results found for query", ErrorSeverity.LOW)
    INVALID_PARAMETER = ("Invalid parameter value", ErrorSeverity.LOW)

    def __init__(self, message: str, severity: ErrorSeverity):
        self.message = message
        self.severity = severity


@dataclass
class ErrorContext:
    """Context information for error handling"""
    query_id: str
    query_text: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    timestamp: float = None
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class ErrorResponse:
    """Standardized error response"""
    success: bool
    error_category: Optional[ErrorCategory]
    error_message: str
    severity: ErrorSeverity
    http_status_code: int
    fallback_used: Optional[str] = None
    recovery_action: Optional[str] = None
    user_message: Optional[str] = None
    context: Optional[ErrorContext] = None


class HeritageRecommenderError(Exception):
    """Base exception for heritage recommender system"""
    def __init__(self,
                 category: ErrorCategory,
                 context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None):
        self.category = category
        self.context = context
        self.original_exception = original_exception
        super().__init__(category.message)


class CriticalError(HeritageRecommenderError):
    """Critical errors that prevent system operation"""
    pass


class ComponentError(HeritageRecommenderError):
    """High severity component failures"""
    pass


class DataQualityError(HeritageRecommenderError):
    """Medium severity data quality issues"""
    pass


class UserInputError(HeritageRecommenderError):
    """Low severity user input issues"""
    pass


class ErrorHandler:
    """Central error handling and recovery system"""

    def __init__(self, logger=None, alert_callback: Optional[Callable] = None):
        """
        Initialize error handler

        Args:
            logger: Logger instance for error logging
            alert_callback: Function to call for critical alerts
        """
        self.logger = logger
        self.alert_callback = alert_callback

        # Error statistics
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
        self.error_history = []
        self.max_history = 1000

    def handle_error(self,
                    error: Exception,
                    context: Optional[ErrorContext] = None,
                    fallback_result: Any = None) -> ErrorResponse:
        """
        Handle error with appropriate response based on severity

        Args:
            error: Exception that occurred
            context: Context information
            fallback_result: Optional fallback result if available

        Returns:
            ErrorResponse with appropriate handling
        """
        # Classify error
        if isinstance(error, HeritageRecommenderError):
            category = error.category
            severity = category.severity
        else:
            # Unknown error - treat as CRITICAL
            category = None
            severity = ErrorSeverity.CRITICAL

        # Update statistics
        self.error_counts[severity] += 1
        self.error_history.append({
            'timestamp': time.time(),
            'severity': severity.name,
            'category': category.name if category else 'UNKNOWN',
            'message': str(error)
        })
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            return self._handle_critical_error(error, category, context)
        elif severity == ErrorSeverity.HIGH:
            return self._handle_high_severity_error(error, category, context, fallback_result)
        elif severity == ErrorSeverity.MEDIUM:
            return self._handle_medium_severity_error(error, category, context)
        else:  # LOW
            return self._handle_low_severity_error(error, category, context)

    def _handle_critical_error(self,
                               error: Exception,
                               category: Optional[ErrorCategory],
                               context: Optional[ErrorContext]) -> ErrorResponse:
        """Handle critical errors"""
        error_msg = f"CRITICAL ERROR: {str(error)}"

        # Log with full stack trace
        if self.logger:
            self.logger.critical(
                error_msg,
                extra={
                    'category': category.name if category else 'UNKNOWN',
                    'context': context.__dict__ if context else {},
                    'stack_trace': traceback.format_exc()
                }
            )

        # Send alert
        if self.alert_callback:
            self.alert_callback(
                severity='CRITICAL',
                message=error_msg,
                context=context
            )

        # Return 503 Service Unavailable
        return ErrorResponse(
            success=False,
            error_category=category,
            error_message=error_msg,
            severity=ErrorSeverity.CRITICAL,
            http_status_code=503,
            user_message="Service temporarily unavailable. Our team has been notified.",
            recovery_action="System restart required",
            context=context
        )

    def _handle_high_severity_error(self,
                                    error: Exception,
                                    category: Optional[ErrorCategory],
                                    context: Optional[ErrorContext],
                                    fallback_result: Any) -> ErrorResponse:
        """Handle high severity errors with fallback"""
        error_msg = f"HIGH SEVERITY: {str(error)}"

        # Determine fallback strategy
        fallback_strategy = self._get_fallback_strategy(category)

        # Log warning
        if self.logger:
            self.logger.warning(
                error_msg,
                extra={
                    'category': category.name if category else 'UNKNOWN',
                    'fallback': fallback_strategy,
                    'context': context.__dict__ if context else {}
                }
            )

        # Return 200 OK with fallback result
        return ErrorResponse(
            success=True,  # Partial success with fallback
            error_category=category,
            error_message=error_msg,
            severity=ErrorSeverity.HIGH,
            http_status_code=200,
            fallback_used=fallback_strategy,
            recovery_action=f"Using fallback: {fallback_strategy}",
            user_message=None,  # Don't expose to user
            context=context
        )

    def _handle_medium_severity_error(self,
                                      error: Exception,
                                      category: Optional[ErrorCategory],
                                      context: Optional[ErrorContext]) -> ErrorResponse:
        """Handle medium severity errors with reduced functionality"""
        error_msg = f"MEDIUM SEVERITY: {str(error)}"

        # Log info
        if self.logger:
            self.logger.info(
                error_msg,
                extra={
                    'category': category.name if category else 'UNKNOWN',
                    'context': context.__dict__ if context else {}
                }
            )

        # Continue with reduced functionality
        return ErrorResponse(
            success=True,
            error_category=category,
            error_message=error_msg,
            severity=ErrorSeverity.MEDIUM,
            http_status_code=200,
            recovery_action="Proceeding with reduced functionality",
            user_message="Some features may be limited for this query.",
            context=context
        )

    def _handle_low_severity_error(self,
                                   error: Exception,
                                   category: Optional[ErrorCategory],
                                   context: Optional[ErrorContext]) -> ErrorResponse:
        """Handle low severity errors with graceful message"""
        error_msg = str(error)

        # Log for analysis
        if self.logger:
            self.logger.debug(
                error_msg,
                extra={
                    'category': category.name if category else 'UNKNOWN',
                    'context': context.__dict__ if context else {}
                }
            )

        # User-friendly messages
        user_messages = {
            ErrorCategory.EMPTY_QUERY: "Please enter a search query.",
            ErrorCategory.UNSUPPORTED_LANGUAGE: "Only English queries are currently supported.",
            ErrorCategory.NO_RESULTS_FOUND: "No results found. Try different keywords.",
            ErrorCategory.INVALID_PARAMETER: "Invalid search parameter."
        }

        return ErrorResponse(
            success=False,
            error_category=category,
            error_message=error_msg,
            severity=ErrorSeverity.LOW,
            http_status_code=400,
            user_message=user_messages.get(category, "Invalid request."),
            context=context
        )

    def _get_fallback_strategy(self, category: Optional[ErrorCategory]) -> str:
        """Determine fallback strategy based on error category"""
        fallback_map = {
            ErrorCategory.MISSING_HORN_WEIGHTS: "uniform_weights",
            ErrorCategory.EMBEDDING_NOT_FOUND: "cluster_average_embedding",
            ErrorCategory.ENTITY_EXTRACTION_FAILED: "text_only_search",
            ErrorCategory.MODEL_LOAD_FAILED: "rule_based_retrieval"
        }
        return fallback_map.get(category, "default_fallback")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'by_severity': {
                severity.name: count
                for severity, count in self.error_counts.items()
            },
            'recent_errors': self.error_history[-10:],  # Last 10 errors
            'error_rate': self._calculate_error_rate()
        }

    def _calculate_error_rate(self) -> float:
        """Calculate error rate over last hour"""
        one_hour_ago = time.time() - 3600
        recent_errors = [e for e in self.error_history if e['timestamp'] > one_hour_ago]
        # Assuming we track total requests separately
        return len(recent_errors)  # Return count for now

    def reset_statistics(self):
        """Reset error statistics (for testing or periodic reset)"""
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
        self.error_history = []


def with_error_handling(error_handler: ErrorHandler,
                        fallback_result: Any = None,
                        component: str = None):
    """
    Decorator for automatic error handling

    Usage:
        @with_error_handling(error_handler, fallback_result=[], component="faiss")
        def search_faiss(query):
            # Implementation
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                query_id=kwargs.get('query_id', 'unknown'),
                query_text=kwargs.get('query_text'),
                component=component
            )

            try:
                return func(*args, **kwargs)
            except HeritageRecommenderError as e:
                response = error_handler.handle_error(e, context, fallback_result)
                if response.success:
                    return fallback_result
                else:
                    raise
            except Exception as e:
                # Wrap unknown exceptions
                critical_error = CriticalError(
                    ErrorCategory.DATABASE_CONNECTION,  # Generic critical
                    context=context,
                    original_exception=e
                )
                response = error_handler.handle_error(critical_error, context)
                raise

        return wrapper
    return decorator


class FallbackChain:
    """
    Implements fallback chain for graceful degradation

    Usage:
        chain = FallbackChain()
        chain.add_strategy("hybrid", hybrid_search, priority=1)
        chain.add_strategy("embedding_only", embedding_search, priority=2)
        chain.add_strategy("popular", get_popular_items, priority=3)
        result = chain.execute(query)
    """

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.strategies = []

    def add_strategy(self, name: str, func: Callable, priority: int):
        """Add a fallback strategy"""
        self.strategies.append({
            'name': name,
            'func': func,
            'priority': priority
        })
        # Sort by priority (lower = higher priority)
        self.strategies.sort(key=lambda x: x['priority'])

    def execute(self, *args, **kwargs) -> tuple:
        """
        Execute strategies in order until one succeeds

        Returns:
            (result, strategy_used)
        """
        last_error = None

        for strategy in self.strategies:
            try:
                result = strategy['func'](*args, **kwargs)

                # Log fallback usage if not primary
                if strategy['priority'] > 1:
                    if self.error_handler.logger:
                        self.error_handler.logger.info(
                            f"Fallback strategy used: {strategy['name']}",
                            extra={'query': kwargs.get('query_text', 'unknown')}
                        )

                return result, strategy['name']

            except Exception as e:
                last_error = e
                if self.error_handler.logger:
                    self.error_handler.logger.warning(
                        f"Strategy {strategy['name']} failed: {str(e)}"
                    )
                continue

        # All strategies failed
        raise CriticalError(
            ErrorCategory.MISSING_KNOWLEDGE_GRAPH,  # Generic critical
            context=ErrorContext(
                query_id=kwargs.get('query_id', 'unknown'),
                additional_info={'all_strategies_failed': True}
            ),
            original_exception=last_error
        )


# Validation utilities

def validate_query(query: str, max_length: int = 500) -> None:
    """
    Validate query input

    Raises:
        UserInputError if validation fails
    """
    if not query or not query.strip():
        raise UserInputError(ErrorCategory.EMPTY_QUERY)

    if len(query) > max_length:
        raise UserInputError(
            ErrorCategory.INVALID_PARAMETER,
            context=ErrorContext(
                query_id='validation',
                additional_info={'max_length': max_length, 'actual': len(query)}
            )
        )

    # Check for only special characters
    if not any(c.isalnum() for c in query):
        raise UserInputError(ErrorCategory.INVALID_QUERY_FORMAT)


def validate_recommendations(recommendations: list,
                            knowledge_graph,
                            check_scores: bool = True) -> list:
    """
    Validate recommendation output

    Args:
        recommendations: List of (doc_id, score) tuples
        knowledge_graph: KG for doc_id validation
        check_scores: Whether to validate score ranges

    Returns:
        Cleaned recommendations list

    Raises:
        DataQualityError if critical validation fails
    """
    if not recommendations:
        return []

    cleaned = []
    for doc_id, score in recommendations:
        # Check doc_id exists
        if knowledge_graph and doc_id not in knowledge_graph:
            continue  # Skip invalid doc_ids

        # Check score range
        if check_scores and not (0 <= score <= 1):
            # Normalize score
            score = max(0, min(1, score))

        cleaned.append((doc_id, score))

    # Remove duplicates (keep highest score)
    seen = {}
    for doc_id, score in cleaned:
        if doc_id not in seen or score > seen[doc_id]:
            seen[doc_id] = score

    return [(doc_id, score) for doc_id, score in seen.items()]


if __name__ == '__main__':
    # Demo error handling
    print("="*60)
    print("ERROR HANDLING SYSTEM DEMO")
    print("="*60)

    # Create error handler
    handler = ErrorHandler()

    # Test different severity levels
    print("\n1. CRITICAL ERROR:")
    try:
        raise CriticalError(
            ErrorCategory.MISSING_KNOWLEDGE_GRAPH,
            context=ErrorContext(query_id='test-1', query_text='test query')
        )
    except Exception as e:
        response = handler.handle_error(e)
        print(f"   HTTP Status: {response.http_status_code}")
        print(f"   User Message: {response.user_message}")

    print("\n2. HIGH SEVERITY ERROR (with fallback):")
    try:
        raise ComponentError(
            ErrorCategory.MISSING_HORN_WEIGHTS,
            context=ErrorContext(query_id='test-2')
        )
    except Exception as e:
        response = handler.handle_error(e, fallback_result={'weights': 'uniform'})
        print(f"   Success: {response.success}")
        print(f"   Fallback: {response.fallback_used}")

    print("\n3. MEDIUM SEVERITY ERROR:")
    try:
        raise DataQualityError(
            ErrorCategory.NO_ENTITIES_DETECTED,
            context=ErrorContext(query_id='test-3')
        )
    except Exception as e:
        response = handler.handle_error(e)
        print(f"   Recovery: {response.recovery_action}")

    print("\n4. LOW SEVERITY ERROR:")
    try:
        raise UserInputError(
            ErrorCategory.EMPTY_QUERY,
            context=ErrorContext(query_id='test-4')
        )
    except Exception as e:
        response = handler.handle_error(e)
        print(f"   User Message: {response.user_message}")

    # Statistics
    print("\n" + "="*60)
    print("ERROR STATISTICS:")
    print("="*60)
    stats = handler.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print(f"By severity: {stats['by_severity']}")

    print("\n✓ Error handling system initialized")

RecommenderError = HeritageRecommenderError
ComponentFailureError = ComponentError
DataError = DataQualityError
InputError = UserInputError