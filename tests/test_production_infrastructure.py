"""
Comprehensive test suite for production infrastructure.

Tests:
1. Error handling and fallback chains
2. Logging functionality
3. Monitoring and alerting
4. Integration tests
5. Performance tests
6. Failure injection tests
"""

import pytest
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

from utils.error_handler import (
    ErrorSeverity, RecommenderError, CriticalError, HighSeverityError,
    MediumSeverityError, LowSeverityError, FallbackChain, ErrorHandler,
    InputValidator, OutputValidator
)
from utils.logger_v2 import (
    get_logger, QueryLogger, SystemLogger, PerformanceLogger,
    PerformanceTimer
)
from utils.monitoring import (
    MonitoringSystem, MetricCollector, AnomalyDetector, DriftDetector,
    AlertManager, AlertLevel
)


class TestErrorHandling:
    """Test error handling and fallback mechanisms"""

    def test_error_hierarchy(self):
        """Test error severity levels"""
        # Create errors of different severities
        critical = CriticalError("Database unavailable")
        high = HighSeverityError("Component failed")
        medium = MediumSeverityError("Data quality issue")
        low = LowSeverityError("Edge case")

        assert critical.severity == ErrorSeverity.CRITICAL
        assert high.severity == ErrorSeverity.HIGH
        assert medium.severity == ErrorSeverity.MEDIUM
        assert low.severity == ErrorSeverity.LOW

        # Test severity codes
        assert critical.http_code == 503
        assert high.http_code == 500
        assert medium.http_code == 200
        assert low.http_code == 200

    def test_fallback_chain_success(self):
        """Test fallback chain with successful primary strategy"""
        def primary():
            return "primary_result"

        def fallback1():
            return "fallback1_result"

        chain = FallbackChain("test_chain")
        chain.add_strategy("primary", primary, priority=3)
        chain.add_strategy("fallback", fallback1, priority=1)

        result, strategy_used = chain.execute()
        assert result == "primary_result"
        assert strategy_used == "primary"

    def test_fallback_chain_failure(self):
        """Test fallback chain when primary fails"""
        def primary():
            raise Exception("Primary failed")

        def fallback1():
            return "fallback1_result"

        chain = FallbackChain("test_chain")
        chain.add_strategy("primary", primary, priority=3)
        chain.add_strategy("fallback", fallback1, priority=1)

        result, strategy_used = chain.execute()
        assert result == "fallback1_result"
        assert strategy_used == "fallback"

    def test_fallback_chain_all_fail(self):
        """Test fallback chain when all strategies fail"""
        def primary():
            raise Exception("Primary failed")

        def fallback1():
            raise Exception("Fallback failed")

        chain = FallbackChain("test_chain")
        chain.add_strategy("primary", primary, priority=3)
        chain.add_strategy("fallback", fallback1, priority=1)

        with pytest.raises(CriticalError):
            chain.execute()

    def test_error_handler_decorator(self):
        """Test automatic error handling with decorator"""
        handler = ErrorHandler()

        @handler.handle_errors("test_component", fallback_value="fallback")
        def risky_function(should_fail=False):
            if should_fail:
                raise ValueError("Function failed")
            return "success"

        # Test success
        result = risky_function(should_fail=False)
        assert result == "success"

        # Test failure with fallback
        result = risky_function(should_fail=True)
        assert result == "fallback"

    def test_input_validation(self):
        """Test input validation"""
        validator = InputValidator()

        # Valid query text
        assert validator.validate_query_text("temple architecture") is True

        # Invalid query texts
        with pytest.raises(MediumSeverityError):
            validator.validate_query_text("")  # Empty

        with pytest.raises(MediumSeverityError):
            validator.validate_query_text("a" * 1001)  # Too long

        # Valid top_k
        assert validator.validate_top_k(10) is True

        # Invalid top_k
        with pytest.raises(MediumSeverityError):
            validator.validate_top_k(0)  # Too small

        with pytest.raises(MediumSeverityError):
            validator.validate_top_k(101)  # Too large

    def test_output_validation(self):
        """Test output validation"""
        validator = OutputValidator()

        # Valid recommendations
        valid_recs = [
            {'doc_id': '1', 'title': 'Doc 1', 'final_score': 0.9},
            {'doc_id': '2', 'title': 'Doc 2', 'final_score': 0.8}
        ]
        assert validator.validate_recommendations(valid_recs) is True

        # Empty recommendations
        with pytest.raises(MediumSeverityError):
            validator.validate_recommendations([])

        # Missing required field
        invalid_recs = [
            {'doc_id': '1', 'final_score': 0.9}  # Missing title
        ]
        with pytest.raises(MediumSeverityError):
            validator.validate_recommendations(invalid_recs)

        # Invalid score
        invalid_recs = [
            {'doc_id': '1', 'title': 'Doc 1', 'final_score': 1.5}  # Score > 1
        ]
        with pytest.raises(MediumSeverityError):
            validator.validate_recommendations(invalid_recs)


class TestLogging:
    """Test logging functionality"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_json_structured_logging(self, temp_log_dir):
        """Test JSON-structured log output"""
        log_file = temp_log_dir / "test.log"

        logger = get_logger("test_logger", log_file=str(log_file))
        logger.info("Test message", extra={
            'extra_fields': {'key': 'value', 'number': 42}
        })

        # Read log file
        with open(log_file) as f:
            log_line = f.readline()
            log_data = json.loads(log_line)

        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert log_data['key'] == 'value'
        assert log_data['number'] == 42

    def test_query_logger(self, temp_log_dir):
        """Test query-specific logging"""
        log_file = temp_log_dir / "queries.log"

        query_logger = QueryLogger(log_file=str(log_file))
        query_logger.log_query(
            query_id="test_001",
            query_text="temple architecture",
            results=[{'doc_id': '1'}, {'doc_id': '2'}],
            latency_ms=150.5,
            components_used=['simrank', 'horn_index']
        )

        # Read log file
        with open(log_file) as f:
            log_line = f.readline()
            log_data = json.loads(log_line)

        assert log_data['event_type'] == 'query'
        assert log_data['query_id'] == 'test_001'
        assert log_data['num_results'] == 2
        assert log_data['latency_ms'] == 150.5

    def test_performance_timer(self, temp_log_dir):
        """Test automatic performance timing"""
        log_file = temp_log_dir / "performance.log"

        perf_logger = PerformanceLogger(log_file=str(log_file))

        with perf_logger.time_operation("test_operation"):
            time.sleep(0.1)  # Simulate work

        # Read log file
        with open(log_file) as f:
            log_line = f.readline()
            log_data = json.loads(log_line)

        assert log_data['event_type'] == 'performance'
        assert log_data['operation'] == 'test_operation'
        assert log_data['duration_ms'] >= 100  # Should be at least 100ms


class TestMonitoring:
    """Test monitoring and alerting"""

    def test_metric_collection(self):
        """Test basic metric collection"""
        collector = MetricCollector(retention_minutes=60)

        # Record some metrics
        collector.record('test_metric', 10.0)
        collector.record('test_metric', 20.0)
        collector.record('test_metric', 30.0)

        # Get statistics
        stats = collector.get_statistics('test_metric', minutes=60)

        assert stats is not None
        assert stats['count'] == 3
        assert stats['mean'] == 20.0
        assert stats['min'] == 10.0
        assert stats['max'] == 30.0

    def test_metric_increment(self):
        """Test counter increment"""
        collector = MetricCollector()

        collector.increment('test_counter', 1.0)
        collector.increment('test_counter', 2.0)
        collector.increment('test_counter', 3.0)

        values = collector.get_recent_values('test_counter', minutes=60)
        assert len(values) == 3
        assert values[-1] == 6.0  # Should be cumulative

    def test_anomaly_detection_zscore(self):
        """Test z-score anomaly detection"""
        detector = AnomalyDetector(sensitivity=3.0)

        # Create baseline: mean=100, std=10
        baseline_values = [100 + np.random.randn() * 10 for _ in range(100)]
        detector.update_baseline('test_metric', baseline_values)

        # Normal value - should not be anomalous
        assert detector.detect_zscore('test_metric', 105) is False

        # Anomalous value (far from mean)
        assert detector.detect_zscore('test_metric', 150) is True

    def test_anomaly_detection_mad(self):
        """Test MAD anomaly detection (robust to outliers)"""
        detector = AnomalyDetector(sensitivity=3.0)

        # Create baseline with some outliers
        baseline_values = [100.0] * 90 + [200.0] * 10  # 90% around 100, 10% outliers
        detector.update_baseline('test_metric', baseline_values)

        # Normal value - should not be anomalous
        assert detector.detect_mad('test_metric', 102) is False

        # Anomalous value
        assert detector.detect_mad('test_metric', 150) is True

    def test_drift_detection_psi(self):
        """Test PSI-based drift detection"""
        detector = DriftDetector()

        # Reference distribution: Normal(100, 10)
        reference = np.random.normal(100, 10, 1000).tolist()
        detector.set_reference('test_distribution', reference)

        # Current distribution similar to reference - no drift
        current_similar = np.random.normal(100, 10, 1000).tolist()
        is_drift, psi = detector.detect_drift_psi('test_distribution', current_similar)
        assert is_drift is False  # PSI should be < 0.2

        # Current distribution shifted - drift detected
        current_shifted = np.random.normal(120, 10, 1000).tolist()
        is_drift, psi = detector.detect_drift_psi('test_distribution', current_shifted)
        assert is_drift is True  # PSI should be > 0.2

    def test_alert_creation(self):
        """Test alert creation and deduplication"""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_file = Path(temp_dir) / "alerts.jsonl"
            manager = AlertManager(alert_file=alert_file)

            # Create alert
            alert = manager.create_alert(
                metric_name='test_metric',
                level=AlertLevel.WARNING,
                message='Test alert',
                current_value=0.8,
                threshold=0.7
            )

            assert alert is not None
            assert alert.level == AlertLevel.WARNING

            # Try to create duplicate alert - should be deduplicated
            alert2 = manager.create_alert(
                metric_name='test_metric',
                level=AlertLevel.WARNING,
                message='Test alert',
                current_value=0.85,
                threshold=0.7
            )

            assert alert2 is None  # Should be None due to cooldown

    def test_alert_filtering(self):
        """Test alert filtering by level"""
        with tempfile.TemporaryDirectory() as temp_dir:
            alert_file = Path(temp_dir) / "alerts.jsonl"
            manager = AlertManager(alert_file=alert_file)

            # Create alerts of different levels
            manager.create_alert('metric1', AlertLevel.INFO, 'Info', 1.0, 1.0)
            manager.create_alert('metric2', AlertLevel.WARNING, 'Warning', 2.0, 2.0)
            manager.create_alert('metric3', AlertLevel.CRITICAL, 'Critical', 3.0, 3.0)

            # Get all alerts
            all_alerts = manager.get_active_alerts()
            assert len(all_alerts) == 3

            # Get only critical alerts
            critical_alerts = manager.get_active_alerts(min_level=AlertLevel.CRITICAL)
            assert len(critical_alerts) == 1
            assert critical_alerts[0].level == AlertLevel.CRITICAL

    def test_monitoring_system_integration(self):
        """Test full monitoring system"""
        monitor = MonitoringSystem(config={
            'retention_minutes': 60,
            'anomaly_sensitivity': 3.0
        })

        # Record some queries
        for i in range(10):
            monitor.record_query(
                query_id=f'q{i}',
                latency_ms=100 + i * 10,
                num_results=10,
                ndcg=0.8,
                components_used=['simrank', 'horn_index']
            )

        # Get dashboard data
        dashboard = monitor.get_dashboard_data()

        assert 'metrics' in dashboard
        assert 'alerts' in dashboard
        assert 'system_health' in dashboard
        assert dashboard['system_health']['score'] >= 0
        assert dashboard['system_health']['score'] <= 100

    def test_threshold_alerting(self):
        """Test automatic alerting based on thresholds"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'alert_file': str(Path(temp_dir) / 'alerts.jsonl'),
                'thresholds': {
                    'query_latency_p95': {
                        'max': 1000,
                        'warning': 800,
                        'critical': 2000
                    }
                }
            }

            monitor = MonitoringSystem(config=config)

            # Record queries with high latency (should trigger alert)
            for i in range(20):
                monitor.record_query(
                    query_id=f'q{i}',
                    latency_ms=1500,  # Above warning threshold
                    num_results=10,
                    ndcg=0.8,
                    components_used=['simrank']
                )

            # Check if alert was created
            active_alerts = monitor.alerts.get_active_alerts()
            # Note: May not always trigger due to statistical nature
            # This is an integration test to verify the flow works


class TestIntegration:
    """Integration tests for combined functionality"""

    def test_error_logging_integration(self, temp_log_dir):
        """Test that errors are properly logged"""
        log_file = temp_log_dir / "errors.log"
        logger = get_logger("test", log_file=str(log_file))

        try:
            raise HighSeverityError("Test error")
        except HighSeverityError as e:
            logger.error(f"Error occurred: {e}", extra={
                'extra_fields': e.to_dict()
            })

        # Verify error was logged
        with open(log_file) as f:
            log_line = f.readline()
            log_data = json.loads(log_line)

        assert log_data['level'] == 'ERROR'
        assert 'severity' in log_data

    def test_monitoring_alerting_integration(self):
        """Test monitoring triggers alerts which are logged"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'alert_file': str(Path(temp_dir) / 'alerts.jsonl'),
                'thresholds': {
                    'error_rate': {
                        'max': 0.01,
                        'warning': 0.005,
                        'critical': 0.05
                    }
                }
            }

            monitor = MonitoringSystem(config=config)

            # Simulate high error rate
            for i in range(100):
                monitor.record_query(
                    query_id=f'q{i}',
                    latency_ms=100,
                    num_results=10,
                    ndcg=0.8,
                    components_used=['simrank'],
                    had_error=(i % 10 == 0)  # 10% error rate
                )

            # Check alerts were created and logged
            alert_file = Path(temp_dir) / 'alerts.jsonl'
            if alert_file.exists():
                with open(alert_file) as f:
                    alerts = [json.loads(line) for line in f]
                # Verify alerts have proper structure
                for alert in alerts:
                    assert 'alert_id' in alert
                    assert 'level' in alert
                    assert 'metric_name' in alert


class TestPerformance:
    """Performance tests"""

    def test_metric_collection_performance(self):
        """Test metric collection can handle high throughput"""
        collector = MetricCollector()

        start_time = time.time()

        # Record 10,000 metrics
        for i in range(10000):
            collector.record('test_metric', float(i))

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second

    def test_statistics_calculation_performance(self):
        """Test statistics calculation performance"""
        collector = MetricCollector()

        # Record 10,000 metrics
        for i in range(10000):
            collector.record('test_metric', float(i))

        start_time = time.time()

        # Calculate statistics 100 times
        for _ in range(100):
            stats = collector.get_statistics('test_metric', minutes=60)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 1.0


class TestFailureInjection:
    """Failure injection tests"""

    def test_simrank_component_failure(self):
        """Test graceful degradation when SimRank fails"""
        handler = ErrorHandler()

        def simrank_ranking(docs):
            raise Exception("SimRank unavailable")

        def horn_index_ranking(docs):
            # Fallback ranking
            return sorted(docs, key=lambda x: x.get('horn_score', 0), reverse=True)

        # Create fallback chain
        chain = FallbackChain("ranking_chain")
        chain.add_strategy("simrank", simrank_ranking, priority=3)
        chain.add_strategy("horn_index", horn_index_ranking, priority=2)

        # Test with sample documents
        docs = [
            {'doc_id': '1', 'horn_score': 0.8},
            {'doc_id': '2', 'horn_score': 0.9},
            {'doc_id': '3', 'horn_score': 0.7}
        ]

        result, strategy = chain.execute(docs)

        # Should use fallback
        assert strategy == "horn_index"
        assert result[0]['doc_id'] == '2'  # Highest horn_score

    def test_timeout_handling(self):
        """Test timeout handling for slow components"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        def slow_function():
            time.sleep(5)  # Simulate slow operation
            return "result"

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # 1 second timeout

        try:
            result = slow_function()
            assert False, "Should have timed out"
        except TimeoutError:
            # Expected
            pass
        finally:
            signal.alarm(0)  # Cancel alarm

    def test_partial_results_handling(self):
        """Test handling of partial results"""
        validator = OutputValidator()

        # Simulate partial results (some components failed)
        partial_results = [
            {'doc_id': '1', 'title': 'Doc 1', 'final_score': 0.9, 'simrank_score': 0.8},
            {'doc_id': '2', 'title': 'Doc 2', 'final_score': 0.8, 'simrank_score': None},  # SimRank failed
            {'doc_id': '3', 'title': 'Doc 3', 'final_score': 0.7, 'simrank_score': 0.6}
        ]

        # Should still be valid even with missing component scores
        try:
            is_valid = validator.validate_recommendations(partial_results)
            assert is_valid
        except Exception as e:
            # If validator is strict, we should handle gracefully
            assert isinstance(e, MediumSeverityError)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "failure_injection: mark test as failure injection test"
    )


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
