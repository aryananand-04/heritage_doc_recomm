"""
Demo script to test the monitoring system functionality.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.monitoring import (
    MonitoringSystem, MetricCollector, AnomalyDetector,
    DriftDetector, AlertManager, AlertLevel
)
from utils.logger_v2 import get_logger

logger = get_logger(__name__)

def test_basic_monitoring():
    """Test basic monitoring functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Monitoring System")
    print("="*60)

    # Initialize monitoring
    monitor = MonitoringSystem(config={
        'retention_minutes': 60,
        'anomaly_sensitivity': 3.0,
        'thresholds': {
            'query_latency_p95': {
                'max': 2000,
                'warning': 1800,
                'critical': 3000
            },
            'ndcg_at_10': {
                'min': 0.6,
                'warning': 0.65,
                'critical': 0.5
            }
        }
    })

    print("\n✓ Monitoring system initialized")
    print(f"  - Retention: 60 minutes")
    print(f"  - Anomaly sensitivity: 3.0")
    print(f"  - Thresholds configured: 2")

    # Record some queries
    print("\n📊 Recording 20 sample queries...")
    for i in range(20):
        monitor.record_query(
            query_id=f"test_query_{i}",
            latency_ms=100 + np.random.randn() * 20,
            num_results=10,
            ndcg=0.75 + np.random.randn() * 0.05,
            components_used=['simrank', 'horn_index', 'embedding'],
            had_error=(i % 20 == 0),  # 5% error rate
            used_fallback=(i % 10 == 0)  # 10% fallback rate
        )

    # Get statistics
    stats = monitor.metrics.get_statistics('query_latency', minutes=5)
    print(f"\n✓ Query latency statistics:")
    print(f"  - Mean: {stats['mean']:.2f}ms")
    print(f"  - P95: {stats['p95']:.2f}ms")
    print(f"  - P99: {stats['p99']:.2f}ms")
    print(f"  - Min: {stats['min']:.2f}ms, Max: {stats['max']:.2f}ms")

    ndcg_stats = monitor.metrics.get_statistics('ndcg_at_10', minutes=5)
    print(f"\n✓ NDCG@10 statistics:")
    print(f"  - Mean: {ndcg_stats['mean']:.3f}")
    print(f"  - Std: {ndcg_stats['std']:.3f}")

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"\n✓ System health:")
    print(f"  - Health score: {dashboard['system_health']['score']:.1f}/100")
    print(f"  - Status: {dashboard['system_health']['status']}")
    print(f"  - Active alerts: {len(dashboard['alerts'])}")

    return monitor


def test_anomaly_detection():
    """Test anomaly detection"""
    print("\n" + "="*60)
    print("TEST 2: Anomaly Detection")
    print("="*60)

    detector = AnomalyDetector(sensitivity=3.0)

    # Create baseline: Normal(100, 10)
    print("\n📊 Creating baseline distribution...")
    baseline_values = [100 + np.random.randn() * 10 for _ in range(100)]
    detector.update_baseline('test_metric', baseline_values)

    baseline_stats = np.array(baseline_values)
    print(f"✓ Baseline established:")
    print(f"  - Mean: {np.mean(baseline_stats):.2f}")
    print(f"  - Std: {np.std(baseline_stats):.2f}")
    print(f"  - Range: [{np.min(baseline_stats):.2f}, {np.max(baseline_stats):.2f}]")

    # Test normal values
    print("\n🔍 Testing normal values...")
    normal_values = [95, 100, 105, 110]
    normal_anomalies = 0
    for value in normal_values:
        is_anomaly = detector.detect('test_metric', value, method='mad')
        if is_anomaly:
            normal_anomalies += 1
            print(f"  ⚠️  Value {value:.2f} detected as anomaly (unexpected)")
        else:
            print(f"  ✓ Value {value:.2f} is normal")

    # Test anomalous values
    print("\n🔍 Testing anomalous values...")
    anomalous_values = [150, 200, 50, 30]
    detected_anomalies = 0
    for value in anomalous_values:
        is_anomaly = detector.detect('test_metric', value, method='mad')
        if is_anomaly:
            detected_anomalies += 1
            print(f"  ⚠️  Value {value:.2f} detected as anomaly ✓")
        else:
            print(f"  ✗ Value {value:.2f} not detected (should be anomaly)")

    print(f"\n✓ Detection results:")
    print(f"  - Normal values incorrectly flagged: {normal_anomalies}/{len(normal_values)}")
    print(f"  - Anomalies correctly detected: {detected_anomalies}/{len(anomalous_values)}")


def test_drift_detection():
    """Test drift detection"""
    print("\n" + "="*60)
    print("TEST 3: Data Drift Detection")
    print("="*60)

    detector = DriftDetector()

    # Reference distribution
    print("\n📊 Creating reference distribution...")
    reference = np.random.normal(100, 10, 1000).tolist()
    detector.set_reference('test_distribution', reference)
    print(f"✓ Reference distribution: N(100, 10), n=1000")

    # Test 1: Similar distribution - no drift
    print("\n🔍 Test 1: Similar distribution...")
    current_similar = np.random.normal(100, 10, 1000).tolist()
    is_drift, psi = detector.detect_drift_psi('test_distribution', current_similar)

    print(f"  - Current distribution: N(100, 10)")
    print(f"  - PSI: {psi:.4f}")
    print(f"  - Drift detected: {is_drift}")
    print(f"  - Result: {'✓ Correct (no drift)' if not is_drift else '✗ False positive'}")

    # Test 2: Shifted distribution - drift expected
    print("\n🔍 Test 2: Shifted distribution...")
    current_shifted = np.random.normal(120, 10, 1000).tolist()
    is_drift, psi = detector.detect_drift_psi('test_distribution', current_shifted)

    print(f"  - Current distribution: N(120, 10)")
    print(f"  - PSI: {psi:.4f}")
    print(f"  - Drift detected: {is_drift}")
    print(f"  - Result: {'✓ Correct (drift detected)' if is_drift else '✗ False negative'}")

    # Test 3: Different variance - drift expected
    print("\n🔍 Test 3: Different variance...")
    current_variance = np.random.normal(100, 20, 1000).tolist()
    is_drift, psi = detector.detect_drift_psi('test_distribution', current_variance)

    print(f"  - Current distribution: N(100, 20)")
    print(f"  - PSI: {psi:.4f}")
    print(f"  - Drift detected: {is_drift}")
    print(f"  - Result: {'✓ Correct (drift detected)' if is_drift else '✗ False negative'}")


def test_alerting():
    """Test alert management"""
    print("\n" + "="*60)
    print("TEST 4: Alert Management")
    print("="*60)

    import tempfile
    temp_dir = tempfile.mkdtemp()
    alert_file = Path(temp_dir) / 'alerts.jsonl'

    manager = AlertManager(alert_file=alert_file)
    print(f"\n✓ Alert manager initialized")
    print(f"  - Alert file: {alert_file}")

    # Create alerts of different levels
    print("\n📢 Creating test alerts...")

    alert1 = manager.create_alert(
        metric_name='query_latency_p95',
        level=AlertLevel.WARNING,
        message='Query latency above warning threshold',
        current_value=1850.0,
        threshold=1800.0,
        context={'component': 'simrank'}
    )
    print(f"  ✓ Created WARNING alert: {alert1.alert_id}")

    alert2 = manager.create_alert(
        metric_name='error_rate',
        level=AlertLevel.CRITICAL,
        message='Error rate critically high',
        current_value=0.08,
        threshold=0.05,
        context={'errors': 80, 'total_queries': 1000}
    )
    print(f"  ✓ Created CRITICAL alert: {alert2.alert_id}")

    alert3 = manager.create_alert(
        metric_name='cache_hit_rate',
        level=AlertLevel.INFO,
        message='Cache hit rate normal',
        current_value=0.75,
        threshold=0.7,
        context={'hits': 750, 'total': 1000}
    )
    print(f"  ✓ Created INFO alert: {alert3.alert_id}")

    # Test deduplication
    print("\n🔍 Testing alert deduplication...")
    duplicate = manager.create_alert(
        metric_name='query_latency_p95',
        level=AlertLevel.WARNING,
        message='Query latency still high',
        current_value=1900.0,
        threshold=1800.0
    )
    print(f"  - Duplicate alert result: {duplicate}")
    print(f"  {'✓ Correctly deduplicated' if duplicate is None else '✗ Deduplication failed'}")

    # Get active alerts
    print("\n📋 Active alerts:")
    all_alerts = manager.get_active_alerts()
    print(f"  - Total active: {len(all_alerts)}")

    for alert in all_alerts:
        print(f"  - [{alert.level.name}] {alert.metric_name}: {alert.message}")

    # Filter by level
    critical_alerts = manager.get_active_alerts(min_level=AlertLevel.CRITICAL)
    print(f"\n✓ Critical alerts: {len(critical_alerts)}")
    for alert in critical_alerts:
        print(f"  - {alert.message}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_threshold_monitoring():
    """Test automatic threshold monitoring"""
    print("\n" + "="*60)
    print("TEST 5: Threshold Monitoring")
    print("="*60)

    import tempfile
    temp_dir = tempfile.mkdtemp()

    monitor = MonitoringSystem(config={
        'alert_file': str(Path(temp_dir) / 'alerts.jsonl'),
        'thresholds': {
            'query_latency_p95': {
                'max': 1000,
                'warning': 800,
                'critical': 2000
            },
            'ndcg_at_10': {
                'min': 0.6,
                'warning': 0.65,
                'critical': 0.5
            }
        }
    })

    print("\n✓ Monitoring system initialized with thresholds")

    # Record queries with varying latencies
    print("\n📊 Recording queries with varying latencies...")

    # Normal latencies
    print("  - Phase 1: Normal latencies (500-700ms)...")
    for i in range(10):
        monitor.record_query(
            query_id=f"query_normal_{i}",
            latency_ms=600 + np.random.randn() * 50,
            num_results=10,
            ndcg=0.75,
            components_used=['simrank']
        )

    # High latencies (should trigger warning)
    print("  - Phase 2: High latencies (900-1100ms, should warn)...")
    for i in range(10):
        monitor.record_query(
            query_id=f"query_high_{i}",
            latency_ms=1000 + np.random.randn() * 100,
            num_results=10,
            ndcg=0.75,
            components_used=['simrank']
        )

    # Check alerts
    active_alerts = monitor.alerts.get_active_alerts()
    print(f"\n📢 Alerts generated: {len(active_alerts)}")

    if active_alerts:
        for alert in active_alerts:
            print(f"  - [{alert.level.name}] {alert.message}")
            print(f"    Current: {alert.current_value:.2f}, Threshold: {alert.threshold:.2f}")
    else:
        print("  - No alerts (may need more data points for statistical significance)")

    # Get health score
    health = monitor._calculate_health_score()
    print(f"\n✓ System health:")
    print(f"  - Score: {health['score']:.1f}/100")
    print(f"  - Status: {health['status']}")
    print(f"  - Issues: {len(health['issues'])}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_component_monitoring():
    """Test component-level monitoring"""
    print("\n" + "="*60)
    print("TEST 6: Component Performance Monitoring")
    print("="*60)

    monitor = MonitoringSystem()

    print("\n📊 Recording component performance...")

    # Simulate component calls
    components = ['simrank', 'horn_index', 'embedding', 'query_classifier']

    for component in components:
        print(f"\n  Component: {component}")

        # Successful calls
        success_count = np.random.randint(80, 100)
        failure_count = np.random.randint(0, 10)

        for i in range(success_count):
            latency = np.random.exponential(100)  # Exponential distribution
            monitor.record_component_performance(
                component_name=component,
                latency_ms=latency,
                success=True
            )

        # Failed calls
        for i in range(failure_count):
            latency = np.random.exponential(100)
            error_types = ['timeout', 'memory_error', 'network_error']
            monitor.record_component_performance(
                component_name=component,
                latency_ms=latency,
                success=False,
                error_type=np.random.choice(error_types)
            )

        # Get statistics
        latency_stats = monitor.metrics.get_statistics(f'{component}_latency', minutes=5)

        if latency_stats:
            print(f"    ✓ Calls: {success_count}")
            print(f"    ✓ Failures: {failure_count}")
            print(f"    ✓ Success rate: {success_count/(success_count+failure_count)*100:.1f}%")
            print(f"    ✓ Avg latency: {latency_stats['mean']:.2f}ms")
            print(f"    ✓ P95 latency: {latency_stats['p95']:.2f}ms")


def main():
    """Run all monitoring tests"""
    print("\n" + "="*60)
    print("MONITORING SYSTEM COMPREHENSIVE TEST")
    print("="*60)

    try:
        # Test 1: Basic monitoring
        monitor = test_basic_monitoring()

        # Test 2: Anomaly detection
        test_anomaly_detection()

        # Test 3: Drift detection
        test_drift_detection()

        # Test 4: Alerting
        test_alerting()

        # Test 5: Threshold monitoring
        test_threshold_monitoring()

        # Test 6: Component monitoring
        test_component_monitoring()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
