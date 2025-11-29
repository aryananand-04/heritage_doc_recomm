"""
Production monitoring system for heritage document recommender.

This module provides:
1. Real-time metrics collection
2. Anomaly detection
3. Data drift monitoring
4. Alerting system
5. Performance tracking
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import threading
from enum import Enum

from utils.logger_v2 import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    RATE = 'rate'


@dataclass
class Alert:
    """Represents a monitoring alert"""
    alert_id: str
    level: AlertLevel
    metric_name: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    context: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'level': self.level.name,
            'metric_name': self.metric_name,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time"""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]

    def to_dict(self) -> Dict:
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


class MetricCollector:
    """Collects and aggregates metrics"""

    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.lock = threading.Lock()

    def record(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        snapshot = MetricSnapshot(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )

        with self.lock:
            self.metrics[metric_name].append(snapshot)

        logger.debug(f"Recorded metric: {metric_name}={value}", extra={
            'extra_fields': {'metric_name': metric_name, 'value': value, 'tags': tags}
        })

    def increment(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        # Get the last value
        with self.lock:
            last_value = 0.0
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                last_value = self.metrics[metric_name][-1].value

        self.record(metric_name, last_value + value, tags)

    def get_recent_values(self, metric_name: str, minutes: int = 5) -> List[float]:
        """Get metric values from the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self.lock:
            if metric_name not in self.metrics:
                return []

            recent = [
                snapshot.value
                for snapshot in self.metrics[metric_name]
                if snapshot.timestamp >= cutoff_time
            ]
            return recent

    def get_statistics(self, metric_name: str, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get statistical summary of a metric"""
        values = self.get_recent_values(metric_name, minutes)

        if not values:
            return None

        values_array = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'p50': float(np.percentile(values_array, 50)),
            'p95': float(np.percentile(values_array, 95)),
            'p99': float(np.percentile(values_array, 99))
        }

    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.retention_minutes)

        with self.lock:
            for metric_name in list(self.metrics.keys()):
                # Remove old snapshots
                while (self.metrics[metric_name] and
                       self.metrics[metric_name][0].timestamp < cutoff_time):
                    self.metrics[metric_name].popleft()

                # Remove empty metrics
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]


class AnomalyDetector:
    """Detects anomalies in metrics using statistical methods"""

    def __init__(self, sensitivity: float = 3.0):
        """
        Args:
            sensitivity: Number of standard deviations for z-score threshold
        """
        self.sensitivity = sensitivity
        self.baselines: Dict[str, Dict[str, float]] = {}

    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric"""
        if not values:
            return

        values_array = np.array(values)
        self.baselines[metric_name] = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'median': float(np.median(values_array)),
            'mad': float(np.median(np.abs(values_array - np.median(values_array))))
        }

    def detect_zscore(self, metric_name: str, value: float) -> bool:
        """Detect anomaly using z-score method"""
        if metric_name not in self.baselines:
            return False

        baseline = self.baselines[metric_name]
        if baseline['std'] == 0:
            return False

        z_score = abs((value - baseline['mean']) / baseline['std'])
        return z_score > self.sensitivity

    def detect_mad(self, metric_name: str, value: float) -> bool:
        """Detect anomaly using Median Absolute Deviation (robust to outliers)"""
        if metric_name not in self.baselines:
            return False

        baseline = self.baselines[metric_name]
        if baseline['mad'] == 0:
            return False

        # Modified z-score using MAD
        mad_score = abs(0.6745 * (value - baseline['median']) / baseline['mad'])
        return mad_score > self.sensitivity

    def detect(self, metric_name: str, value: float, method: str = 'mad') -> bool:
        """
        Detect if a value is anomalous

        Args:
            metric_name: Name of the metric
            value: Current value to check
            method: Detection method ('zscore' or 'mad')
        """
        if method == 'zscore':
            return self.detect_zscore(metric_name, value)
        else:
            return self.detect_mad(metric_name, value)


class DriftDetector:
    """Detects data drift in distributions"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reference_distributions: Dict[str, np.ndarray] = {}

    def set_reference(self, distribution_name: str, values: List[float]):
        """Set reference distribution"""
        self.reference_distributions[distribution_name] = np.array(values)

    def detect_drift_ks(self, distribution_name: str, current_values: List[float],
                        threshold: float = 0.05) -> tuple[bool, float]:
        """
        Detect drift using Kolmogorov-Smirnov test

        Returns:
            (is_drift, p_value)
        """
        if distribution_name not in self.reference_distributions:
            return False, 1.0

        if len(current_values) < 30:  # Need sufficient samples
            return False, 1.0

        from scipy import stats

        reference = self.reference_distributions[distribution_name]
        current = np.array(current_values)

        statistic, p_value = stats.ks_2samp(reference, current)

        return p_value < threshold, p_value

    def detect_drift_psi(self, distribution_name: str, current_values: List[float],
                         bins: int = 10, threshold: float = 0.2) -> tuple[bool, float]:
        """
        Detect drift using Population Stability Index (PSI)

        PSI < 0.1: No significant change
        PSI 0.1-0.2: Small change
        PSI > 0.2: Significant change

        Returns:
            (is_drift, psi_value)
        """
        if distribution_name not in self.reference_distributions:
            return False, 0.0

        if len(current_values) < 30:
            return False, 0.0

        reference = self.reference_distributions[distribution_name]
        current = np.array(current_values)

        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        curr_hist, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions
        ref_props = ref_hist / len(reference)
        curr_props = curr_hist / len(current)

        # Calculate PSI
        psi = 0.0
        for ref_p, curr_p in zip(ref_props, curr_props):
            if ref_p > 0 and curr_p > 0:  # Avoid log(0)
                psi += (curr_p - ref_p) * np.log(curr_p / ref_p)

        return psi > threshold, psi


class AlertManager:
    """Manages alerts with deduplication and notification"""

    def __init__(self, alert_file: Optional[Path] = None):
        self.alert_file = alert_file or Path('logs/alerts.jsonl')
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

        self.active_alerts: Dict[str, Alert] = {}
        self.alert_cooldown_minutes = 15  # Don't re-alert for same issue within 15 min
        self.lock = threading.Lock()

        self.alert_handlers: List[Callable[[Alert], None]] = []

    def register_handler(self, handler: Callable[[Alert], None]):
        """Register a custom alert handler"""
        self.alert_handlers.append(handler)

    def create_alert(self, metric_name: str, level: AlertLevel, message: str,
                     current_value: float, threshold: float,
                     context: Optional[Dict[str, Any]] = None) -> Optional[Alert]:
        """Create and process an alert"""
        alert_key = f"{metric_name}_{level.name}"

        with self.lock:
            # Check if we already have an active alert for this
            if alert_key in self.active_alerts:
                existing_alert = self.active_alerts[alert_key]
                time_since_last = datetime.utcnow() - existing_alert.timestamp

                if time_since_last.total_seconds() < self.alert_cooldown_minutes * 60:
                    # Still in cooldown period, don't create duplicate
                    return None

            # Create new alert
            alert = Alert(
                alert_id=f"{alert_key}_{int(time.time())}",
                level=level,
                metric_name=metric_name,
                message=message,
                current_value=current_value,
                threshold=threshold,
                timestamp=datetime.utcnow(),
                context=context or {}
            )

            self.active_alerts[alert_key] = alert

        # Log alert
        logger.warning(f"Alert created: {message}", extra={
            'extra_fields': alert.to_dict()
        })

        # Write to file
        self._write_alert(alert)

        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        return alert

    def _write_alert(self, alert: Alert):
        """Write alert to JSONL file"""
        with open(self.alert_file, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')

    def clear_alert(self, metric_name: str, level: AlertLevel):
        """Clear an active alert"""
        alert_key = f"{metric_name}_{level.name}"
        with self.lock:
            if alert_key in self.active_alerts:
                del self.active_alerts[alert_key]

    def get_active_alerts(self, min_level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by minimum level"""
        with self.lock:
            alerts = list(self.active_alerts.values())

        if min_level:
            alerts = [a for a in alerts if a.level.value >= min_level.value]

        return sorted(alerts, key=lambda a: (a.level.value, a.timestamp), reverse=True)


class MonitoringSystem:
    """
    Main monitoring system that coordinates metrics, anomaly detection, and alerting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        self.metrics = MetricCollector(
            retention_minutes=config.get('retention_minutes', 60)
        )
        self.anomaly_detector = AnomalyDetector(
            sensitivity=config.get('anomaly_sensitivity', 3.0)
        )
        self.drift_detector = DriftDetector(
            window_size=config.get('drift_window_size', 100)
        )
        self.alerts = AlertManager(
            alert_file=Path(config.get('alert_file', 'logs/alerts.jsonl'))
        )

        # Monitoring configuration
        self.thresholds = self._load_default_thresholds()
        self.thresholds.update(config.get('thresholds', {}))

        # Background monitoring
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None

        logger.info("Monitoring system initialized", extra={
            'extra_fields': {'config': config}
        })

    def _load_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load default monitoring thresholds"""
        return {
            # Quality metrics
            'ndcg_at_10': {'min': 0.6, 'warning': 0.65, 'critical': 0.5},
            'precision_at_5': {'min': 0.5, 'warning': 0.55, 'critical': 0.4},
            'diversity_score': {'min': 0.6, 'warning': 0.65, 'critical': 0.5},

            # Performance metrics
            'query_latency_p95': {'max': 2000, 'warning': 1800, 'critical': 3000},
            'query_latency_p99': {'max': 5000, 'warning': 4000, 'critical': 8000},
            'component_failure_rate': {'max': 0.05, 'warning': 0.03, 'critical': 0.1},

            # System health
            'error_rate': {'max': 0.01, 'warning': 0.005, 'critical': 0.05},
            'fallback_rate': {'max': 0.1, 'warning': 0.05, 'critical': 0.2},
            'cache_hit_rate': {'min': 0.7, 'warning': 0.75, 'critical': 0.5},

            # Data quality
            'empty_result_rate': {'max': 0.05, 'warning': 0.03, 'critical': 0.1},
            'avg_results_returned': {'min': 5, 'warning': 7, 'critical': 3},
        }

    def record_query(self, query_id: str, latency_ms: float, num_results: int,
                     ndcg: float, components_used: List[str],
                     had_error: bool = False, used_fallback: bool = False):
        """Record metrics for a query"""
        # Performance metrics
        self.metrics.record('query_latency', latency_ms)
        self.metrics.record('num_results', num_results)

        # Quality metrics
        self.metrics.record('ndcg_at_10', ndcg)

        # System health
        self.metrics.increment('total_queries')
        if had_error:
            self.metrics.increment('error_count')
        if used_fallback:
            self.metrics.increment('fallback_count')
        if num_results == 0:
            self.metrics.increment('empty_results')

        # Component usage
        for component in components_used:
            self.metrics.increment(f'component_usage_{component}')

        # Check thresholds in real-time
        self._check_thresholds()

    def record_component_performance(self, component_name: str, latency_ms: float,
                                     success: bool, error_type: Optional[str] = None):
        """Record performance metrics for a specific component"""
        self.metrics.record(f'{component_name}_latency', latency_ms)
        self.metrics.increment(f'{component_name}_calls')

        if not success:
            self.metrics.increment(f'{component_name}_failures')
            if error_type:
                self.metrics.increment(f'{component_name}_error_{error_type}')

    def _check_thresholds(self):
        """Check if any metrics exceed thresholds"""
        for metric_name, thresholds in self.thresholds.items():
            stats = self.metrics.get_statistics(metric_name, minutes=5)

            if not stats:
                continue

            current_value = stats['mean']

            # Check max threshold
            if 'max' in thresholds:
                if current_value > thresholds['critical']:
                    self.alerts.create_alert(
                        metric_name=metric_name,
                        level=AlertLevel.CRITICAL,
                        message=f"{metric_name} critically high: {current_value:.2f} > {thresholds['critical']:.2f}",
                        current_value=current_value,
                        threshold=thresholds['critical'],
                        context={'stats': stats}
                    )
                elif current_value > thresholds['warning']:
                    self.alerts.create_alert(
                        metric_name=metric_name,
                        level=AlertLevel.WARNING,
                        message=f"{metric_name} above warning threshold: {current_value:.2f} > {thresholds['warning']:.2f}",
                        current_value=current_value,
                        threshold=thresholds['warning'],
                        context={'stats': stats}
                    )

            # Check min threshold
            if 'min' in thresholds:
                if current_value < thresholds['critical']:
                    self.alerts.create_alert(
                        metric_name=metric_name,
                        level=AlertLevel.CRITICAL,
                        message=f"{metric_name} critically low: {current_value:.2f} < {thresholds['critical']:.2f}",
                        current_value=current_value,
                        threshold=thresholds['critical'],
                        context={'stats': stats}
                    )
                elif current_value < thresholds['warning']:
                    self.alerts.create_alert(
                        metric_name=metric_name,
                        level=AlertLevel.WARNING,
                        message=f"{metric_name} below warning threshold: {current_value:.2f} < {thresholds['warning']:.2f}",
                        current_value=current_value,
                        threshold=thresholds['warning'],
                        context={'stats': stats}
                    )

    def check_anomalies(self, lookback_minutes: int = 60):
        """Check for anomalies in recent metrics"""
        # Update baselines using historical data
        for metric_name in self.thresholds.keys():
            historical_values = self.metrics.get_recent_values(metric_name, lookback_minutes)
            if len(historical_values) >= 30:
                self.anomaly_detector.update_baseline(metric_name, historical_values)

        # Check recent values for anomalies
        for metric_name in self.thresholds.keys():
            recent_values = self.metrics.get_recent_values(metric_name, minutes=5)
            if not recent_values:
                continue

            current_value = recent_values[-1]
            is_anomaly = self.anomaly_detector.detect(metric_name, current_value, method='mad')

            if is_anomaly:
                self.alerts.create_alert(
                    metric_name=metric_name,
                    level=AlertLevel.WARNING,
                    message=f"Anomaly detected in {metric_name}: {current_value:.2f}",
                    current_value=current_value,
                    threshold=0.0,  # No fixed threshold for anomalies
                    context={'detection_method': 'MAD', 'recent_values': recent_values[-10:]}
                )

    def check_drift(self, distribution_name: str, current_values: List[float],
                    method: str = 'psi') -> bool:
        """
        Check for data drift in a distribution

        Args:
            distribution_name: Name of the distribution
            current_values: Current distribution samples
            method: Detection method ('psi' or 'ks')

        Returns:
            True if drift detected
        """
        if method == 'psi':
            is_drift, psi_value = self.drift_detector.detect_drift_psi(
                distribution_name, current_values
            )

            if is_drift:
                self.alerts.create_alert(
                    metric_name=f'drift_{distribution_name}',
                    level=AlertLevel.WARNING,
                    message=f"Data drift detected in {distribution_name} (PSI={psi_value:.3f})",
                    current_value=psi_value,
                    threshold=0.2,
                    context={'method': 'PSI', 'sample_size': len(current_values)}
                )
        else:
            is_drift, p_value = self.drift_detector.detect_drift_ks(
                distribution_name, current_values
            )

            if is_drift:
                self.alerts.create_alert(
                    metric_name=f'drift_{distribution_name}',
                    level=AlertLevel.WARNING,
                    message=f"Data drift detected in {distribution_name} (p={p_value:.4f})",
                    current_value=p_value,
                    threshold=0.05,
                    context={'method': 'KS', 'sample_size': len(current_values)}
                )

        return is_drift

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current data for monitoring dashboard"""
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {},
            'alerts': [],
            'system_health': {}
        }

        # Get statistics for key metrics
        for metric_name in self.thresholds.keys():
            stats = self.metrics.get_statistics(metric_name, minutes=5)
            if stats:
                dashboard_data['metrics'][metric_name] = stats

        # Get active alerts
        active_alerts = self.alerts.get_active_alerts()
        dashboard_data['alerts'] = [alert.to_dict() for alert in active_alerts]

        # Calculate system health score
        dashboard_data['system_health'] = self._calculate_health_score()

        return dashboard_data

    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score (0-100)"""
        health_score = 100.0
        issues = []

        # Check critical metrics
        critical_checks = [
            ('error_rate', 'max'),
            ('query_latency_p95', 'max'),
            ('ndcg_at_10', 'min')
        ]

        for metric_name, check_type in critical_checks:
            stats = self.metrics.get_statistics(metric_name, minutes=5)
            if not stats:
                continue

            value = stats['mean']
            thresholds = self.thresholds.get(metric_name, {})

            if check_type == 'max' and 'max' in thresholds:
                if value > thresholds['critical']:
                    health_score -= 30
                    issues.append(f"{metric_name} critically high")
                elif value > thresholds['warning']:
                    health_score -= 10
                    issues.append(f"{metric_name} above normal")

            elif check_type == 'min' and 'min' in thresholds:
                if value < thresholds['critical']:
                    health_score -= 30
                    issues.append(f"{metric_name} critically low")
                elif value < thresholds['warning']:
                    health_score -= 10
                    issues.append(f"{metric_name} below normal")

        # Factor in active critical alerts
        critical_alerts = self.alerts.get_active_alerts(min_level=AlertLevel.CRITICAL)
        health_score -= len(critical_alerts) * 15

        return {
            'score': max(0, min(100, health_score)),
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'critical',
            'issues': issues,
            'critical_alerts': len(critical_alerts)
        }

    def start_background_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring thread"""
        if self.monitoring_enabled:
            logger.warning("Background monitoring already running")
            return

        self.monitoring_enabled = True

        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    # Check thresholds
                    self._check_thresholds()

                    # Check for anomalies
                    self.check_anomalies(lookback_minutes=60)

                    # Cleanup old metrics
                    self.metrics.cleanup_old_metrics()

                    # Sleep
                    time.sleep(interval_seconds)

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval_seconds)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info(f"Background monitoring started (interval={interval_seconds}s)")

    def stop_background_monitoring(self):
        """Stop background monitoring thread"""
        if not self.monitoring_enabled:
            return

        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Background monitoring stopped")


# Global monitoring instance
_global_monitor: Optional[MonitoringSystem] = None


def get_monitor(config: Optional[Dict[str, Any]] = None) -> MonitoringSystem:
    """Get or create global monitoring instance"""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = MonitoringSystem(config)

    return _global_monitor


def initialize_monitoring(config: Optional[Dict[str, Any]] = None,
                         start_background: bool = True) -> MonitoringSystem:
    """Initialize monitoring system"""
    monitor = get_monitor(config)

    if start_background:
        monitor.start_background_monitoring(interval_seconds=60)

    return monitor
