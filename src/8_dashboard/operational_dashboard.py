"""
Operational dashboard for monitoring heritage document recommender system.

Features:
1. Real-time metrics visualization
2. Active alerts display
3. System health monitoring
4. Performance trends
5. Component status
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.monitoring import get_monitor, initialize_monitoring, AlertLevel
from utils.logger_v2 import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Heritage Recommender - Operations Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .health-good {
        color: #28a745;
        font-weight: bold;
    }
    .health-degraded {
        color: #ffc107;
        font-weight: bold;
    }
    .health-critical {
        color: #dc3545;
        font-weight: bold;
    }
    .alert-critical {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_monitoring_system():
    """Get or initialize monitoring system"""
    try:
        monitor = get_monitor()
        return monitor
    except Exception as e:
        st.error(f"Failed to initialize monitoring: {e}")
        return None


def render_health_status(health_data):
    """Render system health status"""
    st.header("System Health")

    score = health_data['score']
    status = health_data['status']

    # Color coding
    if status == 'healthy':
        health_class = 'health-good'
        color = '#28a745'
    elif status == 'degraded':
        health_class = 'health-degraded'
        color = '#ffc107'
    else:
        health_class = 'health-critical'
        color = '#dc3545'

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Health Score", f"{score:.1f}/100", delta=None)

    with col2:
        st.markdown(f'<p class="{health_class}">Status: {status.upper()}</p>',
                   unsafe_allow_html=True)

    with col3:
        st.metric("Critical Alerts", health_data.get('critical_alerts', 0))

    # Health gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#fff4cc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Issues
    if health_data.get('issues'):
        st.subheader("Active Issues")
        for issue in health_data['issues']:
            st.warning(f"⚠️ {issue}")


def render_active_alerts(alerts_data):
    """Render active alerts"""
    st.header("Active Alerts")

    if not alerts_data:
        st.success("✅ No active alerts")
        return

    # Group alerts by level
    critical_alerts = [a for a in alerts_data if a['level'] == 'CRITICAL']
    warning_alerts = [a for a in alerts_data if a['level'] == 'WARNING']
    info_alerts = [a for a in alerts_data if a['level'] == 'INFO']

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Critical", len(critical_alerts), delta=None)
    col2.metric("Warning", len(warning_alerts), delta=None)
    col3.metric("Info", len(info_alerts), delta=None)

    # Display alerts
    for alert in critical_alerts:
        render_alert(alert, 'critical')

    for alert in warning_alerts:
        render_alert(alert, 'warning')

    for alert in info_alerts:
        render_alert(alert, 'info')


def render_alert(alert, level):
    """Render individual alert"""
    alert_class = f"alert-{level}"

    timestamp = datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00'))
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

    alert_html = f"""
    <div class="{alert_class}">
        <strong>{alert['level']}</strong>: {alert['message']}<br>
        <small>Metric: {alert['metric_name']} | Value: {alert['current_value']:.2f} | Threshold: {alert['threshold']:.2f}</small><br>
        <small>Time: {time_str}</small>
    </div>
    """

    st.markdown(alert_html, unsafe_allow_html=True)


def render_key_metrics(metrics_data):
    """Render key performance metrics"""
    st.header("Key Metrics (Last 5 minutes)")

    # Quality metrics
    st.subheader("Quality Metrics")
    col1, col2, col3 = st.columns(3)

    if 'ndcg_at_10' in metrics_data:
        ndcg_stats = metrics_data['ndcg_at_10']
        col1.metric(
            "NDCG@10",
            f"{ndcg_stats['mean']:.3f}",
            delta=f"±{ndcg_stats['std']:.3f}"
        )

    if 'precision_at_5' in metrics_data:
        prec_stats = metrics_data['precision_at_5']
        col2.metric(
            "Precision@5",
            f"{prec_stats['mean']:.3f}",
            delta=f"±{prec_stats['std']:.3f}"
        )

    if 'diversity_score' in metrics_data:
        div_stats = metrics_data['diversity_score']
        col3.metric(
            "Diversity",
            f"{div_stats['mean']:.3f}",
            delta=f"±{div_stats['std']:.3f}"
        )

    # Performance metrics
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    if 'query_latency_p95' in metrics_data:
        latency_stats = metrics_data['query_latency_p95']
        col1.metric(
            "Latency P95",
            f"{latency_stats['p95']:.0f}ms",
            delta=None
        )

    if 'query_latency_p99' in metrics_data:
        latency_stats = metrics_data['query_latency_p99']
        col2.metric(
            "Latency P99",
            f"{latency_stats['p99']:.0f}ms",
            delta=None
        )

    if 'error_rate' in metrics_data:
        error_stats = metrics_data['error_rate']
        col3.metric(
            "Error Rate",
            f"{error_stats['mean']*100:.2f}%",
            delta=None
        )

    if 'fallback_rate' in metrics_data:
        fallback_stats = metrics_data['fallback_rate']
        col4.metric(
            "Fallback Rate",
            f"{fallback_stats['mean']*100:.2f}%",
            delta=None
        )


def render_metric_trends(monitor):
    """Render metric trends over time"""
    st.header("Metric Trends")

    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 5 minutes", "Last 15 minutes", "Last 30 minutes", "Last 1 hour"],
        index=2
    )

    time_mapping = {
        "Last 5 minutes": 5,
        "Last 15 minutes": 15,
        "Last 30 minutes": 30,
        "Last 1 hour": 60
    }
    minutes = time_mapping[time_range]

    # Metric selector
    available_metrics = [
        'query_latency', 'ndcg_at_10', 'error_rate',
        'fallback_rate', 'num_results', 'diversity_score'
    ]

    selected_metric = st.selectbox("Select Metric", available_metrics)

    # Get metric data
    values = monitor.metrics.get_recent_values(selected_metric, minutes=minutes)

    if values:
        # Create time series
        timestamps = [datetime.utcnow() - timedelta(minutes=minutes) +
                     timedelta(minutes=minutes * i / len(values))
                     for i in range(len(values))]

        df = pd.DataFrame({
            'Time': timestamps,
            'Value': values
        })

        # Plot
        fig = px.line(df, x='Time', y='Value', title=f'{selected_metric} over time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        st.subheader("Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{pd.Series(values).mean():.2f}")
        col2.metric("Median", f"{pd.Series(values).median():.2f}")
        col3.metric("Min", f"{pd.Series(values).min():.2f}")
        col4.metric("Max", f"{pd.Series(values).max():.2f}")
    else:
        st.info(f"No data available for {selected_metric} in the last {minutes} minutes")


def render_component_status(metrics_data):
    """Render status of individual components"""
    st.header("Component Status")

    components = ['simrank', 'horn_index', 'embedding', 'query_classifier']

    component_data = []
    for component in components:
        calls_metric = f'component_usage_{component}'
        failures_metric = f'{component}_failures'
        latency_metric = f'{component}_latency'

        row = {'Component': component}

        # Get usage count
        if calls_metric in metrics_data:
            calls = metrics_data[calls_metric]['count']
            row['Calls'] = calls
        else:
            row['Calls'] = 0

        # Get failure count
        if failures_metric in metrics_data:
            failures = metrics_data[failures_metric]['count']
            row['Failures'] = failures
            row['Success Rate'] = f"{(1 - failures/max(row['Calls'], 1)) * 100:.1f}%"
        else:
            row['Failures'] = 0
            row['Success Rate'] = "100%"

        # Get latency
        if latency_metric in metrics_data:
            latency = metrics_data[latency_metric]['mean']
            row['Avg Latency'] = f"{latency:.0f}ms"
        else:
            row['Avg Latency'] = "N/A"

        # Status
        if row['Calls'] > 0 and row['Failures'] / row['Calls'] < 0.05:
            row['Status'] = '✅ Healthy'
        elif row['Calls'] > 0:
            row['Status'] = '⚠️ Degraded'
        else:
            row['Status'] = '⚪ Inactive'

        component_data.append(row)

    df = pd.DataFrame(component_data)
    st.dataframe(df, use_container_width=True)


def render_recent_queries(monitor):
    """Render recent query log"""
    st.header("Recent Queries")

    # Read query log file
    log_file = Path('logs/queries.log')

    if not log_file.exists():
        st.info("No query logs available")
        return

    # Read last N lines
    with open(log_file) as f:
        lines = f.readlines()[-50:]  # Last 50 queries

    queries = []
    for line in lines:
        try:
            log_data = json.loads(line)
            if log_data.get('event_type') == 'query':
                queries.append({
                    'Query ID': log_data.get('query_id', 'N/A'),
                    'Timestamp': log_data.get('timestamp', 'N/A'),
                    'Results': log_data.get('num_results', 0),
                    'Latency (ms)': log_data.get('latency_ms', 0),
                    'Components': ', '.join(log_data.get('components_used', []))
                })
        except json.JSONDecodeError:
            continue

    if queries:
        df = pd.DataFrame(queries)
        st.dataframe(df.tail(20), use_container_width=True)
    else:
        st.info("No recent queries")


def render_alert_history():
    """Render alert history"""
    st.header("Alert History (Last 24 hours)")

    alert_file = Path('logs/alerts.jsonl')

    if not alert_file.exists():
        st.info("No alert history available")
        return

    # Read alerts
    alerts = []
    cutoff_time = datetime.utcnow() - timedelta(hours=24)

    with open(alert_file) as f:
        for line in f:
            try:
                alert_data = json.loads(line)
                alert_time = datetime.fromisoformat(alert_data['timestamp'].replace('Z', '+00:00'))

                if alert_time >= cutoff_time:
                    alerts.append({
                        'Timestamp': alert_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Level': alert_data['level'],
                        'Metric': alert_data['metric_name'],
                        'Message': alert_data['message'],
                        'Value': f"{alert_data['current_value']:.2f}",
                        'Threshold': f"{alert_data['threshold']:.2f}"
                    })
            except json.JSONDecodeError:
                continue

    if alerts:
        df = pd.DataFrame(alerts)

        # Count by level
        level_counts = df['Level'].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Critical", level_counts.get('CRITICAL', 0))
        col2.metric("Warning", level_counts.get('WARNING', 0))
        col3.metric("Info", level_counts.get('INFO', 0))

        # Timeline
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        alerts_timeline = df.groupby([pd.Grouper(key='Timestamp', freq='1H'), 'Level']).size().reset_index(name='Count')

        fig = px.bar(alerts_timeline, x='Timestamp', y='Count', color='Level',
                    title='Alerts over Time',
                    color_discrete_map={
                        'CRITICAL': '#dc3545',
                        'WARNING': '#ffc107',
                        'INFO': '#17a2b8'
                    })
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(df.sort_values('Timestamp', ascending=False), use_container_width=True)
    else:
        st.success("No alerts in the last 24 hours")


def main():
    """Main dashboard"""
    st.title("🏛️ Heritage Recommender - Operational Dashboard")
    st.markdown("Real-time monitoring and operational metrics")

    # Initialize monitoring
    monitor = get_monitoring_system()

    if monitor is None:
        st.error("Failed to initialize monitoring system")
        return

    # Refresh button
    if st.button("🔄 Refresh", key="refresh_button"):
        st.rerun()

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

    # Get dashboard data
    try:
        dashboard_data = monitor.get_dashboard_data()
    except Exception as e:
        st.error(f"Error fetching dashboard data: {e}")
        logger.error(f"Dashboard data fetch failed: {e}")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Metrics", "Alerts", "Components", "Query Log", "Alert History"]
    )

    # Render selected page
    if page == "Overview":
        render_health_status(dashboard_data['system_health'])
        st.divider()
        render_active_alerts(dashboard_data['alerts'])
        st.divider()
        render_key_metrics(dashboard_data['metrics'])

    elif page == "Metrics":
        render_key_metrics(dashboard_data['metrics'])
        st.divider()
        render_metric_trends(monitor)

    elif page == "Alerts":
        render_active_alerts(dashboard_data['alerts'])

    elif page == "Components":
        render_component_status(dashboard_data['metrics'])

    elif page == "Query Log":
        render_recent_queries(monitor)

    elif page == "Alert History":
        render_alert_history()

    # Footer
    st.sidebar.divider()
    st.sidebar.info(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == '__main__':
    main()
