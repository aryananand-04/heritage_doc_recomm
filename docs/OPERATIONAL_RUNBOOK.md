# Heritage Document Recommender - Operational Runbook

**Version**: 1.0
**Last Updated**: 2025-01-30
**Maintained By**: Operations Team

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Deployment](#deployment)
4. [Monitoring](#monitoring)
5. [Common Operational Tasks](#common-operational-tasks)
6. [Incident Response](#incident-response)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Performance Tuning](#performance-tuning)
9. [Maintenance Procedures](#maintenance-procedures)
10. [Escalation Procedures](#escalation-procedures)

---

## System Overview

### Purpose
The Heritage Document Recommender system provides intelligent recommendations for heritage documents using a multi-strategy approach combining SimRank, Horn's Index, and semantic embeddings.

### Key Components
1. **SimRank Engine**: Graph-based similarity ranking
2. **Horn's Index**: Heritage-specific relevance scoring
3. **Embedding System**: Semantic similarity using FAISS
4. **Query Classifier**: Adaptive query-type detection
5. **Ensemble Ranker**: Multi-strategy fusion
6. **Knowledge Graph**: Document relationship network

### Performance Targets
- **Latency P95**: < 2 seconds
- **Latency P99**: < 5 seconds
- **Availability**: 99.5%
- **Error Rate**: < 1%
- **NDCG@10**: > 0.65

---

## Architecture

### System Diagram
```
User Query → Query Classifier → Adaptive Recommender
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
                SimRank         Horn's Index      Embeddings
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                      ↓
                              Ensemble Ranker
                                      ↓
                              Final Results
```

### Data Flow
1. User submits query
2. Query classifier determines query type
3. Adaptive recommender selects component weights
4. Components compute scores in parallel
5. Ensemble ranker fuses results
6. Validation and filtering applied
7. Results returned to user

### Dependencies
- **External**: None
- **Internal**: Knowledge graph (pickle), FAISS index, trained models
- **Data**: Heritage documents corpus

---

## Deployment

### Prerequisites
```bash
# Python environment
Python 3.8+
pip install -r requirements.txt

# Required files
data/heritage_kg.gpickle
data/faiss_index/
models/ranker/query_classifier.pkl
models/ranker/lambda_mart.pkl
```

### Installation Steps

#### 1. Clone Repository
```bash
git clone <repository-url>
cd heritage_doc_recomm
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Verify Data Files
```bash
python scripts/verify_data.py
```

#### 4. Initialize System
```bash
# Load knowledge graph
python -c "from src.utils.kg_loader import load_knowledge_graph; kg = load_knowledge_graph()"

# Verify FAISS index
python -c "from src.utils.faiss_loader import load_faiss_index; index = load_faiss_index()"

# Test query classifier
python -c "from src.5_ranking.query_classifier import QueryTypeClassifier; clf = QueryTypeClassifier()"
```

#### 5. Start Monitoring
```bash
# Initialize monitoring system
python -c "from utils.monitoring import initialize_monitoring; monitor = initialize_monitoring(start_background=True)"
```

#### 6. Start Dashboard (Optional)
```bash
streamlit run src/8_dashboard/operational_dashboard.py
```

### Configuration

#### Monitoring Configuration (`config/monitoring.yaml`)
```yaml
retention_minutes: 60
anomaly_sensitivity: 3.0
drift_window_size: 100
alert_file: logs/alerts.jsonl

thresholds:
  ndcg_at_10:
    min: 0.6
    warning: 0.65
    critical: 0.5

  query_latency_p95:
    max: 2000
    warning: 1800
    critical: 3000

  error_rate:
    max: 0.01
    warning: 0.005
    critical: 0.05
```

#### Logging Configuration (`config/logging.yaml`)
```yaml
log_level: INFO
log_directory: logs/
enable_json_logging: true

loggers:
  query_logger:
    file: logs/queries.log
    level: INFO

  system_logger:
    file: logs/system.log
    level: INFO

  performance_logger:
    file: logs/performance.log
    level: DEBUG
```

---

## Monitoring

### Key Metrics

#### Quality Metrics
- **NDCG@10**: Ranking quality (target: > 0.65)
- **Precision@5**: Top-5 accuracy (target: > 0.55)
- **Diversity Score**: Result diversity (target: > 0.6)

#### Performance Metrics
- **Query Latency P95**: 95th percentile response time (target: < 2s)
- **Query Latency P99**: 99th percentile response time (target: < 5s)
- **Component Latency**: Per-component timing

#### System Health Metrics
- **Error Rate**: Percentage of failed queries (target: < 1%)
- **Fallback Rate**: Percentage using fallback (target: < 10%)
- **Cache Hit Rate**: Cache effectiveness (target: > 70%)

### Monitoring Dashboard

Access the operational dashboard:
```bash
streamlit run src/8_dashboard/operational_dashboard.py
```

Dashboard URL: `http://localhost:8501`

#### Dashboard Sections
1. **Overview**: Health status, active alerts, key metrics
2. **Metrics**: Detailed metric trends and statistics
3. **Alerts**: Active and historical alerts
4. **Components**: Individual component status
5. **Query Log**: Recent query history
6. **Alert History**: 24-hour alert timeline

### Log Files

#### Query Log (`logs/queries.log`)
Contains all query executions:
```json
{
  "timestamp": "2025-01-30T10:15:30Z",
  "event_type": "query",
  "query_id": "q_12345",
  "query_text": "ancient temple architecture",
  "num_results": 10,
  "latency_ms": 150.5,
  "components_used": ["simrank", "horn_index", "embedding"]
}
```

#### System Log (`logs/system.log`)
System events and errors:
```json
{
  "timestamp": "2025-01-30T10:15:30Z",
  "level": "ERROR",
  "message": "SimRank computation failed",
  "component": "simrank",
  "error_type": "MemoryError"
}
```

#### Performance Log (`logs/performance.log`)
Detailed performance metrics:
```json
{
  "timestamp": "2025-01-30T10:15:30Z",
  "event_type": "performance",
  "operation": "simrank_ranking",
  "duration_ms": 450.2,
  "success": true
}
```

#### Alert Log (`logs/alerts.jsonl`)
All triggered alerts:
```json
{
  "alert_id": "alert_12345",
  "level": "WARNING",
  "metric_name": "query_latency_p95",
  "message": "Query latency above warning threshold",
  "current_value": 1850.0,
  "threshold": 1800.0,
  "timestamp": "2025-01-30T10:15:30Z"
}
```

### Alert Configuration

Alerts are automatically triggered when metrics exceed thresholds. Configure alert handlers:

```python
from utils.monitoring import get_monitor

monitor = get_monitor()

# Custom alert handler
def slack_alert_handler(alert):
    # Send to Slack
    pass

monitor.alerts.register_handler(slack_alert_handler)
```

---

## Common Operational Tasks

### Task 1: Check System Health

```bash
# Quick health check
python scripts/health_check.py

# Expected output:
# System Health: HEALTHY (Score: 95/100)
# - No critical alerts
# - All components operational
# - Performance within targets
```

### Task 2: Restart Monitoring

```bash
# Stop monitoring
python -c "from utils.monitoring import get_monitor; get_monitor().stop_background_monitoring()"

# Start monitoring
python -c "from utils.monitoring import initialize_monitoring; initialize_monitoring(start_background=True)"
```

### Task 3: Clear Old Logs

```bash
# Archive logs older than 30 days
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;
find logs/ -name "*.log.gz" -mtime +90 -delete

# Archive alerts
find logs/ -name "*.jsonl" -mtime +30 -exec gzip {} \;
```

### Task 4: Update Knowledge Graph

```bash
# Backup current KG
cp data/heritage_kg.gpickle data/heritage_kg.gpickle.bak

# Run KG rebuild
python src/1_kg_construction/kg_builder.py

# Verify new KG
python scripts/verify_kg.py

# Reload in running system
python -c "from src.utils.kg_loader import reload_knowledge_graph; reload_knowledge_graph()"
```

### Task 5: Retrain Query Classifier

```bash
# Generate training data
python src/5_ranking/generate_training_data.py

# Train classifier
python src/5_ranking/train_query_classifier.py

# Evaluate performance
python src/5_ranking/evaluate_classifier.py

# Deploy new model
cp models/ranker/query_classifier_new.pkl models/ranker/query_classifier.pkl
```

### Task 6: Run Comprehensive Evaluation

```bash
# Run full evaluation suite
python src/7_evaluation/run_comprehensive_evaluation.py

# Generate reports
ls evaluation/reports/
# - comprehensive_evaluation_report.json
# - fairness_evaluation_report.json
# - explanation_quality_report.json
```

---

## Incident Response

### Severity Levels

| Level | Response Time | Description |
|-------|---------------|-------------|
| **P1 - Critical** | 15 minutes | System down or unusable |
| **P2 - High** | 1 hour | Major feature broken |
| **P3 - Medium** | 4 hours | Degraded performance |
| **P4 - Low** | Next business day | Minor issues |

### Incident Response Workflow

1. **Detect**: Alert triggered or user report
2. **Acknowledge**: Assign incident owner
3. **Diagnose**: Check logs, metrics, dashboard
4. **Mitigate**: Apply immediate fix
5. **Resolve**: Implement permanent solution
6. **Review**: Post-incident analysis

### P1 - Critical Incident Response

#### Scenario: System Completely Down

**Symptoms**:
- All queries failing (error rate > 50%)
- Dashboard unreachable
- Critical alerts firing

**Response**:
1. Check system health:
   ```bash
   python scripts/health_check.py
   ```

2. Review recent errors:
   ```bash
   tail -100 logs/system.log | grep ERROR
   ```

3. Check component status:
   ```bash
   python scripts/component_status.py
   ```

4. Attempt restart:
   ```bash
   python scripts/restart_system.py
   ```

5. If restart fails, escalate to on-call engineer

**Recovery**:
- Switch to backup system if available
- Notify stakeholders
- Document incident timeline

### P2 - High Severity Response

#### Scenario: Single Component Failure

**Symptoms**:
- Fallback rate > 20%
- Specific component errors in logs
- Performance degraded

**Response**:
1. Identify failing component:
   ```bash
   grep "ERROR" logs/system.log | grep -o '"component": "[^"]*"' | sort | uniq -c
   ```

2. Check component health:
   ```bash
   python scripts/test_component.py --component=simrank
   ```

3. Verify fallback working:
   ```bash
   # Should see results despite component failure
   python scripts/test_query.py --query="test query"
   ```

4. Fix component:
   ```bash
   # Reload KG if SimRank issue
   python -c "from src.utils.kg_loader import reload_knowledge_graph; reload_knowledge_graph()"

   # Rebuild FAISS if embedding issue
   python src/3_embeddings/build_faiss_index.py
   ```

### P3 - Medium Severity Response

#### Scenario: Performance Degradation

**Symptoms**:
- Latency P95 > 2 seconds
- NDCG@10 < 0.6
- No critical failures

**Response**:
1. Check recent performance trends:
   ```bash
   python scripts/analyze_performance.py --hours=24
   ```

2. Identify slow components:
   ```bash
   grep "performance" logs/performance.log | python scripts/analyze_latency.py
   ```

3. Check for anomalies:
   ```bash
   python scripts/check_anomalies.py
   ```

4. Optimize slow components or adjust weights

---

## Troubleshooting Guide

### Issue: High Query Latency

**Symptoms**: Query latency P95 > 2 seconds

**Diagnosis**:
```bash
# Check component latencies
python scripts/component_latency_analysis.py

# Check for slow queries
grep "latency_ms" logs/queries.log | awk -F'"latency_ms": ' '{print $2}' | sort -n | tail -20
```

**Solutions**:
1. Increase cache size
2. Optimize FAISS index parameters
3. Reduce top_k for slow components
4. Enable parallel component execution

### Issue: Low NDCG Scores

**Symptoms**: NDCG@10 < 0.6 consistently

**Diagnosis**:
```bash
# Analyze recent rankings
python scripts/analyze_ranking_quality.py

# Check component weights
python scripts/show_current_weights.py
```

**Solutions**:
1. Retrain LTR models with recent data
2. Update component weights
3. Verify ground truth quality
4. Check for data drift

### Issue: High Error Rate

**Symptoms**: Error rate > 1%

**Diagnosis**:
```bash
# Count errors by type
grep '"level": "ERROR"' logs/system.log | grep -o '"error_type": "[^"]*"' | sort | uniq -c

# Find error patterns
python scripts/analyze_errors.py --hours=24
```

**Solutions**:
1. Fix specific error types
2. Improve input validation
3. Add missing error handlers
4. Update fallback chains

### Issue: Memory Issues

**Symptoms**: MemoryError in logs, high RAM usage

**Diagnosis**:
```bash
# Check memory usage
ps aux | grep python | awk '{print $6/1024 " MB - " $11}'

# Profile memory
python -m memory_profiler scripts/profile_memory.py
```

**Solutions**:
1. Reduce FAISS index size
2. Implement batch processing
3. Clear caches periodically
4. Increase system RAM

### Issue: Knowledge Graph Corruption

**Symptoms**: SimRank failures, graph loading errors

**Diagnosis**:
```bash
# Verify KG integrity
python scripts/verify_kg.py

# Check KG statistics
python scripts/kg_stats.py
```

**Solutions**:
1. Restore from backup:
   ```bash
   cp data/heritage_kg.gpickle.bak data/heritage_kg.gpickle
   ```

2. Rebuild from source:
   ```bash
   python src/1_kg_construction/kg_builder.py
   ```

---

## Performance Tuning

### Optimization Checklist

#### 1. Component-Level Optimization

**SimRank**:
```python
# Reduce iterations for faster computation
simrank_config = {
    'max_iterations': 5,  # Default: 10
    'decay_factor': 0.8,
    'cache_size': 10000
}
```

**Horn's Index**:
```python
# Optimize batch size
horn_config = {
    'batch_size': 100,  # Adjust based on RAM
    'enable_caching': True
}
```

**FAISS**:
```python
# Use GPU acceleration if available
faiss.StandardGpuResources()

# Optimize index type
# IVF: Fast, approximate
# Flat: Accurate, slow
index_type = 'IVF'
```

#### 2. Query-Level Optimization

```python
# Adjust top_k per component
component_top_k = {
    'simrank': 20,      # Expensive, get fewer
    'horn_index': 50,   # Cheap, get more
    'embedding': 30     # Medium cost
}
```

#### 3. Ensemble Optimization

```python
# Use faster fusion method
ensemble_config = {
    'fusion_method': 'rrf',  # Fastest
    # 'fusion_method': 'cascade',  # Slower but better
}
```

### Performance Monitoring

```python
# Track performance over time
from utils.monitoring import get_monitor

monitor = get_monitor()

# Get performance percentiles
stats = monitor.metrics.get_statistics('query_latency', minutes=60)
print(f"P50: {stats['p50']:.0f}ms")
print(f"P95: {stats['p95']:.0f}ms")
print(f"P99: {stats['p99']:.0f}ms")
```

---

## Maintenance Procedures

### Daily Maintenance

```bash
# Check system health
python scripts/health_check.py

# Review alerts
tail -50 logs/alerts.jsonl

# Check disk space
df -h
```

### Weekly Maintenance

```bash
# Run comprehensive evaluation
python src/7_evaluation/run_comprehensive_evaluation.py

# Archive old logs
bash scripts/archive_logs.sh

# Update documentation
git pull origin main
```

### Monthly Maintenance

```bash
# Retrain models with recent data
python src/5_ranking/train_ltr.py

# Rebuild knowledge graph
python src/1_kg_construction/kg_builder.py

# Performance review
python scripts/generate_monthly_report.py

# Update dependencies
pip list --outdated
pip install --upgrade <packages>
```

### Quarterly Maintenance

```bash
# Major version updates
# Review and update requirements.txt

# Data quality audit
python scripts/data_quality_audit.py

# Security audit
pip-audit

# Capacity planning
python scripts/capacity_analysis.py
```

---

## Escalation Procedures

### On-Call Rotation

| Time Period | Primary | Secondary |
|-------------|---------|-----------|
| Mon-Tue | Engineer A | Engineer B |
| Wed-Thu | Engineer C | Engineer A |
| Fri-Sun | Engineer B | Engineer C |

### Escalation Path

1. **L1 - Operations Team** (First responder)
   - Monitor alerts
   - Basic troubleshooting
   - Execute runbook procedures

2. **L2 - Engineering Team** (Domain experts)
   - Complex debugging
   - Code fixes
   - Performance tuning

3. **L3 - Architecture Team** (System architects)
   - Design changes
   - Major incidents
   - Capacity planning

### Contact Information

```yaml
operations_team:
  email: ops@heritage-recommender.com
  slack: #heritage-ops
  pagerduty: heritage-ops-oncall

engineering_team:
  email: eng@heritage-recommender.com
  slack: #heritage-eng
  pagerduty: heritage-eng-oncall

architecture_team:
  email: arch@heritage-recommender.com
  slack: #heritage-arch
```

### Escalation Triggers

**Auto-escalate to L2 if**:
- P1 incident not resolved in 30 minutes
- Multiple P2 incidents in 1 hour
- Health score < 50 for > 15 minutes

**Auto-escalate to L3 if**:
- P1 incident not resolved in 2 hours
- System design issue identified
- Capacity limits reached

---

## Appendix

### A. Useful Commands

```bash
# Quick system check
python -c "from utils.monitoring import get_monitor; print(get_monitor().get_dashboard_data()['system_health'])"

# Test query
python -c "from src.5_ranking.adaptive_recommender import AdaptiveRecommender; rec = AdaptiveRecommender(); print(rec.rank_documents([], 'test', [], 1.0))"

# Count recent queries
wc -l logs/queries.log

# Find errors
grep -c ERROR logs/system.log

# Show active alerts
python -c "from utils.monitoring import get_monitor; alerts = get_monitor().alerts.get_active_alerts(); print(f'Active alerts: {len(alerts)}')"
```

### B. Configuration Files

All configuration files are in `config/`:
- `monitoring.yaml`: Monitoring thresholds
- `logging.yaml`: Logging configuration
- `recommender.yaml`: Recommender parameters
- `ltr.yaml`: Learning-to-rank configuration

### C. Backup Procedures

```bash
# Backup knowledge graph
cp data/heritage_kg.gpickle backups/heritage_kg_$(date +%Y%m%d).gpickle

# Backup models
tar -czf backups/models_$(date +%Y%m%d).tar.gz models/

# Backup logs (before archiving)
tar -czf backups/logs_$(date +%Y%m%d).tar.gz logs/
```

### D. Recovery Procedures

```bash
# Restore knowledge graph
cp backups/heritage_kg_YYYYMMDD.gpickle data/heritage_kg.gpickle

# Restore models
tar -xzf backups/models_YYYYMMDD.tar.gz

# Reload system
python scripts/reload_system.py
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-30 | Operations Team | Initial version |

---

## Feedback

For questions or improvements to this runbook:
- **Email**: ops@heritage-recommender.com
- **Slack**: #heritage-ops
- **Git**: Open an issue or PR
