"""
Evaluation Dashboard Page

Features:
- Performance metrics comparison
- Load evaluation results from results/evaluation_results.json
- Interactive charts for precision, recall, NDCG, MRR
- Comparison across different ranking methods
- Temporal analysis if available
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path


@st.cache_data
def load_evaluation_results():
    """Load evaluation results from JSON file."""
    results_path = Path('results/evaluation_results.json')

    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def create_metrics_comparison_chart(results):
    """Create bar chart comparing different metrics."""
    methods = list(results.keys())
    metrics = ['precision@10', 'recall@10', 'ndcg@10', 'mrr']

    data = {metric: [] for metric in metrics}

    for method in methods:
        for metric in metrics:
            if metric in results[method]:
                data[metric].append(results[method][metric])
            else:
                data[metric].append(0)

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=methods,
            y=data[metric],
            marker_color=colors[i]
        ))

    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="Method",
        yaxis_title="Score",
        barmode='group',
        height=500,
        hovermode='x unified'
    )

    return fig


def create_radar_chart(results):
    """Create radar chart for multi-dimensional comparison."""
    methods = list(results.keys())
    metrics = ['precision@10', 'recall@10', 'ndcg@10', 'mrr']

    fig = go.Figure()

    for method in methods:
        values = []
        for metric in metrics:
            if metric in results[method]:
                values.append(results[method][metric])
            else:
                values.append(0)

        # Close the polygon
        values.append(values[0])
        metrics_closed = metrics + [metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[m.upper() for m in metrics_closed],
            fill='toself',
            name=method
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Multi-Dimensional Performance Comparison",
        height=500
    )

    return fig


def create_precision_recall_curve(results):
    """Create precision-recall curve if available."""
    fig = go.Figure()

    for method, metrics in results.items():
        # Check if we have precision/recall at different k values
        k_values = []
        precisions = []
        recalls = []

        for key, value in metrics.items():
            if key.startswith('precision@'):
                k = int(key.split('@')[1])
                k_values.append(k)
                precisions.append(value)

        for key, value in metrics.items():
            if key.startswith('recall@'):
                k = int(key.split('@')[1])
                if k in k_values:
                    idx = k_values.index(k)
                    if idx < len(recalls):
                        recalls[idx] = value
                    else:
                        recalls.append(value)

        if precisions and recalls and len(precisions) == len(recalls):
            # Sort by k values
            sorted_data = sorted(zip(k_values, precisions, recalls))
            k_sorted, prec_sorted, rec_sorted = zip(*sorted_data)

            fig.add_trace(go.Scatter(
                x=rec_sorted,
                y=prec_sorted,
                mode='lines+markers',
                name=method,
                text=[f'k={k}' for k in k_sorted],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Recall: %{x:.4f}<br>' +
                             'Precision: %{y:.4f}<br>' +
                             '%{text}<extra></extra>'
            ))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        hovermode='closest'
    )

    return fig


def create_metric_heatmap(results):
    """Create heatmap of all metrics across methods."""
    methods = list(results.keys())

    # Collect all unique metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(all_metrics)

    # Create matrix
    matrix = []
    for method in methods:
        row = []
        for metric in all_metrics:
            row.append(results[method].get(metric, 0))
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_metrics,
        y=methods,
        colorscale='Blues',
        text=[[f'{val:.4f}' for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Method: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Evaluation Metrics Heatmap",
        xaxis_title="Metric",
        yaxis_title="Method",
        height=400,
        xaxis={'tickangle': -45}
    )

    return fig


def render():
    """Render evaluation dashboard page."""
    st.markdown('<h1 class="main-header">📈 Evaluation Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("""
    Compare performance metrics across different ranking methods.
    Metrics include Precision, Recall, NDCG (Normalized Discounted Cumulative Gain),
    and MRR (Mean Reciprocal Rank).
    """)

    # Load evaluation results
    results = load_evaluation_results()

    if results is None:
        st.warning("No evaluation results found at `results/evaluation_results.json`")

        st.info("""
        **To generate evaluation results:**

        1. Run the evaluation script:
           ```bash
           python src/5_evaluation/evaluate_ranker.py
           ```

        2. Results will be saved to `results/evaluation_results.json`
        """)

        # Show mock data for demonstration
        if st.checkbox("Show Demo with Mock Data"):
            results = {
                "SimRank": {
                    "precision@5": 0.72,
                    "precision@10": 0.68,
                    "recall@5": 0.45,
                    "recall@10": 0.62,
                    "ndcg@5": 0.75,
                    "ndcg@10": 0.71,
                    "mrr": 0.78
                },
                "Horn's Index": {
                    "precision@5": 0.65,
                    "precision@10": 0.61,
                    "recall@5": 0.41,
                    "recall@10": 0.58,
                    "ndcg@5": 0.68,
                    "ndcg@10": 0.64,
                    "mrr": 0.70
                },
                "Embeddings": {
                    "precision@5": 0.80,
                    "precision@10": 0.74,
                    "recall@5": 0.52,
                    "recall@10": 0.68,
                    "ndcg@5": 0.82,
                    "ndcg@10": 0.77,
                    "mrr": 0.84
                },
                "Hybrid (0.4/0.3/0.3)": {
                    "precision@5": 0.85,
                    "precision@10": 0.79,
                    "recall@5": 0.56,
                    "recall@10": 0.74,
                    "ndcg@5": 0.87,
                    "ndcg@10": 0.82,
                    "mrr": 0.89
                }
            }

            st.success("Showing mock evaluation data for demonstration")
        else:
            return

    # Display results
    st.markdown("---")
    st.markdown("### 📊 Performance Overview")

    # Method comparison table
    methods = list(results.keys())

    # Create summary table
    summary_data = []
    for method in methods:
        summary_data.append({
            'Method': method,
            'Precision@10': results[method].get('precision@10', 'N/A'),
            'Recall@10': results[method].get('recall@10', 'N/A'),
            'NDCG@10': results[method].get('ndcg@10', 'N/A'),
            'MRR': results[method].get('mrr', 'N/A')
        })

    df_summary = pd.DataFrame(summary_data)

    # Highlight best values
    def highlight_max(s):
        if s.name == 'Method':
            return [''] * len(s)
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]

    st.dataframe(
        df_summary.style.apply(highlight_max, axis=0),
        use_container_width=True,
        hide_index=True
    )

    # Key metrics
    st.markdown("---")
    st.markdown("### 🎯 Key Metrics")

    metric_cols = st.columns(len(methods))

    for i, method in enumerate(methods):
        with metric_cols[i]:
            st.markdown(f"**{method}**")

            precision = results[method].get('precision@10', 0)
            recall = results[method].get('recall@10', 0)
            ndcg = results[method].get('ndcg@10', 0)
            mrr = results[method].get('mrr', 0)

            st.metric("Precision@10", f"{precision:.4f}" if precision else "N/A")
            st.metric("Recall@10", f"{recall:.4f}" if recall else "N/A")
            st.metric("NDCG@10", f"{ndcg:.4f}" if ndcg else "N/A")
            st.metric("MRR", f"{mrr:.4f}" if mrr else "N/A")

    # Visualizations
    st.markdown("---")
    st.markdown("### 📈 Visualizations")

    # Metrics comparison bar chart
    metrics_chart = create_metrics_comparison_chart(results)
    st.plotly_chart(metrics_chart, use_container_width=True)

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Radar chart
        radar_chart = create_radar_chart(results)
        st.plotly_chart(radar_chart, use_container_width=True)

    with viz_col2:
        # Heatmap
        heatmap = create_metric_heatmap(results)
        st.plotly_chart(heatmap, use_container_width=True)

    # Precision-Recall curve
    st.markdown("---")
    pr_curve = create_precision_recall_curve(results)
    if pr_curve.data:
        st.plotly_chart(pr_curve, use_container_width=True)

    # Detailed metrics breakdown
    st.markdown("---")
    st.markdown("### 🔍 Detailed Metrics")

    selected_method = st.selectbox("Select Method", methods)

    if selected_method:
        st.markdown(f"**All metrics for {selected_method}:**")

        metrics_data = []
        for metric, value in sorted(results[selected_method].items()):
            metrics_data.append({
                'Metric': metric.upper(),
                'Value': f"{value:.6f}" if isinstance(value, (int, float)) else value
            })

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Best performing method
    st.markdown("---")
    st.markdown("### 🏆 Best Performing Method")

    # Determine best method for each metric
    best_methods = {}
    metrics_to_check = ['precision@10', 'recall@10', 'ndcg@10', 'mrr']

    for metric in metrics_to_check:
        best_score = -1
        best_method = None

        for method in methods:
            score = results[method].get(metric, 0)
            if score > best_score:
                best_score = score
                best_method = method

        best_methods[metric] = (best_method, best_score)

    best_cols = st.columns(len(metrics_to_check))

    for i, metric in enumerate(metrics_to_check):
        with best_cols[i]:
            method, score = best_methods[metric]
            st.success(f"**{metric.upper()}**\n\n{method}\n\n{score:.4f}")

    # Overall best (average rank)
    st.markdown("**Overall Best Method (by average rank):**")

    method_ranks = {method: [] for method in methods}

    for metric in metrics_to_check:
        # Rank methods for this metric
        scores = [(method, results[method].get(metric, 0)) for method in methods]
        scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (method, _) in enumerate(scores, 1):
            method_ranks[method].append(rank)

    avg_ranks = {method: sum(ranks) / len(ranks) for method, ranks in method_ranks.items()}
    best_overall = min(avg_ranks, key=avg_ranks.get)

    st.info(f"🥇 **{best_overall}** (Average Rank: {avg_ranks[best_overall]:.2f})")

    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Evaluation Results")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        if st.button("Export Summary as CSV", use_container_width=True):
            csv = df_summary.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="evaluation_summary.csv",
                mime="text/csv"
            )

    with export_col2:
        if st.button("Export Full Results as JSON", use_container_width=True):
            st.download_button(
                "Download JSON",
                data=json.dumps(results, indent=2),
                file_name="evaluation_results_full.json",
                mime="application/json"
            )


if __name__ == "__main__":
    render()
