def create_metric_heatmap(results):
    """Create heatmap of all metrics across methods."""
    methods = list(results.keys())

    # Collect all unique metrics (only numeric ones)
    all_metrics = set()
    for method_result in results.values():
        for metric_name, metric_value in method_result.items():
            if isinstance(metric_value, (int, float)):
                all_metrics.add(metric_name)

    all_metrics = sorted(all_metrics)

    # Create matrix
    matrix = []
    for method in methods:
        row = []
        for metric in all_metrics:
            val = results[method].get(metric, 0)
            # Ensure it's a number
            if isinstance(val, (int, float)):
                row.append(val)
            else:
                row.append(0)
        matrix.append(row)

    # Format text safely
    text_matrix = []
    for row in matrix:
        text_row = []
        for val in row:
            try:
                text_row.append(f'{val:.4f}')
            except (ValueError, TypeError):
                text_row.append(str(val))
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_metrics,
        y=methods,
        colorscale='Blues',
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Method: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>'
    ))