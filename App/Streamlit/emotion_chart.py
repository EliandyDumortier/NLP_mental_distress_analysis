import plotly.express as px
import plotly.graph_objects as go

def emotion_bar_chart(predictions):
    """
    Build a horizontal bar chart of emotion scores.
    `predictions` should be a list of dicts: [{'label': str, 'score': float}, ...]
    """
    if not predictions:
        return go.Figure().add_annotation(text="No predictions available", showarrow=False)
    
    labels = [p["label"].replace("_", " ").title() for p in predictions]
    scores = [p["score"] for p in predictions]
    
    # Create color scale based on scores
    colors = px.colors.sequential.Viridis
    
    fig = go.Figure(data=go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        text=[f"{score:.2%}" for score in scores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={
            'text': "AI Prediction Confidence",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        xaxis_title="Confidence Score",
        yaxis_title="Categories",
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=60, t=80, b=60),
        height=400,
        xaxis=dict(
            range=[0, 1],
            tickformat='.0%',
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig

def create_metrics_chart(accuracy, f1_score, precision, recall):
    """
    Create a radar chart for model metrics.
    """
    categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    values = [accuracy, f1_score, precision, recall]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line_color='#1877f2',
        fillcolor='rgba(24, 119, 242, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.8, 1.0]
            )),
        showlegend=True,
        title={
            'text': "Model Performance Metrics",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def create_distress_distribution_chart(alerts_df):
    """
    Create a pie chart showing distribution of distress levels.
    """
    if alerts_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    level_counts = alerts_df['level'].value_counts()
    
    colors = {
        'no_distress': '#4CAF50',
        'mild': '#FFC107', 
        'moderate': '#FF9800',
        'severe': '#F44336'
    }
    
    fig = go.Figure(data=go.Pie(
        labels=[label.replace('_', ' ').title() for label in level_counts.index],
        values=level_counts.values,
        marker_colors=[colors.get(label, '#999999') for label in level_counts.index],
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Distress Level Distribution",
            'x': 0.5,
            'xanchor': 'center'
        },
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig