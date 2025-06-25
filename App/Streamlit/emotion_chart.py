import plotly.express as px

def emotion_bar_chart(predictions):
    """
    Build a horizontal bar chart of emotion scores.
    `predictions` should be a list of dicts: [{'label': str, 'score': float}, ...]
    """
    labels = [p["label"] for p in predictions]
    scores = [p["score"] for p in predictions]

    fig = px.bar(
        x=scores,
        y=labels,
        orientation="h",
        labels={"x": "Score", "y": "Emotion"},
        title="Emotion Distribution",
    )
    fig.update_layout(margin=dict(l=100, r=20, t=50, b=20))
    return fig
