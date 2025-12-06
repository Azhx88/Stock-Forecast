import plotly.graph_objects as go

def line_chart(x, y, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name=name))
    return fig
