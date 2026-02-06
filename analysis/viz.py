import plotly.express as px
import numpy as np

def fig_scatter_identity(x, y, x_label: str, y_label: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig = px.scatter(x=x, y=y, labels={"x": x_label, "y": y_label}, title="Scatter con línea identidad")
    try:
        minv = float(np.nanmin([x.min(), y.min()]))
        maxv = float(np.nanmax([x.max(), y.max()]))
        fig.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv,
                      line=dict(color="red", dash="dash"))
    except Exception:
        pass
    return fig