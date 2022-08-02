import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Iterable, Optional
import pandas as pd

from chemeye.arrays import tsne


class TSNEPlotter:
    def __init__(self, descriptors:np.array) -> None:
        self.__descriptors = np.copy(descriptors)
        
    def plot(self, x_name, y_name, color_category:Optional[Iterable]=None) -> go.Figure:
        arr = tsne(self.__descriptors)
        df = pd.DataFrame({
            x_name: arr[:, 0],
            y_name: arr[:, 1]
        })

        opacity = 1
        color = None
        if color_category:
            df['color'] = color_category
            opacity = 0.5
            color = 'color'

        df['color'] = df['color'].fillna('missing color')  # Replace NaN w/ string bc px doesn't like NaN
            
        return px.scatter(df, x=x_name, y=y_name, color=color, render_mode='svg', opacity=opacity)
