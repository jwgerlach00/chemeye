import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Iterable, Optional
import pandas as pd
from matplotlib.colors import CSS4_COLORS

from chemeye.arrays import tsne


class TSNEPlotter:
    def __init__(self, descriptors:np.array) -> None:
        self.__descriptors = np.copy(descriptors)
    
    @staticmethod
    def css_color_map(color_category:Iterable) -> dict:
        unique_colors = set(color_category)
        
        css_colors = list(CSS4_COLORS.keys())
        
        color_map = {}
        for i, color in enumerate(unique_colors):
            color_map[color] = css_colors[i]
        return color_map
        
    def plot(self, x_name, y_name, color_category:Optional[Iterable]=None, css_color_map:bool=False) -> go.Figure:
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
        
        if css_color_map:
            return px.scatter(df, x=x_name, y=y_name, color=color, render_mode='svg', opacity=opacity,
                              color_discrete_map=self.css_color_map(color_category))
        else:
            return px.scatter(df, x=x_name, y=y_name, color=color, render_mode='svg', opacity=opacity,
                              color_discrete_sequence=px.colors.qualitative.Alphabet)
