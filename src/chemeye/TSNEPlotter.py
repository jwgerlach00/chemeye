import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Iterable, Optional, Tuple
import pandas as pd
from matplotlib.colors import CSS4_COLORS

from chemeye.arrays import tsne


class TSNEPlotter:
    X_NAME = 'tsne-x'
    Y_NAME = 'tsne-y'
    
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
    
    @staticmethod
    def tsne_df(descriptors, x_name:str=X_NAME, y_name:str=Y_NAME) -> pd.DataFrame:
        arr = tsne(descriptors)
        return pd.DataFrame({
            x_name: arr[:, 0],
            y_name: arr[:, 1]
        })
    
    @staticmethod
    def plot(df:pd.DataFrame, x_col_name:str=X_NAME, y_col_name:str=Y_NAME, color_category:Optional[Iterable]=None,
             css_color_map:bool=False) -> go.Figure:
        opacity = 0.5
        color = None
        
        if color_category:
            df['color'] = color_category
            color = 'color'
            df['color'] = df['color'].fillna('missing color')  # Replace NaN w/ string bc px doesn't like NaN
        
        if css_color_map:
            plot = px.scatter(df, x=x_col_name, y=y_col_name, color=color, render_mode='svg', opacity=opacity,
                              color_discrete_map=TSNEPlotter.css_color_map(color_category))
        else:
            plot = px.scatter(df, x=x_col_name, y=y_col_name, color=color, render_mode='svg', opacity=opacity,
                              color_discrete_sequence=px.colors.qualitative.Alphabet)
        return plot
    
    def main(self, color_category:Optional[Iterable]=None, css_color_map:bool=False) -> Tuple[go.Figure, pd.DataFrame]:
        df = self.tsne_df(self.__descriptors)
        return (
            TSNEPlotter.plot(df, color_category=color_category, css_color_map=css_color_map),
            df
        )
        