from copy import copy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Iterable, Optional

from chemeye.arrays import tsne


class TSNEPlotter:
    def __init__(self, descriptors:np.array) -> None:
        self.__descriptors = np.copy(descriptors)
        
    def plot(self, x_name, y_name, color_category:Optional[Iterable]=None) -> go.Figure:
        arr = tsne(self.__descriptors)
        df = ({
            x_name: arr[:, 0],
            y_name: arr[:, 1]
        })

        opacity = 1
        color = None
        if color_category:
            df['color'] = color_category
            opacity = 0.25
            color = 'color'
            
        return px.scatter(df, x=x_name, y=y_name, color=color, render_mode='svg', opacity=opacity)
