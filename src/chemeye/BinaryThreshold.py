from typing import Iterable
from plotly import graph_objects as go
import plotly.express as px


class BinaryThreshold:
    def __init__(self, values:Iterable[float], decision_boundary:float) -> None:
        '''Class for plotting histograms of values colored by binary activity with decision boundary line.'''
        self.values = values
        self.decision_boundary = decision_boundary
        self.binary_values = [1 if value >= decision_boundary else 0 for value in values]
        
        self.histogram = self.__histogram
        
        return self.__histogram()
        

    @staticmethod
    def histogram(values:Iterable[float], binary_values:Iterable[int], decision_boundary:float) -> go.Figure:
        '''Plot histogram of values colored by binary activity with decision boundary line.'''
        # Compute number & percent active
        num_active = binary_values.count(1)
        num_total = len(binary_values)
        percent_active = round((num_active/num_total)*100, 2)

        # Historgram
        fig = px.histogram(x=values, color=binary_values, nbins=100, opacity=0.8,
                           title=f'Active: {num_active}/{num_total} ({percent_active}%)')
        fig.update_layout(yaxis_title='Count')

        # Threshold line
        fig.add_vline(x=decision_boundary, line_width=1, line_dash='dash', line_color='black',
                      annotation_text='Threshold', annotation_position='top right')

        return fig
    
    def __histogram(self):
        '''Instance method for BinaryThreshold.histogram.'''
        return BinaryThreshold.histogram(self.values, self.binary_values, self.decision_boundary)
