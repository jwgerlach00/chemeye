from typing import Iterable, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from rdkit import DataStructs


class SimMat:
    def __init__(self, row_prints:Iterable, col_prints:Iterable, key_type:str='ecfp',
                 labels:Optional[Iterable[str]]=None) -> None:
        self.__row_prints = row_prints
        self.__col_prints = col_prints
        self.__key_type = key_type
        self.__labels = labels
    
    @staticmethod
    def sim_matrix(row_prints:Iterable, col_prints:Iterable, key_type:str='ecfp') -> np.array:
        full_array = np.zeros((len(row_prints), len(col_prints)))

        for i, row in enumerate(row_prints):
            for j, col in enumerate(col_prints):
                if key_type == 'ecfp':
                    similarity = DataStructs.TanimotoSimilarity(row, col)
                elif key_type == 'maccs':
                    similarity = DataStructs.FingerprintSimilarity(row, col)
                else:
                    raise('Key type must be either "ecfp" or "maccs".')
                    
                full_array[i][j] = similarity
        
        return full_array
    
    @staticmethod
    def plot(sim_arr:np.array, labels:Optional[Iterable[str]]=None) -> go.Figure:
        fig = px.imshow(sim_arr, x=labels, y=labels) if labels else px.imshow(sim_arr)
        fig.update_layout(title={ 'x': 0.5 })
        return fig
    
    def main(self) -> Tuple[go.Figure, np.array]:
        sim_arr = self.sim_matrix(row_prints=self.__row_prints, col_prints=self.__col_prints, key_type=self.__key_type)
        fig = self.plot(sim_arr, labels=self.__labels)
        return (fig, sim_arr)