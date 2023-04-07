import pandas as pd
import numpy as np
import naclo
import stse


def build_descriptor_arr(mols:list, descriptor_type:str, ecfp_radius:int=None, ecfp_n_bits:int=None,
              other_descriptors_df:pd.DataFrame=None) -> np.ndarray:
    if descriptor_type == 'ecfp':
        descriptors = naclo.mols_2_ecfp(mols, radius=ecfp_radius, n_bits=ecfp_n_bits, return_numpy=True)
    elif descriptor_type == 'ecfp_other':
        descriptors = naclo.mols_2_ecfp_plus_descriptors(mols, other_descriptors_df, ecfp_radius=ecfp_radius,
                                                         ecfp_n_bits=ecfp_n_bits, z_norm=True)
    elif descriptor_type == 'other':
        descriptors = stse.dataframes.z_norm(other_descriptors_df)
    elif descriptor_type == 'maccs':
        descriptors = naclo.mols_2_maccs(mols)
    else:
        raise(ValueError, 'Invalid descriptor type')
    
    return np.array(descriptors)
