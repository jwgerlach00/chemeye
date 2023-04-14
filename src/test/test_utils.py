import unittest
from chemeye import utils
import naclo
import numpy as np
import pandas as pd


class TestBuildDescriptorArr(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_mols = naclo.smiles_2_mols(['CN=C=O', 'CCC', 'O'])
        cls.other_df = pd.DataFrame({
            'a': [0.001, 0.2, 0.03],
            'b': [4, 5, 6],
            'c': [700, 800, 900]
        })
        return super().setUpClass()

    def test_ecfp_1024(self):
        out = utils.build_descriptor_arr(self.test_mols, 'ecfp', ecfp_radius=2, ecfp_n_bits=1024)
        self.assertEqual(out.shape, (3, 1024))
        
    def test_ecfp_2048(self):
        out = utils.build_descriptor_arr(self.test_mols, 'ecfp', ecfp_radius=2, ecfp_n_bits=2048)
        self.assertEqual(out.shape, (3, 2048))
        
    def test_ecfp_radius_3(self):
        rad2 = utils.build_descriptor_arr(self.test_mols, 'ecfp', ecfp_radius=2, ecfp_n_bits=1024)
        rad1 = utils.build_descriptor_arr(self.test_mols, 'ecfp', ecfp_radius=1, ecfp_n_bits=1024)
        self.assertFalse(np.array_equal(rad2, rad1))
        
    def test_ecfp_other_shape(self):
        out = utils.build_descriptor_arr(self.test_mols, 'ecfp_other', ecfp_radius=2, ecfp_n_bits=1024,
                                         other_descriptors_df=self.other_df)
        self.assertEqual(out.shape, (3, 1024+len(self.other_df.columns)))
        
    def test_ecfp_other_znorm(self):
        out = utils.build_descriptor_arr(self.test_mols, 'ecfp_other', ecfp_radius=2, ecfp_n_bits=1024,
                                         other_descriptors_df=self.other_df)
        other_arr = self.other_df.to_numpy()
        other_arr_znorm = (other_arr - other_arr.mean(axis=0)) / other_arr.std(axis=0)
        # other_znorm = (self.other_df - self.other_df.mean(axis=0)) / self.other_df.std(axis=0)
        print(other_arr_znorm)
        print(out[:, 1024:])
        # self.assertTrue(np.array_equal(out[:, 1024:], other_arr_znorm))
        

if __name__ == '__main__':
    unittest.main()
