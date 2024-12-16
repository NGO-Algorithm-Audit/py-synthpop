import unittest
from synthpop import Synthpop
import pandas as pd
import numpy as np

class TestProcessing(unittest.TestCase):

    def test_add_NaN_columns_for_numeric_columns(self):
        df = pd.DataFrame({'a':[1,2,np.nan], 'b':[1,1,1], 'c':['x','y',None]})
        spop = Synthpop()
        dtype_map = {'a':'float','b':'float', 'c':'categorical'}
        res,dtype_res = spop.pre_preprocess(df,dtype_map,nan_fill=-8)

        self.assertTrue('a_NaN' in res,"Nan column not made")
        self.assertFalse('b_NaN' in res,"Nan column should not be made if there are no NaNs")
        self.assertFalse('c_NaN' in res,"Nan column should not be made for categorical columns")
        self.assertTrue(res['a_NaN'][2])
        self.assertEqual(res['a'][2], -8)
        self.assertEqual(dtype_res['a_NaN'],'category')
        self.assertEqual(spop.map_column_to_NaN_column['a'],'a_NaN')
    def test_make_visit_sequence_when_one_is_given(self):

        visit_seq = ['x','a','b']
        spop = Synthpop(visit_sequence=visit_seq)
        spop.map_column_to_NaN_column = {'a':'a_NaN','c':'c_NaN'}

        spop.include_nan_columns()

        self.assertSequenceEqual(spop.visit_sequence,['x','a_NaN','a','b'])


    def test_apply_and_remove_added_NaN_columns(self):
        df = pd.DataFrame({'a':[1,2,-8],'a_NaN':[False,True,False], 'b':[1,1,1], 'c':['x','y',None]})

        spop = Synthpop()
        spop.map_column_to_NaN_column = {'a':'a_NaN'}

        res = spop.post_postprocessing(df)
        self.assertTrue(np.isnan(res['a'][1]), "NaNs should be placed where indicated")
        self.assertFalse('a_NaN' in res, "indicator columns should be removed")


if __name__ == '__main__':
    unittest.main()