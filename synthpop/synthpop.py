import numpy as np
import pandas as pd

# classes
from synthpop.validator import Validator
from synthpop.processor import Processor
# global variables
from synthpop import NUM_COLS_DTYPES
from synthpop.processor import NAN_KEY
from synthpop.method import CART_METHOD, METHODS_MAP, NA_METHODS


class Synthpop:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 # predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=False,
                 default_method=CART_METHOD,
                 numtocat=None,
                 catgroups=None,
                 seed=None):
        # initialise the validator and processor
        self.validator = Validator(self)
        self.processor = Processor(self)

        # initialise arguments
        self.method = method
        self.visit_sequence = visit_sequence
        self.predictor_matrix = None
        self.proper = proper
        self.cont_na = cont_na
        self.smoothing = smoothing
        self.default_method = default_method
        self.numtocat = numtocat
        self.catgroups = catgroups
        self.seed = seed
        self.map_column_to_NaN_column = {}
        # check init
        self.validator.check_init()

    def include_nan_columns(self):
        for (col,nan_col) in self.map_column_to_NaN_column.items():
            if col not in self.visit_sequence:
                continue

            index_of_col = self.visit_sequence.index(col)
            self.visit_sequence.insert(index_of_col,nan_col)

    def pre_preprocess(self,df,dtypes,nan_fill):
        for column in df:
            if dtypes[column] != 'float':
                continue
            maybe_nans = df[column].isnull()
            if not maybe_nans.any():
                continue

            df.loc[maybe_nans,column] = nan_fill

            nan_col_name = column+"_NaN"
            df.loc[:,nan_col_name] = maybe_nans
            self.map_column_to_NaN_column[column] = nan_col_name

            dtypes[nan_col_name] = 'category'

        return df,dtypes

    def post_postprocessing(self,syn_df):
        for column in syn_df:
            if column in self.map_column_to_NaN_column.keys():
                nan_col_name = self.map_column_to_NaN_column[column]
                column_NaN_at = syn_df[nan_col_name]
                syn_df.loc[column_NaN_at,column] = None
                syn_df = syn_df.drop(columns=nan_col_name)

        return syn_df

    def fit(self, df, dtypes=None):
        # TODO check df and check/EXTRACT dtypes
        # - all column names of df are unique
        # - all columns data of df are consistent
        # - all dtypes of df are correct ('int', 'float', 'datetime', 'category', 'bool'; no object)
        # - can map dtypes (if given) correctly to df
        # should create map col: dtype (self.df_dtypes)
        df,dtypes = self.pre_preprocess(df,dtypes,-8)

        self.df_columns = df.columns.tolist()
        # Only set visit_sequence if not provided in init
        if self.visit_sequence is None:
            self.visit_sequence = df.columns.tolist()
        elif isinstance(self.visit_sequence, list) and all(isinstance(x, int) for x in self.visit_sequence):
            # Convert numeric indices to column names
            self.visit_sequence = [df.columns[i] for i in self.visit_sequence]
        
        self.include_nan_columns()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes

        # check processor
        self.validator.check_processor()
        # preprocess

        #processor.preprocess has side effects on the processor object and on this (self) object
        #processor.processing_dict[NAN_KEY][col]
        #spop.df_dtypes[col_nan_name]
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        print(processed_df)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)

        # check fit
        self.validator.check_fit()
        # fit
        self._fit(processed_df)

    def _fit(self, df):
        self.saved_methods = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, visit_step in self.visit_sequence.sort_values().items():
            print('train_{}'.format(col))

            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)
            # fit the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            # save the method
            self.saved_methods[col] = col_method

    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        return self.post_postprocessing(processed_synth_df)

    def _generate(self):
        # Only generate columns that were in the visit sequence
        synth_df = pd.DataFrame(data=np.zeros([self.k, len(self.visit_sequence)]), columns=self.visit_sequence.index)

        for col, visit_step in self.visit_sequence.sort_values().items():
            print('generate_{}'.format(col))

            # reload the method
            col_method = self.saved_methods[col]
            # predict with the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            synth_df[col] = col_method.predict(synth_df[col_predictors])

            # change all missing values to 0
            if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                synth_df.loc[nan_indices, col] = 0

            # map dtype to original dtype (only excpetion if column is full of NaNs)
            if synth_df[col].notna().any():
                synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        return synth_df
