import pytest
import pandas as pd
from synthpop import Synthpop
from datasets.adult import df, dtypes

def test_synthpop_default_parameters():
    """Test Synthpop with default parameters using Adult dataset."""
    # Initialize Synthpop
    spop = Synthpop()
    
    # Fit the model
    spop.fit(df, dtypes)
    
    # Generate synthetic data
    synth_df = spop.generate(len(df))
    
    # Verify the synthetic dataframe has the same shape as original
    assert synth_df.shape == df.shape
    
    # Verify the synthetic dataframe has the same columns as original
    assert all(synth_df.columns == df.columns)
    
    # Verify the method attribute contains expected default values
    assert isinstance(spop.method, pd.Series)
    assert 'age' in spop.method.index
    assert spop.method['age'] == 'sample'  # age should use sample method
    assert all(spop.method[spop.method != 'sample'] == 'cart')  # rest should use cart
    
    # Verify visit sequence is properly set
    assert isinstance(spop.visit_sequence, pd.Series)
    assert len(spop.visit_sequence) == len(df.columns)
    assert all(spop.visit_sequence.index == df.columns)
    
    # Verify predictor matrix is properly set
    assert isinstance(spop.predictor_matrix, pd.DataFrame)
    assert spop.predictor_matrix.shape == (len(df.columns), len(df.columns))
    assert all(spop.predictor_matrix.index == df.columns)
    assert all(spop.predictor_matrix.columns == df.columns)

def test_synthpop_custom_visit_sequence():
    """Test Synthpop with custom visit sequence using Adult dataset."""
    # Define custom visit sequence
    visit_sequence = [0, 1, 5, 3, 2]
    
    # Initialize Synthpop with custom visit sequence
    spop = Synthpop(visit_sequence=visit_sequence)
    
    # Fit the model
    spop.fit(df, dtypes)
    
    # Generate synthetic data
    synth_df = spop.generate(len(df))
    
    # Verify only specified columns were synthesized
    expected_columns = ['age', 'workclass', 'marital.status', 'education', 'fnlwgt']
    assert len(synth_df.columns) == len(expected_columns)
    assert all(col in synth_df.columns for col in expected_columns)
    
    # Verify visit sequence matches what was specified
    assert len(spop.visit_sequence) == len(visit_sequence)
    assert spop.visit_sequence['age'] == 0
    assert spop.visit_sequence['workclass'] == 1
    assert spop.visit_sequence['marital.status'] == 2
    assert spop.visit_sequence['education'] == 3
    assert spop.visit_sequence['fnlwgt'] == 4
    
    # Verify predictor matrix has correct shape for subset of columns
    assert spop.predictor_matrix.shape == (len(expected_columns), len(expected_columns))
    assert all(col in spop.predictor_matrix.columns for col in expected_columns)
    assert all(col in spop.predictor_matrix.index for col in expected_columns)
    
    # Verify specific predictor relationships from example
    pred_matrix = spop.predictor_matrix
    assert pred_matrix.loc['age', 'age'] == 0
    assert pred_matrix.loc['workclass', 'age'] == 1
    assert pred_matrix.loc['workclass', 'workclass'] == 0
    assert pred_matrix.loc['fnlwgt', ['age', 'workclass', 'education', 'marital.status']].sum() == 4
    assert pred_matrix.loc['education', ['age', 'workclass', 'marital.status']].sum() == 3
    assert pred_matrix.loc['marital.status', ['age', 'workclass']].sum() == 2
