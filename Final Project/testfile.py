'''
Vincent Liu, Zhiming Huang, Junjin Wang
CSE 163
Final Project
This is a file that we use to make sure our data processing
went smoothly. Since we cannot test the graph or the ML, so
data processing was the only section we tested on.
'''
import pandas as pd
from cse163_utils import assert_equals
import data_processing


def test_check_null(
        not_null_outcome_df: pd.DataFrame,
        original_df: pd.DataFrame) -> None:
    '''
    This test function will test whether we pass the check_null function
    in data_processing file. If it worked, it should return nothing. Otherwise,
    we will recevice an return of what we have, and what we expected.
    '''
    assert_equals(not_null_outcome_df, data_processing.check_null(original_df))


def test_split_quality(outcome_df: pd.DataFrame,
                       not_null_outcome_df: pd.DataFrame) -> None:
    '''
    This test function will test whether we pass the test_spilt_quality
    funciton in data_processing file. If it worked out, it should return
    nothing. Otherwise, we will recevice an return of what we have, and
    what we expected.
    '''
    assert_equals(
        outcome_df,
        data_processing.split_quality(not_null_outcome_df))
