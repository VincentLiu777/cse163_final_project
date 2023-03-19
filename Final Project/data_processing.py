'''
Vincent Liu, Zhiming Huang, Junjin Wang
CSE 163
Final Project
This program checked null value in the dataframe and removed null values,
sorted all white wines into two categories based on their quality values,
renamed columns of dataframe after sorting,
normalized values of each feature in the dataframe.
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def check_null(white_wine: pd.DataFrame) -> pd.DataFrame:
    '''
    This method takes a pandas DataFrame, and also returns a DataFrame after
    checking null values.
    It will check the null value in the dataframe.
    If there are null values in the dataframe, it will drop all null values.
    '''
    white_wine.dropna(inplace=True)
    white_wine['quality'] = white_wine['quality'].astype(int)
    return white_wine


def assign_target(x: int) -> int:
    '''
    This method takes an int which means the quality value of white wine, and
    returns an int.
    All quality values are integers.
    If the quality value is 0, 1, 2 ,3, 4, 5, 6, it will return -1, which
    means bad wine.
    If the quality value is 7, 8, 9, it will return 1, which means good wine.
    '''
    if ((x == 7) | (x == 8) | (x == 9)):
        return 1
    else:
        return -1


def split_quality(white_wine: pd.DataFrame) -> pd.DataFrame:
    '''
    This method takes a pandas DataFrame and returns a pandas DataFrame.
    It will apply the function(assign_target) we wrote as the standard for
    assigning the corresponding quality for each one in a new column called
    wine_class.
    '''
    white_wine['wine_class'] = white_wine['quality'].apply(assign_target)
    return white_wine


def rename_columns(white_wine: pd.DataFrame) -> pd.DataFrame:
    '''
    This method takes a pandas DataFrame, and returns a pandas DataFrame after
    renaming columns.
    It will rename those columns that have more than one words into one word.
    For example, fixed acidity will change to fixed_acidity.
    '''
    white_wine = white_wine.rename(
        columns={
            'fixed acidity': 'fixed_acidity',
            'volatile acidity': 'volatile_acidity',
            'citric acid': 'citric_acid',
            'residual sugar': 'residual_sugar',
            'free sulfur dioxide': 'free_sulfur_dioxide',
            'total sulfur dioxide': 'total_sulfur_dioxide'})
    return white_wine


def normalize(white_wine: pd.DataFrame) -> pd.DataFrame:
    '''
    This method takes a pandas DataFrame, and returns DataFrame.
    It normalizes values of each feature into 0 to 1.
    It transforms values of features to be on a similar scale.
    It improves the performance and training stability of the model.
    '''
    features_white = white_wine.drop('quality', axis='columns')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normal_df = scaler.fit_transform(features_white)
    normal_df = pd.DataFrame(normal_df, columns=features_white.columns)
    return normal_df
