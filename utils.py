import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error


def _read_df_in_format(root):
    def reformat_id(id):
        # split and reformat the df
        row, col = id.split('_')
        return int(row[1:]), int(col[1:])
    df = pd.read_csv(root)
    df['row'], df['col'] = zip(*df['Id'].map(reformat_id))
    df.drop('Id', axis=1, inplace=True)
    return df


def _convert_df_to_matrix(df, n_row, n_col):
    row_id = df['row'].to_numpy() - 1 # id starts from 1
    col_id = df['col'].to_numpy() - 1

    data_matrix = np.zeros((n_row, n_col), dtype=np.int8)
    # Check! Data type could cause rounding errors!
    data_matrix[row_id, col_id] = df['Prediction'].to_numpy()
    is_provided = data_matrix != 0

    return data_matrix, is_provided


def preprocess(arr, n_row, n_col, imputation):
    masked = np.ma.masked_equal(arr, 0)
    col_mean = np.tile(np.ma.mean(masked, axis=0).data, (n_row, 1))
    col_std = np.tile(np.ma.std(masked, axis=0).data, (n_row, 1))
    normalized_cols = ((masked - col_mean) / col_std).data

    if imputation == "zero":
        imputed = np.nan_to_num(normalized_cols, nan=0.0)
    elif imputation == "mean":
        pass

    return imputed, col_mean.data, col_std.data


def compute_rmse(predictions, labels):
    return math.sqrt(mean_squared_error(predictions, labels))


def train():
    pass