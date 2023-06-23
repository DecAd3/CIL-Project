import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
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


def _load_data_for_surprise(data_matrix):
    itemID = []
    userID = []
    rating = []

    for uid in range(data_matrix.shape[0]):
        for iid in range(data_matrix.shape[1]):
            userID.append(uid)
            itemID.append(iid)
            rating.append(data_matrix[uid,iid])
    
    ratings_dict = {'itemID': itemID,
                    'userID': userID,
                    'rating': rating}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data_surprise = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader=reader)

    return data_surprise

def preprocess(arr, n_row, n_col, imputation):
    ### Column Normalize
    masked = np.ma.masked_equal(arr, 0)
    mean_cols = np.tile(np.ma.mean(masked, axis=0).data, (n_row, 1))
    std_cols = np.tile(np.ma.std(masked, axis=0).data, (n_row, 1))
    normalized_arr = ((masked - mean_cols) / std_cols).data
    mask_arr = normalized_arr != 0

    ### Imputation
    if imputation == "zero":
        imputed_arr = normalized_arr
    elif imputation == "mean":
        imputed_arr = mean_cols * (normalized_arr == 0) + arr * (normalized_arr != 0)

    return imputed_arr, mask_arr, mean_cols, std_cols


def compute_rmse(predictions, labels):
    return math.sqrt(mean_squared_error(predictions, labels))


def train():
    pass