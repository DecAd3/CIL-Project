import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
import math
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def _load_data_for_surprise(df):
    itemID = []
    userID = []
    rating = []

    UIDs = df['row'].to_numpy()
    IIDs = df['col'].to_numpy()
    PREDs = df['Prediction'].to_numpy()

    for i in range(df.shape[0]):
        userID.append(UIDs[i]-1)
        itemID.append(IIDs[i]-1)
        rating.append(PREDs[i])
    
    ratings_dict = {'itemID': itemID,
                    'userID': userID,
                    'rating': rating}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data_surprise = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader=reader)

    return data_surprise


def _load_data_for_BFM(df):
    df.rename(columns={'row': 'user_id'}, inplace=True)
    df.rename(columns={'col': 'movie_id'}, inplace=True)
    df.rename(columns={'Prediction': 'rating'}, inplace=True)
    return df


def _load_data_for_VAE(arr, batch_size):
    data = torch.tensor(np.ma.masked_equal(arr, 0).data).float()
    mask = data != 0
    indices = torch.arange(data.shape[0])
    dataloader = DataLoader(TensorDataset(data, mask, indices), batch_size=batch_size, shuffle=True)
    return dataloader, data, mask, indices


def preprocess(arr, n_row, n_col, imputation):
    ### Column Normalization
    masked = np.ma.masked_equal(arr, 0)
    # to check: mean along row / col have effects on results?
    mean_cols = np.tile(np.ma.mean(masked, axis=0).data, (n_row, 1))
    std_cols = np.tile(np.ma.std(masked, axis=0).data, (n_row, 1))
    normalized_arr = ((masked - mean_cols) / std_cols).data

    ### Imputation
    if imputation == "zero":
        imputed_arr = normalized_arr
    elif imputation == "mean":
        imputed_arr = mean_cols * (normalized_arr == 0) + arr * (normalized_arr != 0)

    return imputed_arr, mean_cols, std_cols


def postprocess(raw_predictions, data_mean = None, data_std = None, min_rate = 1, max_rate = 5, denorm = True):
    denormalized_predictions = raw_predictions
    if denorm:
        denormalized_predictions = raw_predictions * data_std + data_mean
    clipped_predicted = np.clip(denormalized_predictions, min_rate, max_rate)
    return clipped_predicted


def compute_rmse(predictions, labels):
    return math.sqrt(mean_squared_error(predictions, labels))


def generate_submission(sub_sample_path, store_path, data_matrix, clip_min=1, clip_max=5):
    print("Start generating submissions...")

    # print("Loading requests specified by submission samples...")
    df = _read_df_in_format(sub_sample_path)
    nrows = df.shape[0]
    # print(f"Storing {nrows} records for submission as requested...")
    row_id = df['row'].to_numpy() - 1
    col_id = df['col'].to_numpy() - 1
    data_matrix = np.clip(data_matrix, clip_min, clip_max)
    df['Prediction'] = data_matrix[row_id, col_id]

    def reformat_id(record):
        return f"r{record['row']:.0f}_c{record['col']:.0f}"

    df['Id'] = df.apply(reformat_id, axis=1)
    df = df.drop(['row', 'col'], axis=1)
    df.to_csv(store_path, columns=['Id', 'Prediction'], index=False)

    print("Generating ends. ")
