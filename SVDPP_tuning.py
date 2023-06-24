import numpy as np
from utils import _load_data_for_surprise, _convert_df_to_matrix, compute_rmse, _read_df_in_format
import os
from surprise import SVDpp
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import pandas as pd

df = _read_df_in_format('./data/data_train.csv')
data = _load_data_for_surprise(df)

param_grid = {'n_factors':[8, 12], 'n_epochs': [15, 25], 'lr_all': [0.005, 0.09], 'reg_all': [0.01, 0.03]}

grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv = 3)

grid_search.fit(data)

print(grid_search.best_params)