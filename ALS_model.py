import os
import numpy as np
from utils import _convert_df_to_matrix, preprocess, postprocess, compute_rmse

class ALS_model:
    def __init__(self, args):
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.imputation = args.als_args.imputation
        self.svd_rank = args.als_args.svd_rank
        self.latent_dim = args.als_args.latent_dim
        self.num_iterations = args.als_args.num_iterations
        self.reg_param = args.als_args.reg_param
        self.seed_value = args.random_seed
        self.verbose = args.verbose
        self.save_predictions = args.save_predictions
        self.predictions_folder = args.predictions_folder
        self.model_name = 'als'
        
    def train(self, df_train):
        np.random.seed(self.seed_value)
        train_data, is_provided = _convert_df_to_matrix(df_train, self.num_users, self.num_items)
        train_data, data_mean, data_std = preprocess(train_data, self.num_users, self.num_items, self.imputation)
        # Initialize user and item latent factor matrices        
        U, sigma, VT = np.linalg.svd(train_data)
        user_factors = U[:, :self.latent_dim]
        item_factors = VT[:self.latent_dim, :]
        masked_A = is_provided * train_data

        for _ in range(self.num_iterations):
            if self.verbose:
                print("epoch: ", _)
            for row_idx in range(self.num_users):
                # Update user factors
                sigma1 = item_factors @ (is_provided[row_idx, :][:, np.newaxis] * item_factors.T)    # np.diag(is_provided[row_idx, :])
                sigma2 = item_factors @ masked_A[row_idx,:]
                update_factors = np.linalg.solve(sigma1 + self.reg_param * np.eye(self.latent_dim), sigma2)
                user_factors[row_idx, :] = update_factors # self.update_factors(train_data, item_factors, user_factors, is_provided, axis=0, index = row_idx)
            for col_idx in range(self.num_items):
                # Update item factors
                sigma1 = user_factors.T @ (is_provided[:, col_idx][:, np.newaxis] * user_factors) # np.diag(is_provided[:, col_idx])
                sigma2 = user_factors.T @ masked_A[:,col_idx]
                update_factors = np.linalg.solve(sigma1 + self.reg_param * np.eye(self.latent_dim), sigma2)
                item_factors[:, col_idx] = update_factors # self.update_factors(train_data, user_factors, item_factors, is_provided, axis=1, index = col_idx)
            if self.verbose:
                self.reconstructed_matrix = postprocess(user_factors @ item_factors, data_mean, data_std) 
                predictions = self.reconstructed_matrix[df_train['row'].values - 1, df_train['col'].values - 1]
                labels = df_train['Prediction'].values
                print('Current Train RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))
        self.reconstructed_matrix = postprocess(user_factors @ item_factors, data_mean, data_std)
            
    # def update_factors(self, ratings, fixed_factors, update_factors, is_provided, axis, index):
    #     np.random.seed(self.seed_value)
    #     # Solve for factors using least squares
    #     sigma1 = None
    #     sigma2 = None
    #     if axis == 0:   # update U
    #         for col in range(self.num_items):
    #             item1 = is_provided[index, col] * (fixed_factors[:,col][:, np.newaxis] @ fixed_factors[:, col][np.newaxis, :])
    #             item2 = is_provided[index, col] * ratings[index, col] * fixed_factors[:,col]
    #             if sigma1 is None:
    #                 sigma1 = item1
    #                 sigma2 = item2
    #             else:
    #                 sigma1 += item1
    #                 sigma2 += item2  
            
    #     else:           # update V
    #         for row in range(self.num_users):
    #             item1 = is_provided[row, index] * (fixed_factors[row, :][:, np.newaxis] @ fixed_factors[row, :][np.newaxis, :]) # check dimension
    #             item2 = is_provided[row, index] * ratings[row, index] * fixed_factors[row, :]
    #             if sigma1 is None:
    #                 sigma1 = item1
    #                 sigma2 = item2
    #             else:
    #                 sigma1 += item1
    #                 sigma2 += item2

    #     update_factors = np.linalg.solve(sigma1 + self.reg_param * np.eye(self.latent_dim), sigma2)

    #     return update_factors
    
    def predict(self, df_test):
        predictions = self.reconstructed_matrix[df_test['row'].values - 1, df_test['col'].values - 1]
        labels = df_test['Prediction'].values
        print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        if self.save_predictions:
            np.savetxt(os.path.join('.', self.predictions_folder, self.model_name + '_pred_full.txt'), self.reconstructed_matrix)
            np.savetxt(os.path.join('.', self.predictions_folder, self.model_name + '_pred_test.txt'), predictions) 

    def predict_submission(self, row_idx, col_idx):
        return self.reconstructed_matrix[row_idx, col_idx]