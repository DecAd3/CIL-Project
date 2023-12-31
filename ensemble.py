import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from utils import compute_rmse, postprocess, _read_df_in_format
import pandas as pd

class Ensemble_Model:
    def __init__(self, args):
        self.fold_number = args.ens_args.fold_number
        self.shuffle = args.ens_args.shuffle
        self.regressor = args.ens_args.regressor
        self.data_ensemble_folder = args.ens_args.data_ensemble_folder
        self.model_list = args.ens_args.model_list
        self.seed_value = args.random_seed
        self.regressors = []
        self.generate_submissions = args.generate_submissions
        self.sample_data = args.sample_data
        self.submission_folder = args.submission_folder

    # return regessor for blending
    def get_regressor(self):
        if self.regressor == 'linear':
            return LinearRegression()
        elif self.regressor == 'SGD':
            return SGDRegressor(random_state=self.seed_value)
        elif self.regressor == 'BayesianRidge':
            return BayesianRidge()
        elif self.regressor == 'GradientBoost':
            return GradientBoostingRegressor(random_state=self.seed_value)    # learning_rate=0.15, , n_estimators = 100
        raise ValueError("illegal regressor type provided")
    
    # In one ensemble, all meta models defined in self.model_list are trained with same fold data.
    # Given fold index k used to train meta models in a specific ensemble, if mode is "train",
    # this function returns two arrays with shape N1*d, N2*d respectively, encoding prediction results under different meta models, 
    # where N1 is the number of training data in k-th fold,
    # N2 is the number of testing data in k-th fold and d is the number of meta models, i.e. len(self.model_list).
    # If mode is "test", directly return an array with shape N*d, where N is the number of instances in sampleSubmission.csv
    def obtain_predictions_from_all_models_in_one_ensemble(self, train_indices, test_indices, fold_index, mode="train"):
        pred_train_all = None
        pred_test_all = None
        for model_idx in range(len(self.model_list)):
            model_name = self.model_list[model_idx]
            pred_fn = self.data_ensemble_folder + model_name + "_fold_" + str(fold_index) + "_" + mode + '.txt' # "_cv" + str(self.fold_number) + 
            pred_ins = np.loadtxt(pred_fn)
            
            pred_train = None
            if train_indices is not None:
                pred_train = pred_ins[train_indices]
            else:
                pred_train = pred_ins
            if pred_train_all is None:
                pred_train_all = np.empty((pred_train.shape[0], len(self.model_list)))
            pred_train_all[:, model_idx] = pred_train
            if test_indices is not None:
                pred_test = pred_ins[test_indices]
                if pred_test_all is None:
                    pred_test_all = np.empty((pred_test.shape[0], len(self.model_list)))
                pred_test_all[:, model_idx] = pred_test
        return pred_train_all, pred_test_all

    def train(self, df_train):
        assert(self.fold_number >= 2)
        kf = KFold(n_splits=self.fold_number, shuffle=self.shuffle, random_state=self.seed_value)
        print("Start training ensemble ...")
        rmse_test_all = 0
        for fold_index, (train_index, test_index) in enumerate(kf.split(df_train)):
            df_train_fold = df_train.iloc[train_index.tolist()]
            df_test_fold = df_train.iloc[test_index.tolist()]
            pred_train_all, pred_test_all = self.obtain_predictions_from_all_models_in_one_ensemble(train_index.tolist(), test_index.tolist(), fold_index)
            gt_train = df_train_fold['Prediction'].values
            gt_test = df_test_fold['Prediction'].values
            # train and append the blending regressor
            reg = self.get_regressor().fit(pred_test_all, gt_test)
            self.regressors.append(reg)
            testing_test = reg.predict(pred_test_all)
            testing_train = reg.predict(pred_train_all) 
            rmse_train = compute_rmse(testing_train, gt_train)
            rmse_test = compute_rmse(testing_test, gt_test)
            rmse_test_all += rmse_test
            print('RMSE (fold - {}): train data - {:.4f}, test data - {:.4f}'.format(fold_index, rmse_train, rmse_test))
        predict_whole_train = self.predict(df_train, mode="train")
        gt_whole_train = df_train['Prediction'].values
        print('RMSE (whole training dataset): {:.4f}'.format(compute_rmse(predict_whole_train, gt_whole_train)))
        rmse_test_all /= (self.fold_number * 1.0)
        print('RMSE (testing average): {:.4f}'.format(rmse_test_all))

    def predict(self, df_test=None, mode="test"):
        predict_res = None
        for fold_index in range(self.fold_number):
            pred, _ = self.obtain_predictions_from_all_models_in_one_ensemble(None, None, fold_index, mode=mode)
            # layer 1 ensemble: apply blending regressor
            pred_ins = self.regressors[fold_index].predict(pred)
            if predict_res is None:
                predict_res = np.empty((pred.shape[0], self.fold_number))
            predict_res[:, fold_index] = pred_ins
        # layer 2 ensemble: apply averaging
        predict_res = np.mean(predict_res, axis=1)
        predict_res = postprocess(predict_res, denorm=False)
        if self.generate_submissions and mode == "test":
            df = pd.read_csv(self.sample_data)
            df['Prediction'] = predict_res
            submission_file = self.submission_folder + '/ensemble_' + self.regressor + "_" + str(self.model_list) + "_" + str (self.fold_number) 
            submission_file += '.csv'
            df.to_csv(submission_file, index=False)
        return predict_res