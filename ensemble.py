import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from utils import compute_rmse

class Ensemble:
    def __init__(self, args):
        self.fold_number = args.ens_args.fold_number
        self.shuffle = args.ens_args.shuffle
        self.regressor = args.ens_args.regressor
        self.data_ensemble_folder = args.ens_args.data_ensemble_folder
        self.model_list = args.ens_args.model_list
        self.seed_value = args.random_seed
        self.regressors = []

    def get_regressor(self):
        if self.regressor == 'linear':
            return LinearRegression()
        raise ValueError("illegal regressor type provided")
    
    def obtain_predictions_from_all_models_in_one_ensemble(self, df_train_fold, df_test_fold, fold_index):
        pred_train_all = None
        if df_train_fold is not None:
            pred_train_all = np.empty((0, 0))
        pred_test_all = None
        if df_test_fold is not None:
            pred_test_all = np.empty((0, 0))
        for model_name in self.model_list:
            pred_fn = self.data_ensemble_folder + model_name + "_" + str(fold_index) + '.txt'
            pred_ins = np.loadtxt(pred_fn) # check type, shape
            if df_train_fold is not None:
                pred_train = pred_ins[df_train_fold['row'].values - 1, df_train_fold['col'].values - 1]
                pred_train_all = np.hstack((pred_train_all, pred_train))
            if df_test_fold is not None:
                pred_test = pred_ins[df_test_fold['row'].values - 1, df_test_fold['col'].values - 1]
                pred_test_all = np.hstack((pred_test_all, pred_test))
        return pred_train_all, pred_test_all

    def train(self, df_train):
        assert(self.fold_number >= 2)
        kf = KFold(n_splits=self.fold_number, shuffle=self.shuffle, random_state=self.seed_value)
        for fold_index, (train_index, test_index) in enumerate(kf.split(df_train)):
            df_train_fold = df_train[train_index]
            df_test_fold = df_train[test_index]
            pred_train_all, pred_test_all = self.obtain_predictions_from_all_models_in_one_ensemble(df_train_fold, df_test_fold, fold_index)
            gt_train = df_train_fold['Prediction'].values
            gt_test = df_test_fold['Prediction'].values
            reg = self.get_regressor().fit(pred_train_all, gt_train)
            testing = reg.predict(pred_test_all)
            self.regressors.append(reg)
            print('RMSE (fold - {i}): {:.4f}'.format(fold_index, compute_rmse(testing, gt_test)))
        predict_whole_train = self.predict(df_train)
        gt_whole_train = df_train['Prediction'].values
        print('RMSE (whole training dataset): {:.4f}'.format(compute_rmse(predict_whole_train, gt_whole_train)))

    def predict(self, df_test):
        predict_res = np.empty((0, 0))
        for fold_index in self.fold_number:
            _, pred_test_all = self.obtain_predictions_from_all_models_in_one_ensemble(None, df_test, fold_index)
            pred_ins = self.regressors[fold_index].predict(pred_test_all)
            predict_res = np.hstack((predict_res, pred_ins))
        predict_res = np.mean(predict_res, axis=1)
        return predict_res 