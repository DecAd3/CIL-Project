from train import process_config, get_model
import os 
import sys
import numpy as np
from utils import _convert_df_to_matrix, preprocess, postprocess, compute_rmse, generate_submission, _read_df_in_format
from sklearn.model_selection import KFold

def cross_validation(args):
    np.random.seed(args.random_seed)
    df = _read_df_in_format(args.train_data)
    df_submission_test = None
    if args.cv_args.save_full_pred:
        df_submission_test = _read_df_in_format(args.sample_data)
    cv_folder = args.cv_args.cv_folder
    model_name = args.model_name
    model_instance_name = args.model_instance_name

    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)
    
    kf = KFold(n_splits=args.cv_args.fold_number, shuffle=True, random_state=args.random_seed)

    for idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        df_train = df.iloc[train_idx.tolist()]
        df_test = df.iloc[test_idx.tolist()]

        print('Cross Validation - Fold {}: Number of training sumples: {}, number of test samples: {}.'.format(idx+1, len(df_train), len(df_test)))

        model_name_base = model_instance_name + '_fold_' + str(idx)
        model = get_model(args, df_train.copy(deep=True), df_test.copy(deep=True))
        model.train(df_train.copy(deep=True))   # , df_test.copy(deep=True)
        args.cv_args.cv_model_name = model_name_base + "_train.txt"
        model.predict(df.copy(deep=True), pred_file_name = args.cv_args.cv_model_name)
        args.cv_args.cv_model_name = model_name_base + "_test.txt"
        model.predict(df_submission_test.copy(deep=True), pred_file_name = args.cv_args.cv_model_name)

if __name__ == '__main__':
    args = process_config(sys.argv[1])
    cross_validation(args)