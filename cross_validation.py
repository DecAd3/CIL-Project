from train import process_config, get_model
import os 
import sys
from utils import _convert_df_to_matrix, preprocess, postprocess, compute_rmse, generate_submission, _read_df_in_format
from sklearn.model_selection import KFold

def cross_validation(args):
    df = _read_df_in_format(args.train_data)
    cv_folder = args.cv_args.cv_folder
    model_name = args.model_name
    if not os.path.exists(cv_folder):
        os.makedirs(cv_folder)
    
    kf = KFold(n_splits=args.cv_args.fold_number, shuffle=True, random_state=args.random_seed)

    for idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        df_train = df.iloc[train_idx.tolist()]
        df_test = df.iloc[test_idx.tolist()]

        print('Fold {}: Number of training sumples: {}, number of test samples: {}.'.format(idx+1, len(df_train), len(df_test)))
        args.cv_args.cv_model_name = model_name + '_fold_' + str(idx)

        model = get_model(args, df_train, df_test)
        model.train(df_train)

if __name__ == '__main__':
    args = process_config(sys.argv[1])
    cross_validation(args)