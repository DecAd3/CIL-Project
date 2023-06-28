import itertools
import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

from recommenders.utils.timer import Timer
from recommenders.ncf_singlenode import NCF
from recommenders.dataset import Dataset as NCFDataset
from utils import _read_df_in_format, _convert_df_to_matrix, preprocess, postprocess, generate_submission


class NCF_model:
    def __init__(self, args):
        self.EPOCHS = args.ncf_args.EPOCHS
        self.BATCH_SIZE = args.ncf_args.BATCH_SIZE
        self.SEED = args.random_seed
        self.n_factors = args.ncf_args.n_factors
        self.learning_rate = args.ncf_args.learning_rate
        self.train_file = args.ncf_args.train_file
        self.sample_submission = args.sample_data
        self.save_file = args.ncf_args.save_file
        self.all_predictions_file = args.ncf_args.all_predictions_file
        self.model = None
        self.data_mean = 0
        self.data_std = 0
        self.num_users = args.num_users
        self.num_items = args.num_items

    def train(self, df_train):
        train_data, is_provided = _convert_df_to_matrix(df_train, self.num_users, self.num_items)
        train_data, self.data_mean, self.data_std = preprocess(train_data, self.num_users, self.num_items, "zero")

        users = []
        items = []
        ratings = []
        for i in range(len(train_data)):
            row = train_data[i]
            # row = (row - np.min(row)) / (np.max(row) - np.min(row))
            for j in range(len(train_data[0])):
                if is_provided[i, j]:
                    users.append(str(i))
                    items.append(str(j))
                    ratings.append(str(row[j] + 0.00001))

        df = pd.DataFrame(list(zip(users, items, ratings)), columns=["userID", "itemID", "rating"])
        df.to_csv(self.train_file)

        data = NCFDataset(train_file=self.train_file, test_file=None, seed=self.SEED, binary=False,
                          overwrite_test_file_full=False, n_neg_test=0, n_neg=0)

        # for user_input, item_input, labels in data.train_loader(self.BATCH_SIZE):
        #     user_input = np.array([data.user2id[x] for x in user_input])
        #     item_input = np.array([data.item2id[x] for x in item_input])
        #     labels = np.array(labels)
        #     print(user_input)
        #     print(item_input)
        #     print(labels)
        #     print("***************************")

        self.model = NCF(
            n_users=data.n_users,
            n_items=data.n_items,
            model_type="neumf",
            n_factors=self.n_factors,
            layer_sizes=[16, 8, 4],
            n_epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            learning_rate=self.learning_rate,
            verbose=1,
            seed=self.SEED
        )

        with Timer() as train_time:
            self.model.fit(data)

        print("Took {} seconds for training.".format(train_time))

    def predict(self, df_test):
        if df_test is not None:
            raise NotImplementedError("Please don't do train-test split :), I haven't tested it yet~")

        with Timer() as test_time:
            predictions = [[row, col, self.model.predict(row, col)]
                           for (row, col) in itertools.product(np.arange(self.num_users), np.arange(self.num_items))]

            predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
            predictions.to_csv(self.all_predictions_file)

        print("Took {} seconds for prediction.".format(test_time.interval))

        print("genarating submissions...")
        reconstructed = np.zeros((self.num_users, self.num_items))
        all_pred = pd.read_csv(self.all_predictions_file)
        for ind, row in all_pred.iterrows():
            reconstructed[int(row.userID), int(row.itemID)] = row.prediction

        reconstructed = postprocess(reconstructed, self.data_mean, self.data_std)
        generate_submission(self.sample_submission, self.save_file, reconstructed)


# if __name__ == '__main__':
#     df = _read_df_in_format("./data/data_train.csv")
#     model = NCF_model(None)
#     model.train(df)
#     model.predict(None)

# Model parameters
# EPOCHS = 25
# BATCH_SIZE = 128
# SEED = 42
# n_factors = 16
# learning_rate = 1e-3
# train_file = "./data/actual.csv"
# sample_submission = "./data/sampleSubmission.csv"
# save_file = "./submission_neural.csv"
# all_predictions_file = "./predictions_alle.csv"

# df = _read_df_in_format("./data/data_train.csv")
# train_data, is_provided = _convert_df_to_matrix(df, 10000, 1000)
# train_data, data_mean, data_std = preprocess(train_data, 10000, 1000, "zero")
#
# users = []
# items = []
# ratings = []
#
# for i in range(len(train_data)):
#     row = train_data[i]
#     # row = (row - np.min(row)) / (np.max(row) - np.min(row))
#     for j in range(len(train_data[0])):
#         if train_data[i, j] != 0:
#             users.append(str(i))
#             items.append(str(j))
#             ratings.append(str(row[j]+0.00001))
#
# df = pd.DataFrame(list(zip(users, items, ratings)), columns=["userID", "itemID", "rating"])
# df.to_csv(self.train_file)
#
#
# data = NCFDataset(train_file=self.train_file, test_file=None, seed=self.SEED, binary=False, overwrite_test_file_full=False, n_neg_test=0, n_neg=0)
#
# # for user_input, item_input, labels in data.train_loader(BATCH_SIZE):
# #     user_input = np.array([data.user2id[x] for x in user_input])
# #     item_input = np.array([data.item2id[x] for x in item_input])
# #     labels = np.array(labels)
# #     print(user_input)
# #     print(item_input)
# #     print(labels)
# #     print("***************************")
#
#
# model = NCF (
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="neumf",
#     n_factors=self.n_factors,
#     layer_sizes=[16,8,4],
#     n_epochs=self.EPOCHS,
#     batch_size=self.BATCH_SIZE,
#     learning_rate=self.learning_rate,
#     verbose=1,
#     seed=self.SEED
# )
#
# with Timer() as train_time:
#     model.fit(data)
#
#
# print("Took {} seconds for training.".format(train_time))

# with Timer() as test_time:
#     predictions = [[row, col, model.predict(row, col)]
#                    for (row, col) in itertools.product(np.arange(10000), np.arange(1000))]
#
#     predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
#     predictions.to_csv(self.all_predictions_file)
#
# print("Took {} seconds for prediction.".format(test_time.interval))
#
# print("genarating submissions...")
# reconstructed = np.zeros((10000, 1000))
# all_pred = pd.read_csv(self.all_predictions_file)
# for ind, row in all_pred.iterrows():
#     reconstructed[int(row.userID), int(row.itemID)] = row.prediction
#
# reconstructed = postprocess(reconstructed, data_mean, data_std)
# generate_submission(self.sample_submission, self.save_file, reconstructed)





# train = pd.read_csv(train_file)
#
# with Timer() as test_time:
#
#     users, items, preds = [], [], []
#     item = list(train.itemID.unique())
#     for user in train.userID.unique():
#         user = [user] * len(item)
#         users.extend(user)
#         items.extend(item)
#         preds.extend(list(model.predict(user, item, is_list=True)))
#
#     all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})
#     print(len(all_predictions))
#
#     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
#     print(len(merged))
#     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
#     print(len(all_predictions))
#
# print("Took {} seconds for prediction.".format(test_time.interval))
#
# all_predictions.to_csv("./predictions.csv")

#
# import os
# import sys
# # import scrapbook as sb
# from tempfile import TemporaryDirectory
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR') # only show error messages
#
# from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
# from recommenders.models.deeprec.models.xDeepFM import XDeepFMModel
# from recommenders.models.deeprec.io.iterator import FFMTextIterator
#
# print("System version: {}".format(sys.version))
# print("Tensorflow version: {}".format(tf.__version__))
#
#
#
# EPOCHS = 10
# BATCH_SIZE = 4096
# RANDOM_SEED = 42  # Set this to None for non-deterministic result
#
#
# data_path = './data/t'
# yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
# output_file = os.path.join(data_path, r'output.txt')
# train_file = os.path.join(data_path, r'cretio_tiny_train')
# valid_file = os.path.join(data_path, r'cretio_tiny_valid')
# test_file = os.path.join(data_path, r'cretio_tiny_test')
#
# if not os.path.exists(yaml_file):
#     download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')
#
# print('Demo with Criteo dataset')
# hparams = prepare_hparams(yaml_file,
#                           FEATURE_COUNT=2300000,
#                           FIELD_COUNT=39,
#                           cross_l2=0.01,
#                           embed_l2=0.01,
#                           layer_l2=0.01,
#                           learning_rate=0.002,
#                           batch_size=BATCH_SIZE,
#                           epochs=EPOCHS,
#                           cross_layer_sizes=[20, 10],
#                           init_value=0.1,
#                           layer_sizes=[20,20],
#                           use_Linear_part=True,
#                           use_CIN_part=True,
#                           use_DNN_part=True)
# print(hparams)
#
# model = XDeepFMModel(hparams, FFMTextIterator, seed=RANDOM_SEED)
#
# print(model.run_eval(test_file))
# model.fit(train_file, valid_file)
#
# result = model.run_eval(test_file)
# print(result)
