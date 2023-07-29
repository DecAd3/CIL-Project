# part of the code is derived from the tutorial of Microsoft recommenders. https://github.com/microsoft/recommenders

import itertools
import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

from recommenders.utils.timer import Timer
from recommenders.ncf_singlenode import NCF
from recommenders.dataset import Dataset as NCFDataset
from utils import _read_df_in_format, _convert_df_to_matrix, preprocess, postprocess, generate_submission, compute_rmse


class NCF_model:
    def __init__(self, args):
        self.EPOCHS = args.ncf_args.EPOCHS
        self.BATCH_SIZE = args.ncf_args.BATCH_SIZE
        self.SEED = args.random_seed
        self.n_factors = args.ncf_args.n_factors
        self.learning_rate = args.ncf_args.learning_rate
        self.train_file = args.ncf_args.train_file
        self.sample_submission = args.sample_data
        self.save_file = args.submission_folder
        self.model = None
        self.reconstructed = None
        self.data_mean = 0
        self.data_std = 0
        self.num_users = args.num_users
        self.num_items = args.num_items
        self.generate_submissions = args.generate_submissions


    def train(self, df_train):
        # preprocess data
        train_data, is_provided = _convert_df_to_matrix(df_train, self.num_users, self.num_items)
        train_data, self.data_mean, self.data_std = preprocess(train_data, self.num_users, self.num_items, "zero")

        # convert data structure for input into NCF
        users = []
        items = []
        ratings = []
        for i in range(len(train_data)):
            row = train_data[i]
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

        # Create NCF model
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

        # Train model
        with Timer() as train_time:
            self.model.fit(data)

        print("Took {} seconds for training.".format(train_time))

    def predict(self, df_test, pred_file_name=None):
        print("Predicting...")
        with Timer() as test_time:
            predictions = [[row, col, self.model.predict(row, col)]
                           for (row, col) in itertools.product(np.arange(self.num_users), np.arange(self.num_items))]

            predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])

        print("Took {} seconds for prediction.".format(test_time.interval))

        print("Reconstructing matrix...")
        self.reconstructed = np.zeros((self.num_users, self.num_items))
        all_pred = predictions.copy(deep=True)
        for ind, row in all_pred.iterrows():
            self.reconstructed[int(row.userID), int(row.itemID)] = row.prediction

        self.reconstructed = postprocess(self.reconstructed, self.data_mean, self.data_std)

        if not self.generate_submissions:
            print("Computing test loss...")
            predictions = self.reconstructed[df_test['row'].values - 1, df_test['col'].values - 1]
            labels = df_test['Prediction'].values
            print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))
            return predictions
        else:
            print("Genarating submissions...")
            generate_submission(self.sample_submission, self.save_file + "/ncf_submit.csv", self.reconstructed)

        return None
