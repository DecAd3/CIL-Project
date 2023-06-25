import argparse
import pickle
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

import myfm
from myfm import MyFMOrderedProbit, MyFMRegressor, RelationBlock
from myfm.gibbs import MyFMOrderedProbit
from myfm.utils.callbacks.libfm import (
    LibFMLikeCallbackBase,
    OrderedProbitCallback,
    RegressionCallback,
)
from myfm.utils.encoders import CategoryValueToSparseEncoder

from utils import _load_data_for_BFM, _read_df_in_format, _convert_df_to_matrix, preprocess, postprocess, compute_rmse, generate_submission

class BFM_model:
    def __init__(self, args):
        # no fold index
        self.algorithm = args.bfm_args.algorithm    # ["regression", "oprobit"]
        self.iteration = args.bfm_args.iteration    # mcmc iterations
        self.dimension = args.bfm_args.dimension    # fm embedding dimension
        self.use_iu = args.bfm_args.use_iu          # Additional features."all users who have evaluated a movie in the train set"
        self.use_ii = args.bfm_args.use_ii          # Additional features."all movies rated by a user" as a feature of user/movie.
        self.seed_value = args.random_seed
        self.min_rate = args.min_rate
        self.max_rate = args.max_rate
        self.generate_submissions = args.generate_submissions
        self.sample_data = args.sample_data
        self.submission_folder = args.submission_folder

    def train(self, df_train):
        self.df_train = df_train

    def predict(self, df_test):
        np.random.seed(self.seed_value)
        df_train = self.df_train
        df_train = _load_data_for_BFM(df_train)
        if self.generate_submissions:
            df_test = _read_df_in_format(self.sample_data)
        df_test = _load_data_for_BFM(df_test)

        if self.algorithm == "oprobit":
            # interpret the rating (1, 2, 3, 4, 5) as class (0, 1, 2, 3, 4).
            for df_ in [df_train, df_test]:
                df_["rating"] -= 1
                df_["rating"] = df_.rating.astype(np.int32)

        implicit_data_source = df_train
        user_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.user_id.values
        )
        movie_to_internal = CategoryValueToSparseEncoder[int](
            implicit_data_source.movie_id.values
        )

        print(
            "df_train.shape = {}, df_test.shape = {}".format(df_train.shape, df_test.shape)
        )

        movie_vs_watched: Dict[int, List[int]] = dict()
        user_vs_watched: Dict[int, List[int]] = dict()

        for row in implicit_data_source.itertuples():
            user_id = row.user_id
            movie_id = row.movie_id
            movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            user_vs_watched.setdefault(user_id, list()).append(movie_id)

        # setup grouping
        feature_group_sizes = []

        feature_group_sizes.append(len(user_to_internal))  # user ids

        if self.use_iu:
            # all movies which a user watched
            feature_group_sizes.append(len(movie_to_internal))

        feature_group_sizes.append(len(movie_to_internal))  # movie ids

        if self.use_ii:
            feature_group_sizes.append(
                len(user_to_internal)  # all the users who watched a movies
            )

        grouping = [i for i, size in enumerate(feature_group_sizes) for _ in range(size)]

        def augment_user_id(user_ids: List[int]) -> sps.csr_matrix:
            X = user_to_internal.to_sparse(user_ids)
            if not self.use_iu:
                return X
            data: List[float] = []
            row: List[int] = []
            col: List[int] = []
            for index, user_id in enumerate(user_ids):
                watched_movies = user_vs_watched.get(user_id, [])
                normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                for mid in watched_movies:
                    data.append(normalizer)
                    col.append(movie_to_internal[mid])
                    row.append(index)
            return sps.hstack(
                [
                    X,
                    sps.csr_matrix(
                        (data, (row, col)),
                        shape=(len(user_ids), len(movie_to_internal)),
                    ),
                ],
                format="csr",
            )

        def augment_movie_id(movie_ids: List[int]):
            X = movie_to_internal.to_sparse(movie_ids)
            if not self.use_ii:
                return X

            data: List[float] = []
            row: List[int] = []
            col: List[int] = []

            for index, movie_id in enumerate(movie_ids):
                watched_users = movie_vs_watched.get(movie_id, [])
                normalizer = 1 / max(len(watched_users), 1) ** 0.5
                for uid in watched_users:
                    data.append(normalizer)
                    row.append(index)
                    col.append(user_to_internal[uid])
            return sps.hstack(
                [
                    X,
                    sps.csr_matrix(
                        (data, (row, col)),
                        shape=(len(movie_ids), len(user_to_internal)),
                    ),
                ]
            )

        # Create RelationBlock.
        train_blocks: List[RelationBlock] = []
        test_blocks: List[RelationBlock] = []
        for source, target in [(df_train, train_blocks), (df_test, test_blocks)]:
            unique_users, user_map = np.unique(source.user_id, return_inverse=True)
            target.append(RelationBlock(user_map, augment_user_id(unique_users)))
            unique_movies, movie_map = np.unique(source.movie_id, return_inverse=True)
            target.append(RelationBlock(movie_map, augment_movie_id(unique_movies)))

        trace_path = "./output/bfm/rmse_{0}.csv".format(self.algorithm)

        callback: LibFMLikeCallbackBase
        fm: Union[MyFMRegressor, MyFMOrderedProbit]
        if self.algorithm == "regression":
            fm = myfm.MyFMRegressor(rank=self.dimension)
            callback = RegressionCallback(
                self.iteration,
                None,
                df_test.rating.values,
                X_rel_test=test_blocks,
                clip_min=self.min_rate,
                clip_max=self.max_rate,
                trace_path=trace_path,
            )
        else:
            fm = myfm.MyFMOrderedProbit(rank=self.dimension)
            callback = OrderedProbitCallback(
                self.iteration,
                None,
                df_test.rating.values,
                n_class=5,
                X_rel_test=test_blocks,
                trace_path=trace_path,
            )

        fm.fit(
            None,
            df_train.rating.values,
            X_rel=train_blocks,
            grouping=grouping,
            n_iter=callback.n_iter,
            callback=callback,
            n_kept_samples=1,
        )
        
        # with open(
        #     "./output/bfm/callback_result_{0}.pkl".format(self.algorithm), "wb"
        # ) as ofs:
        #     pickle.dump(callback, ofs)

        if self.generate_submissions:
            result = fm.predict(None, X_rel = test_blocks).clip(self.min_rate, self.max_rate)
            df = pd.read_csv(self.sample_data)
            df['Prediction'] = result
            submission_file = self.submission_folder + '/bfm.csv'
            df.to_csv(submission_file, index=False)