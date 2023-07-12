import numpy as np
from utils import _convert_df_to_matrix, _load_data_for_VAE, compute_rmse, generate_submission
from VAE import VAE
import torch
import torch.optim as optim

class VAE_model:
    def __init__(self, args, df_train, df_test):
        self.args = args
        self.df_train = df_train
        self.df_test = df_test
        self.num_iterations = args.vae_args.num_iterations
        self.model = VAE(args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.vae_args.lr, weight_decay=args.vae_args.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.vae_args.gamma)

    def train(self, df_train=None):
        print("Start training VAE model ...")
        data_train, _ = _convert_df_to_matrix(self.df_train, 10000, 1000)
        dataloader_train, data_train, mask_train, indices_train = _load_data_for_VAE(data_train, self.args.vae_args.batch_size)

        for epoch in range(self.num_iterations):
            self.model.train()
            for idx, (data_batch, mask_batch, indices_batch) in enumerate(dataloader_train):
                reconstructed_batch, mu, log_var, _ = self.model(data_batch, indices_batch)

                reconstruction_loss = ((data_batch - reconstructed_batch) ** 2 * mask_batch).sum(axis=1).mean()
                kl_loss = 0.5 * (torch.square(mu) + torch.exp(log_var) - 1 - log_var).sum(axis=1).mean()
                loss = reconstruction_loss + self.args.vae_args.beta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            if not self.args.generate_submissions:
                self.model.eval()
                validation_matrix = self.model(data_train, indices_train)[0].detach().numpy().clip(1, 5)
                predictions = validation_matrix[self.df_test['row'].values - 1, self.df_test['col'].values - 1]
                labels = self.df_test['Prediction'].values
                print('Epoch: {}/{}, RMSE: {:.4f}'.format(epoch, self.num_iterations, compute_rmse(predictions, labels)))
            else:
                print('Epoch: {}/{}'.format(epoch, self.num_iterations))

        self.reconstructed_matrix = self.model(data_train, indices_train)[0].detach().numpy()
        print("Training ends. ")

    def predict(self, df_test=None):
        if not self.args.generate_submissions:
            predictions = self.reconstructed_matrix[self.df_test['row'].values - 1, self.df_test['col'].values - 1]
            labels = self.df_test['Prediction'].values
            print('RMSE: {:.4f}'.format(compute_rmse(predictions, labels)))

        else:
            submission_file = self.args.submission_folder + "/vae.csv"
            generate_submission(self.args.sample_data, submission_file, self.reconstructed_matrix)

