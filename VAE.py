import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.vae_args.hidden_dim
        self.latent_dim = args.vae_args.latent_dim
        self.dropout = args.vae_args.dropout
        self.bias = torch.nn.Parameter(torch.ones((10000, 1)) * 3)
        self.encoder_net = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(1000, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim*2),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1000)
        )

    def encoder(self, data):
        latent = self.encoder_net(data)
        mu, log_var = latent[:, :self.latent_dim], latent[:, self.latent_dim:]
        return mu, log_var

    def reparameterization(self, mu, log_var):
        if self.training:
            epsilon = torch.randn_like(log_var)
            z = mu + epsilon * torch.sqrt(log_var.exp())
            return z
        else:
            return mu

    def decoder(self, z):
        return self.decoder_net(z)

    def forward(self, data, indices):
        normalized_data = torch.nn.functional.normalize(data)
        mu, log_var = self.encoder(normalized_data)
        z = self.reparameterization(mu, log_var)
        reconstructed_data = self.decoder(z) + self.bias[indices].tile((1, 1000))
        return reconstructed_data, mu, log_var, z

