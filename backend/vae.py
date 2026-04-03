# backend/vae.py
# VAE = Variational Autoencoder
# Think of it as: learns the "shape" of the data,
# then helps us generate realistic alternative versions of a patient

import torch
import torch.nn as nn
import numpy as np

INPUT_DIM = 13   # 13 features
LATENT_DIM = 6   # compressed representation size


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder: takes 13 features → compresses to 6 numbers
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, LATENT_DIM)      # mean
        self.fc_logvar = nn.Linear(16, LATENT_DIM)  # variance

        # Decoder: takes 6 numbers → reconstructs 13 features
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, INPUT_DIM)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_loss


def train_vae(X_scaled, epochs=50):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training VAE...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, mu, logvar = model(X_tensor)
        loss = vae_loss(recon, X_tensor, mu, logvar)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'backend/models/vae.pt')
    print("VAE saved!")
    return model


def load_vae():
    model = VAE()
    model.load_state_dict(torch.load('backend/models/vae.pt',
                          map_location=torch.device('cpu')))
    model.eval()
    return model