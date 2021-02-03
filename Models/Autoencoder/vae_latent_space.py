import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder, Trainer, MNISTDataLoader, Reshape


DIR_IMAGES = 'images'

def plot_latent(autoencoder, data, batch_size, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        z = z.reshape(batch_size, 2)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig(f'{DIR_IMAGES}/vae_latent_space.png')


def plot_reconstructed(autoencoder, r0=(5, 15), r1=(5, 15), n=12):
    w = 28
    img = np.zeros((n*w, n*w))

    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        self.relu = nn.ReLU()

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = Reshape([1, 28 * 28])(x)
        x = self.linear1(x)
        x = self.relu(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z


def main():
    nof_epochs = 8
    batch_size = 1000
    learning_rate = 1e-3
    loader = MNISTDataLoader(batch_size=batch_size)

    encoder = VariationalEncoder(2)
    decoder = nn.Sequential(
        nn.Linear(2, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        Reshape([1, 28, 28])
    )
    autoencoder = Autoencoder(encoder, decoder)
    criterion = lambda x, x_hat: ((x - x_hat)**2).sum() + autoencoder.encoder.kl

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=learning_rate,
    )

    Trainer(
        nof_epochs=nof_epochs,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        model=autoencoder
    ).run()

    plot_latent(autoencoder, loader.train, batch_size)

    plot_reconstructed(autoencoder)


if __name__ == "__main__":
    main()
