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
            cb = plt.colorbar()
            break
    plt.savefig(f'{DIR_IMAGES}/linear_latent_space.png')


def plot_reconstructed(autoencoder, r0=(2, 10), r1=(2, 10), n=12):
    w = 28
    img = np.zeros((n*w, n*w))

    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]])
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig(f'{DIR_IMAGES}/linear_latent_digits.png')


def main():
    nof_epochs = 16
    batch_size = 128
    learning_rate = 1e-3
    momentum = .8
    loader = MNISTDataLoader(batch_size=batch_size)
    criterion = lambda x_hat, x: ((x - x_hat)**2).sum()

    encoder = nn.Sequential(
        Reshape([1, 28 * 28]),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 2),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.Linear(2, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.ReLU(),
        Reshape([1, 28, 28])
    )

    autoencoder = Autoencoder(encoder, decoder)

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
