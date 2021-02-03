import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder, Trainer, MNISTDataLoader


DIR_IMAGES = 'images/conv_latent_space'

def plot_latent(autoencoder, data, batch_size, batch_id, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x)
        z = z.detach().numpy()
        z = np.transpose(z, (1, 0, 2, 3))
        print(i)
        z = z[batch_id].reshape(batch_size, 2)
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches and batch_id == 0:
            plt.colorbar()
            break
        elif i > num_batches:
            break
    plt.savefig(f'{DIR_IMAGES}/filter_{batch_id}.png')


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


def main():
    nof_epochs = 16
    batch_size = 64
    learning_rate = 1e-3
    loader = MNISTDataLoader(batch_size=batch_size)
    criterion = nn.MSELoss()

    conv_encoder = nn.Sequential(
        nn.Conv2d(1, 16, 15),
        nn.ReLU(),
        nn.Conv2d(16, 32, 7),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, (2, 1)),
        nn.ReLU(),
    )

    conv_decoder = nn.Sequential(
        nn.ConvTranspose2d(64, 64, (2, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 64, 3),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 7),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 15)
    )

    autoencoder = Autoencoder(conv_encoder, conv_decoder)

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

    for i in range(64):
        plot_latent(autoencoder, loader.train, batch_size, i)

    plot_reconstructed(autoencoder)


if __name__ == "__main__":
    main()
