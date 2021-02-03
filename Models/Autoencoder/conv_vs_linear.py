import torch
import torch.nn as nn
from autoencoder import Autoencoder, Trainer, MNISTDataLoader, Reshape

DIR_IMAGES = 'images'

def save_conv_vs_linear(conv_losses, linear_losses, save=True):
    _, (ax1, ax2) = plt.subplots(2, sharex=True)
    epochs = np.arange(1, len(conv_losses))
    train_conv_losses, test_conv_losses = conv_losses
    train_linear_losses, test_linear_losses = linear_losses

    ax1.set_ylabel('Error (ECM)', fontsize=10)
    ax2.set_ylabel('Error (ECM)', fontsize=10)
    ax2.set_xlabel('Epocas', fontsize=10)
    ax1.set_title('Losses - Entrenamiento')
    ax2.set_title('Losses - Testeo')

    ax1.plot(epochs, train_conv_avglosses, '-r', label='AEC')
    ax1.plot(epochs, test_linear_avglosses, '-b', label='AEL')
    ax2.plot(epochs, test_conv_losses, '-r', label='AEC')
    ax2.plot(epochs, test_linear_avgloss, '-b', label='AEL')

    ax1.grid()
    ax2.grid()

    plt.savefig(f'{DIR_IMAGES}/conv_vs_linear.png') if save else plt.show()
    plt.clf()


def main():
    nof_epochs = 16
    batch_size = 1000
    learning_rate = 1e-3
    loader = MNISTDataLoader(batch_size=batch_size)
    criterion = nn.MSELoss()

    encoder = nn.Sequential(
        nn.Conv2d(1, 16, 15),
        nn.ReLU(),
        nn.Conv2d(16, 32, 7),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5),
        nn.ReLU(),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(64, 32, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 7),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 15)
    )

    linear_encoder = nn.Sequential(
        Reshape(1, 28 * 28)
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
    )

    linear_decoder = nn.Sequential(
        nn.Linear(64, 28 * 28),
        nn.ReLU()
        Reshape(1, 28, 28)
    )

    conv_autoencoder = Autoencoder(conv_encoder, conv_decoder)
    linear_autoencoder = Autoencoder(linear_encoder, linear_decoder)

    optimizer = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    conv_losses = Trainer(
        nof_epochs=nof_epochs,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer(conv_autoencoder)
        model=conv_autoencoder
    ).run()

    linear_losses = Trainer(
        nof_epochs=nof_epochs,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer(linear_autoencoder),
        model=linear_autoencoder
    ).run()

    save_conv_vs_linear(conv_losses, linear_losses)


if __name__ == "__main__":
    main()
