import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import time
from torchvision.utils import save_image


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


def imsave(filename, img):
    npimg = img.numpy()
    plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))


class MNISTDataLoader():
    def __init__(self, batch_size, data_dir='./data'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transform
        )

        self.batch_size = batch_size
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform
        )

        self.test = DataLoader(testset, batch_size=batch_size, shuffle=True)

    @property
    def trainset_size(self):
        return len(self.train.dataset)

    @property
    def nof_train_batches(self):
        return len(self.train)

    @property
    def testset_size(self):
        return len(self.test.dataset)

    @property
    def nof_test_batches(self):
        return len(self.test)

    def plot_loader(self, nof_samples, filename='MNIST_DIGITS.png', save=True):
        width = 6
        height = 10
        color_map = 'gray_r'

        dataiter = iter(self.train)
        images, _ = dataiter.next()
        _ = plt.figure()
        for index in range(1, nof_samples + 1):
            plt.subplot(width, height, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap=color_map)
        plt.savefig(filename) if save else plt.show()


class Trainer():
    def __init__(self, nof_epochs, loader, model, criterion, optimizer):
        self.nof_epochs = nof_epochs
        self.loader = loader
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer

    def __print_trainlog(self, epoch, batch_id, loss_batch):
        log = f'[TRAIN]\t' \
              f'Train Epoch: {epoch}' \
              f'[{batch_id * self.loader.batch_size}/{self.loader.trainset_size}' \
              f'({100. * batch_id / self.loader.nof_train_batches:.0f}%)]' \
              f'\tLoss: {loss_batch.item():.6f}'
        print(log)

    @staticmethod
    def __print_epochlog(epoch, train_avgloss, test_avgloss, epoch_time):
        log = f'[EPOCH]\t' \
              f'epoch={epoch}\t' \
              f'train_avgloss={train_avgloss:.6f}\t' \
              f'test_avgloss={test_avgloss:.6f}\t' \
              f'epoch_time={epoch_time:.6f}\n'
        print(log)

    def train(self, epoch):
        self.model.train()
        train_avgloss = 0
        for batch_id, (batch_input, _) in enumerate(self.loader.train, 0):
            self.optimizer.zero_grad()
            batch_output = self.model(batch_input)
            loss_batch = self.criterion(batch_output, batch_input)
            loss_batch.backward()
            self.optimizer.step()
            train_avgloss += loss_batch
            self.__print_trainlog(epoch, batch_id, loss_batch)

        train_avgloss /= self.loader.nof_train_batches
        return train_avgloss.item()

    def test(self):
        self.model.eval()
        test_avgloss = 0
        with torch.no_grad():
            for batch_input, _ in self.loader.test:
                batch_output = self.model(batch_input)
                test_avgloss += self.criterion(batch_output, batch_input).item()

        test_avgloss /= self.loader.nof_test_batches
        return test_avgloss

    def run(self):
        test_time = 0
        train_avglosses = []
        test_avglosses = []
        for epoch in range(1, self.nof_epochs + 1):
            initial_time = time()
            train_avgloss = self.train(epoch)
            test_avgloss = self.test()
            epoch_time = time() - initial_time
            test_time += epoch_time
            train_avglosses.append(train_avglosses)
            test_avglosses.append(test_avglosses)
            self.__print_epochlog(epoch, train_avgloss, test_avgloss, epoch_time)
        return (train_avglosses, test_avglosses)

        print(f'[FINISH]\tTraining Finished in {test_time:.6f}')

    def sample_net_output(self, epoch, nof_samples):
        sample, _ = iter(self.loader.test).next()
        output = self.model(sample)
        sample = sample[0:nof_samples]
        output = output[0:nof_samples]
        both = torch.cat((sample, output)).detach()
        imsave(
            'net_output.png',
            torchvision.utils.make_grid(both, normalize=True, range=(0, 1))
        )

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, batch):
        return batch.view(batch.size(0), *self.shape)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, help='Number of epochs', required=True)
    parser.add_argument('-bs', type=int, help='Batch size', required=True)
    parser.add_argument('-opt', choices=['ADAM', 'SGD'], help='Optimizer', required=True)
    parser.add_argument('-lr', type=float, help='Learning rate', required=True)
    parser.add_argument('-m', type=float, help='Momentum', required=True)
    args = parser.parse_args()

    nof_epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr
    momentum = args.m
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

    model = Autoencoder(encoder, decoder)

    ADAMoptimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    SGDoptimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    optimizer = SGDoptimizer if args.opt == 'SGD' else ADAMoptimizer
    trainer = Trainer(
        nof_epochs=nof_epochs,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        model=model
    )
    trainer.run()


if __name__ == "__main__":
    main()

# python3 autoencoder -epoch 2 -bs 64 -opt ADAM -lr 0.001