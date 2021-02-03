For each model, its corresponding report can be found in ```docs```


# Lotka Volterra Model

A predictive model using Dynamic Sistems through the well-known Lotka Volterra equations.

## Running

- ```charts.py``` runs the simulation saving its results in the directory ```images```.

This can be run with:

```
python charts.py
```

**Note: Python 3.7 or higher is required**

# Integrate and Fire Model

A theorical neuroscience model to analyze the action potential that neurons fire when their membrane potential reaches a threshold value.

## Running

- ```charts.py``` runs the simulation saving its results in the directory ```images```.

This can be run with:

```
python charts.py
```

# Linear, Convolutional, and Variational Autoencoder

A comparison between 3 generative models using Neuronal Networks.

## Running

- ```autoencoder.py``` implements the autoencoder base class. This script can also be executed individually in order to train an autoencoder with a given  topology (see ```docs.pdf```).
- ```conv_vs_linear.py``` compares the average train and test losses between the convolutional and linear autoencoder.
- ```conv_latent_space.py``` plots the latent space of the convolutional autoencoder.
- ```linear_latent_space.py``` plots the latent space of the linear autoencoder
- ```vae_latent_space.py``` plots the latent space of the variational autoencoder
- ```run.py``` executes the ```autoencoder.py``` script for each of the parameters contained in the cartesian product of the following lists (The recorded data can be found in log files saved in a directory called ```runs```)-
```
    nsof_epochs = [8, 16]
    batch_sizes = [128, 256, 1000]
    optimizers = ['SGD', 'ADAM']
    learning_rates = [1e-1, 1e-2, 1e-3]
    momentums = [.2, .4, .6, .8]
```
Except for the ```autoencoder.py``` file, these scripts can be run individually as the ones described in the two previous models.


```autoencoder.py``` can be used to train an autoencoder with certain parameters as follows:
```
python autoencoder.py -epoch 2 -bs 64 -opt ADAM -lr 0.001 -momentum 0.8
```
