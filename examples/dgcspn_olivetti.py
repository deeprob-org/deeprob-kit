import numpy as np
import torch
import torchvision.utils as utils

from sklearn.datasets import fetch_openml

import deeprob.spn.models as spn
from deeprob.utils.statistics import compute_mean_quantiles
from deeprob.torch.datasets import SupervisedDataset, WrappedDataset
from deeprob.torch.transforms import Normalize
from deeprob.torch.routines import train_model, test_model


def load_olivetti_faces_datasest():
    # Fetch the dataset
    data, targets = fetch_openml(data_id=41083, data_home='datasets', return_X_y=True, as_frame=True)
    data, targets = data.to_numpy().astype(np.float32), targets.to_numpy().astype(np.int64)
    data = 255.0 * data.reshape([400, 1, 64, 64])
    data_train, data_test = data[:350], data[350:]
    targets_train, targets_test = targets[:350], targets[350:]

    # Compute mean and standard deviation
    data_mean, data_std = np.mean(data_train).item(), np.std(data_train).item()

    # Instantiate the supervised datasets
    data_train = SupervisedDataset(data_train, targets_train)
    data_test = SupervisedDataset(data_test, targets_test)

    # Wrap the supervised datasets for unsupervised setting
    mean = torch.tensor(data_mean, dtype=torch.float32)
    std = torch.tensor(data_std, dtype=torch.float32)
    transform = Normalize(mean, std)
    data_train = WrappedDataset(data_train, unsupervised=True, transform=transform)
    data_test = WrappedDataset(data_test, unsupervised=True, transform=transform)
    return data_train, data_test


if __name__ == '__main__':
    # Load the Olivetti-Faces dataset
    in_shape = (1, 64, 64)
    data_train, data_test = load_olivetti_faces_datasest()

    # Compute mean quantiles for leaf distributions initialization
    # This will initialize the location of the batched gaussian distributions with the mean of quantiles bins
    n_batch = 8
    preproc_data = np.asarray([x.numpy() for x in data_train])
    quantiles_loc = compute_mean_quantiles(preproc_data, n_batch)

    # Instantiate the model
    dgcspn = spn.DgcSpn(
        in_shape,
        n_batch=n_batch,             # The number of batch distributions at leaves
        sum_channels=2,              # The number of sum channels for spatial sum layers
        depthwise=[True, False],     # Only the first product layer uses depthwise convolution
        quantiles_loc=quantiles_loc  # Use mean quantiles as leaves locations initializer
    )

    # Train the model using generative setting, i.e. by maximizing the log-likelihood
    train_model(
        dgcspn, data_train, data_test, setting='generative',
        lr=1e-2, batch_size=64, epochs=1, checkpoint='checkpoint-dgcspn-olivetti.pt', verbose=False
    )

    # Test the model using generative setting
    mu_ll, sigma_ll = test_model(dgcspn, data_test, setting='generative', batch_size=64)
    print('Mean LL: {:.4f} - Two Stddev LL: {:.4f}'.format(mu_ll, sigma_ll))

    # Compute image tensors with some missing data patterns
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dgcspn.to(device)
    samples = torch.stack([data_test[i] for i in range(len(data_test))]).to(device)
    uncomplete_samples = torch.clone(samples)
    uncomplete_samples[:, :, :, :in_shape[2] // 2] = np.nan

    # Complete the images by most probable explanation (MPE) query
    dgcspn.eval()  # Make sure to switch to evaluation mode
    complete_samples = dgcspn.mpe(uncomplete_samples)
    complete_images = torch.stack([data_train.transform.backward(x) for x in complete_samples.cpu()])
    images = torch.stack([data_train.transform.backward(x) for x in samples.cpu()])

    # Compute the mean squared (completion) error (MSE)
    # Note the multiplication by two, because we only consider the error on the completed part
    completion_sqerr = (complete_images.long() - images.long()) ** 2.0
    completion_mse = 2.0 * torch.mean(completion_sqerr).item()
    print('Completion MSE: {:.2f}'.format(completion_mse))

    # Save the image completions
    nrow = 10
    images = torch.cat([images, complete_images])
    images = images.reshape(2, -1, nrow, *in_shape)
    images = images.permute(1, 0, 2, 3, 4, 5)
    images = images.reshape(len(data_test) * 2, *in_shape)
    samples_filename = 'dgcspn-olivetti-completions.png'
    print("Plotting generated samples to {} ...".format(samples_filename))
    utils.save_image(images / 255.0, samples_filename, nrow=nrow, padding=0)

    # Save the model to file
    model_filename = 'dgcspn-olivetti.pt'
    print("Saving model's definition and parameters to {}".format(model_filename))
    torch.save(dgcspn.state_dict(), model_filename)
