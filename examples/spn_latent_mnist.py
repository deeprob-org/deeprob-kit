import numpy as np
import sklearn as sk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

import deeprob.utils as utils
import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg
import deeprob.spn.utils as spnutils
from deeprob.spn.learning.wrappers import learn_classifier

if __name__ == '__main__':
    # Setup the MNIST datasets
    n_classes = 10
    n_features = (image_c, image_h, image_w) = (1, 28, 28)
    n_dimensions = np.prod(n_features).item()
    transform = transforms.ToTensor()
    data_train = datasets.MNIST('datasets', train=True, transform=transform, download=True)
    data_test = datasets.MNIST('datasets', train=False, transform=transform, download=True)

    # Build the autoencoder for features extraction
    latent_dim = 24  # Use 24 features as latent space
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_dimensions, 512), nn.ReLU(inplace=True),
        nn.Linear(512, 256), nn.ReLU(inplace=True),
        nn.Linear(256, latent_dim), nn.Tanhshrink(),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
        nn.Linear(256, 512), nn.ReLU(inplace=True),
        nn.Linear(512, 784), nn.Sigmoid(),
        nn.Unflatten(1, n_features)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = nn.Sequential(encoder, decoder).to(device)

    # Train the autoencoder, by minimizing the reconstruction binary cross-entropy
    epochs = 25
    batch_size = 100
    lr = 1e-3
    train_loader = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.BCELoss()
    tk_epochs = tqdm(range(epochs), bar_format='{l_bar}{bar:24}{r_bar}', unit='epoch')
    for epoch in tk_epochs:
        train_loss = 0.0
        for (inputs, _) in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.shape[0]
        train_loss /= len(train_loader)
        tk_epochs.set_description('Train Loss: {}'.format(round(train_loss, 4)))

    # Compute the (train data) latent space features using the encoder
    train_loader = data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    x_train = np.empty([len(data_train), latent_dim], dtype=np.float32)
    y_train = np.empty(len(data_train), dtype=np.int64)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            outputs = encoder(inputs).cpu()
            x_train[i * batch_size:i * batch_size + batch_size] = outputs.numpy()
            y_train[i * batch_size:i * batch_size + batch_size] = targets.numpy()

    # Compute the (test data) latent space features using the encoder
    test_loader = data.DataLoader(data_test, batch_size=batch_size, shuffle=False)
    x_test = np.empty([len(data_test), latent_dim], dtype=np.float32)
    y_test = np.empty(len(data_test), dtype=np.int64)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = encoder(inputs).cpu()
            x_test[i * batch_size:i * batch_size + batch_size] = outputs.numpy()
            y_test[i * batch_size:i * batch_size + batch_size] = targets.numpy()

    # Preprocess the datasets using standardization
    transform = utils.DataStandardizer()
    transform.fit(x_train)
    x_train = transform.forward(x_train)
    x_test = transform.forward(x_test)

    # Learn the SPN structure and parameters, as a classifier
    # Note that we consider the train data as features + targets
    distributions = [spn.Gaussian] * latent_dim + [spn.Categorical]
    data_train = np.column_stack([x_train, y_train])
    root = learn_classifier(
        data_train,
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='kmeans',  # Use K-Means for splitting rows
        split_cols='rdc',     # Use RDC for splitting columns
        min_rows_slice=200,   # The minimum number of rows required to split furthermore
        split_rows_kwargs={'n': 2},   # Use n=2 number of clusters for K-Means
        split_cols_kwargs={'d': 0.3}  # Use d=0.3 as threshold for RDC independence test
    )

    # Print some statistics about the model's structure and parameters
    print("SPN structure and parameters statistics:")
    print(spnutils.compute_statistics(root))

    # Save the model to a JSON file
    spn_filename = 'spn-latent-mnist.json'
    print("Saving the SPN structure and parameters to {} ...".format(spn_filename))
    spn.save_spn_json(root, spn_filename)

    # Make some predictions on the test set classes
    # This is done by running a Maximum Probable Explanation (MPE) query
    nan_classes = np.full([len(x_test), 1], np.nan)
    data_test = np.column_stack([x_test, nan_classes])
    spnalg.mpe(root, data_test, inplace=True)
    y_pred = data_test[:, -1]

    # Plot a classification report
    print("Classification Report:")
    print(sk.metrics.classification_report(y_test, y_pred))

    # Sample some examples for each class
    # This is done by conditional sampling w.r.t. the example classes
    n_samples = 10
    nan_features = np.full([n_samples * n_classes, latent_dim], np.nan)
    classes = np.tile(np.arange(n_classes), [1, n_samples]).T
    samples = np.column_stack([nan_features, classes])
    spnalg.sample(root, samples, inplace=True)
    features = samples[:, :-1]

    # Apply the inverse preprocessing transformation
    # Then apply the features extractor's decoder and plot the examples on a grid
    with torch.no_grad():
        images = transform.backward(features)
        inputs = torch.tensor(images, dtype=torch.float32, device=device)
        data_images = decoder(inputs).cpu()
        samples_filename = 'spn-latent-mnist-samples.png'
        print("Plotting generated samples to {} ...".format(samples_filename))
        torchvision.utils.save_image(data_images, samples_filename, nrow=n_samples, padding=0)
