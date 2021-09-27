import numpy as np
import matplotlib.pyplot as plt

from deeprob.spn.structure.leaf import Gaussian
from deeprob.spn.learning.wrappers import learn_estimator
from deeprob.spn.algorithms.sampling import sample

from deeprob.spn.models.ratspn import GaussianRatSpn
from deeprob.flows.models.realnvp import RealNVP1d
from deeprob.flows.models.maf import MAF
from deeprob.torch.routines import train_model


def energy_domain(domain, resolution=256):
    left, right = domain
    x = np.linspace(left, right, num=resolution)
    y = np.linspace(left, right, num=resolution)
    x, y = np.meshgrid(x, y)
    return np.column_stack([x.flatten(), y.flatten()])


def w1(z):
    return np.sin(2.0 * np.pi * z[:, 0] / 4.0)


def w2(z):
    return 3.0 * np.exp(-0.5 * ((z[:, 0] - 1.0) / 0.6) ** 2)


def w3(z):
    return 3.0 / (1.0 + np.exp((1.0 - z[:, 0]) / 0.3))


def moons_pdf(z):
    u = 0.5 * ((np.linalg.norm(z, axis=1) - 2.0) / 0.4) ** 2
    v = np.exp(-0.5 * ((z[:, 0] - 2.0) / 0.6) ** 2)
    w = np.exp(-0.5 * ((z[:, 0] + 2.0) / 0.6) ** 2)
    p = np.exp(-u) * (v + w)
    return p / np.sum(p)


def waves_pdf(z):
    p = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    return p / np.sum(p)


def split_pdf(z):
    u = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    v = np.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    p = u + v
    return p / np.sum(p)


def nails_pdf(z):
    u = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    v = np.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    p = u + v
    return p / np.sum(p)


def generate_samples(pdf, domain, n_samples, noise=0.05):
    probs = pdf(domain)
    indices = np.arange(len(domain))
    idx = np.random.choice(indices, size=n_samples, p=probs)
    points = domain[idx] + np.random.randn(n_samples, 2) * noise
    return points.astype(np.float32)


def energy_plot_pdf(ax, pdf):
    ax.imshow(pdf, interpolation='bilinear', cmap='jet')
    ax.axis('off')


def energy_plot_samples(ax, domain, resolution, samples):
    left, right = domain
    hist, _, _ = np.histogram2d(
        samples[:, 1], samples[:, 0],
        bins=resolution,
        range=[[left, right], [left, right]],
        density=True
    )
    energy_plot_pdf(ax, hist)


def spn_sample_energy(data, n_samples):
    distributions = [Gaussian, Gaussian]
    spn = learn_estimator(
        data, distributions,
        learn_leaf='mle',
        split_rows='gmm',
        split_cols='random',
        min_rows_slice=512,
        split_rows_kwargs={'n': 8},
        verbose=False
    )
    nans = np.tile(np.nan, [n_samples, 2])
    return sample(spn, nans, inplace=True)


def rat_sample_energy(data, n_samples):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = GaussianRatSpn(
        in_features=2, optimize_scale=True,
        rg_depth=1, rg_repetitions=1, rg_batch=16, rg_sum=16
    )
    train_model(
        model, data_train, data_valid, setting='generative',
        lr=1e-3, batch_size=256, epochs=100, patience=1, verbose=False
    )
    model.eval()
    return model.sample(n_samples).cpu().numpy()


def nvp_sample_energy(data, n_samples):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = RealNVP1d(in_features=2, n_flows=10, units=128, batch_norm=False)
    train_model(
        model, data_train, data_valid, setting='generative',
        lr=1e-4, batch_size=256, epochs=100, patience=1, checkpoint='energy-checkpoint.pt', verbose=False
    )
    model.eval()
    return model.sample(n_samples).cpu().numpy()


def maf_sample_energy(data, n_samples):
    n_train = int(0.9 * len(data))
    data_valid = data[n_train:]
    data_train = data[:n_train]
    model = MAF(in_features=2, n_flows=10, units=128, batch_norm=False)
    train_model(
        model, data_train, data_valid, setting='generative',
        lr=1e-4, batch_size=256, epochs=100, patience=1, checkpoint='energy-checkpoint.pt', verbose=False
    )
    model.eval()
    return model.sample(n_samples).cpu().numpy()


if __name__ == '__main__':
    domain = (-4.0, 4.0)
    resolution = 128
    n_samples = 100_000
    pdfs_domain = energy_domain(domain, resolution)

    # The PDFs functions
    pdfs = {
        'moons': moons_pdf,
        'waves': waves_pdf,
        'split': split_pdf,
        'nails': nails_pdf
    }

    # Compute the grid-like shape pdf
    pdfs_grid = {k: p(pdfs_domain).reshape(resolution, resolution) for k, p in pdfs.items()}

    # Generate some data samples
    samples = {k: generate_samples(p, pdfs_domain, n_samples, noise=0.05) for k, p in pdfs.items()}

    # Initialize the result plot
    fig, axs = plt.subplots(figsize=(20, 16), ncols=5, nrows=len(samples), squeeze=False)

    # Learn a model for each density function
    for i, energy in enumerate(samples):
        print('Model: SPN - Energy: ' + energy)
        energy_plot_samples(axs[i, 0], domain, resolution, spn_sample_energy(samples[energy], n_samples))
        axs[0, 0].set_title('SPN', fontdict={'fontsize': 32})

        print('Model: RAT-SPN - Energy: ' + energy)
        energy_plot_samples(axs[i, 1], domain, resolution, rat_sample_energy(samples[energy], n_samples))
        axs[0, 1].set_title('RAT-SPN', fontdict={'fontsize': 32})

        print('Model: RealNVP - Energy: ' + energy)
        energy_plot_samples(axs[i, 2], domain, resolution, nvp_sample_energy(samples[energy], n_samples))
        axs[0, 2].set_title('RealNVP', fontdict={'fontsize': 32})

        print('Model: MAF - Energy: ' + energy)
        energy_plot_samples(axs[i, 3], domain, resolution, maf_sample_energy(samples[energy], n_samples))
        axs[0, 3].set_title('MAF', fontdict={'fontsize': 32})

    # Plot the true density functions
    for i, energy in enumerate(pdfs_grid):
        energy_plot_pdf(axs[i, -1], pdfs_grid[energy])
        axs[0, -1].set_title('True', fontdict={'fontsize': 32})

    # Save the results
    plt.tight_layout()
    plt.savefig('energy.png')
