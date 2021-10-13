## Code Examples

A collection of code examples can be found in the `examples` directory.
In order to run the code examples, it is necessary clone the repository.
However, additional datasets are not required.
Note that the given examples are not intended to produce state-of-the-art results,
but only to present the library.

The following table contains a description about them and a code complexity ranging from one to three stars.
The *Complexity* column consists of a measure that roughly represents how many features of the library are used, as well as
the expected time required to run the script.

|        Example       |                                    Description                                    | Complexity |
|----------------------|-----------------------------------------------------------------------------------|:----------:|
| naive_model.py       | Learn, evaluate and print statistics about a naive factorized model.              |      ⭐     |      
| spn_plot.py          | Instantiate, prune, marginalize and plot some SPNs.                               |      ⭐     |
| clt_plot.py          | Learn a Binary CLT and plot it.                                                   |      ⭐     |
| spn_moments.py       | Instantiate and compute moments statistics about the random variables.            |      ⭐     |
| sklearn_interface.py | Learn and evaluate a SPN using the scikit-learn interface.                        |      ⭐     |
| spn_custom_leaf.py   | Learn, evaluate and serialize a SPN with a user-defined leaf distribution.        |      ⭐     |
| clt_to_spn.py        | Learn a Binary CLT, convert it to a structured decomposable SPN and plot it.      |      ⭐     |
| spn_clt_em.py        | Instantiate a SPN with Binary CLTs, apply EM algorithm and sample some data.      |     ⭐⭐     |
| clt_queries.py       | Learn a Binary CLT, plot it, run some queries and sample some data.               |     ⭐⭐     |
| ratspn_mnist.py      | Train and evaluate a RAT-SPN on MNIST.                                            |     ⭐⭐     |
| dgcspn_olivetti.py   | Train, evaluate and complete some images with DGC-SPN on Olivetti-Faces.          |     ⭐⭐     |
| dgcspn_mnist.py      | Train and evaluate a DGC-SPN on MNIST.                                            |     ⭐⭐     |
| nvp1d_moons.py       | Train and evaluate a 1D RealNVP on Moons dataset.                                 |     ⭐⭐     |
| maf_cifar10.py       | Train and evaluate a MAF on CIFAR10.                                              |     ⭐⭐⭐    |
| nvp2d_mnist.py       | Train and evaluate a 2D RealNVP on MNIST.                                         |     ⭐⭐⭐    |
| nvp2d_cifar10.py     | Train and evaluate a 2D RealNVP on CIFAR10.                                       |     ⭐⭐⭐    |
| spn_latent_mnist.py  | Train and evaluate a SPN on MNIST using the features extracted by an autoencoder. |     ⭐⭐⭐    |
