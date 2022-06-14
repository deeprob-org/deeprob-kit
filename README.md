[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![PyPI version](https://badge.fury.io/py/deeprob-kit.svg)](https://badge.fury.io/py/deeprob-kit)
[![codecov](https://codecov.io/gh/deeprob-org/deeprob-kit/branch/main/graph/badge.svg?token=4ZDC22QYEJ)](https://codecov.io/gh/deeprob-org/deeprob-kit)
[![Pytest-Coverage](https://github.com/deeprob-org/deeprob-kit/actions/workflows/pytest-coverage.yml/badge.svg)](https://github.com/deeprob-org/deeprob-kit/actions/workflows/pytest-coverage.yml)
![Pylint-Report](https://github.com/deeprob-org/deeprob-kit/actions/workflows/pylint-report.yml/badge.svg)

![Logo](docs/source/deeprob-logo.svg)

# DeeProb-kit

**DeeProb-kit** is a general-purpose Python library providing a collection of deep probabilistic models (DPMs) which
are easy to use and extend.
It also includes efficiently implemented learning techniques, inference routines and statistical algorithms.
The availability of a representative selection of the most common DPMs in a single library makes it possible to combine
them in a straightforward manner, a common practice in deep learning research nowadays, which however is still missing
for certain class of models. 
Moreover, **DeeProb-kit** provides high-quality fully-documented APIs, and it will help the community to accelerate research
on DPMs as well as improve experiments' reproducibility.

## Features

- Inference algorithms for SPNs. [^1] [^4]
- Learning algorithms for SPNs structure. [^1] [^2] [^3] [^4] [^5]
- Chow-Liu Trees (CLT) as SPN leaves. [^12] [^13]
- Batch Expectation-Maximization (EM) for SPNs with arbitrarily leaves. [^14] [^15]
- Structural marginalization and pruning algorithms for SPNs.
- High-order moments computation for SPNs.
- JSON I/O operations for SPNs and CLTs. [^4]
- Plotting operations based on NetworkX for SPNs and CLTs. [^4]
- Randomized And Tensorized SPNs (RAT-SPNs). [^6]
- Deep Generalized Convolutional SPNs (DGC-SPNs). [^11]
- Masked Autoregressive Flows (MAFs). [^7]
- Real Non-Volume-Preserving (RealNVP) flows. [^8]
- Non-linear Independent Component Estimation (NICE) flows. [^9]

The collection of implemented models is summarized in the following table.

| Model      | Description                                        |
|------------|----------------------------------------------------|
| Binary-CLT | Binary Chow-Liu Tree (CLT)                         |
| SPN        | Vanilla Sum-Product Network                        |
| MSPN       | Mixed Sum-Product Network                          |
| XPC        | Random Probabilistic Circuit                       |
| RAT-SPN    | Randomized and Tensorized Sum-Product Network      |
| DGC-SPN    | Deep Generalized Convolutional Sum-Product Network |
| MAF        | Masked Autoregressive Flow                         |
| NICE       | Non-linear Independent Components Estimation Flow  |
| RealNVP    | Real-valued Non-Volume-Preserving Flow             |

## Installation

The library can be installed either from PIP repository or by source code.
```shell
# Install from PIP repository
pip install deeprob-kit
```
```shell
# Install from `main` git branch
pip install -e git+https://github.com/deeprob-org/deeprob-kit.git@main#egg=deeprob-kit
```

## Project Directories

The documentation is generated automatically by Sphinx using sources stored in the [docs](docs) directory.

A collection of code examples and experiments can be found in the [examples](examples) and [experiments](experiments)
directories respectively.
Moreover, benchmark code can be found in the [benchmark](benchmark) directory.

## Related Repositories

- [SPFlow](https://github.com/SPFlow/SPFlow)
- [RAT-SPN](https://github.com/cambridge-mlg/RAT-SPN)
- [Random-PC](https://github.com/gengala/Random-Probabilistic-Circuits)
- [LibSPN-Keras](https://github.com/pronobis/libspn-keras)
- [MAF](https://github.com/gpapamak/maf)
- [RealNVP](https://github.com/chrischute/real-nvp)

## References

[^1]: Peharz et al. [*On Theoretical Properties of Sum-Product Networks*](http://proceedings.mlr.press/v38/peharz15.pdf). AISTATS (2015).
[^2]: Poon and Domingos. [*Sum-Product Networks: A New Deep Architecture*](https://arxiv.org/pdf/1202.3732.pdf). UAI (2011).
[^3]: Molina, Vergari et al. [*Mixed Sum-Product Networks: A Deep Architecture for Hybrid Domains*](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16865/16619). AAAI (2018).
[^4]: Molina, Vergari et al. [*SPFLOW : An easy and extensible library for deep probabilistic learning using Sum-Product Networks*](https://arxiv.org/pdf/1901.03704.pdf). CoRR (2019).
[^5]: Di Mauro et al. [*Sum-Product Network structure learning by efficient product nodes discovery*](http://www.di.uniba.it/~ndm/pubs/dimauro18ia.pdf). AIxIA (2018).
[^6]: Peharz et al. [*Probabilistic Deep Learning using Random Sum-Product Networks*](http://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf). UAI (2020). 
[^7]: Papamakarios et al. [*Masked Autoregressive Flow for Density Estimation*](https://proceedings.neurips.cc/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf). NeurIPS (2017).
[^8]: Dinh et al. [*Density Estimation using RealNVP*](https://arxiv.org/pdf/1605.08803v3.pdf). ICLR (2017).
[^9]: Dinh et al. [*NICE: Non-linear Independent Components Estimation*](https://arxiv.org/pdf/1410.8516.pdf). ICLR (2015).
[^10]: Papamakarios, Nalisnick et al. [*Normalizing Flows for Probabilistic Modeling and Inference*](https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf). JMLR (2021).
[^11]: Van de Wolfshaar and Pronobis. [*Deep Generalized Convolutional Sum-Product Networks for Probabilistic Image Representations*](http://proceedings.mlr.press/v138/wolfshaar20a/wolfshaar20a.pdf). PGM (2020).
[^12]: Rahman et al. [*Cutset Networks: A Simple, Tractable, and Scalable Approach for Improving the Accuracy of Chow-Liu Trees*](https://link.springer.com/content/pdf/10.1007%2F978-3-662-44851-9_40.pdf). ECML-PKDD (2014).
[^13]: Di Mauro, Gala et al. [*Random Probabilistic Circuits*](https://openreview.net/pdf?id=xzn1RVTCyB). UAI (2021).
[^14]: Desana and Schnörr. [*Learning Arbitrary Sum-Product Network Leaves with Expectation-Maximization*](https://arxiv.org/pdf/1604.07243.pdf). CoRR (2016).
[^15]: Peharz et al. [*Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits*](http://proceedings.mlr.press/v119/peharz20a/peharz20a.pdf). ICML (2020).
