[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![PyPI version](https://badge.fury.io/py/deeprob-kit.svg)](https://badge.fury.io/py/deeprob-kit)

![Logo](docs/deeprob-logo.svg)

## Abstract

**DeeProb-kit** is a general-purpose Python library providing a collection of deep probabilistic models (DPMs) which
are easy to use and extend.
It also includes efficiently implemented learning techniques, inference routines and statistical algorithms.
The availability of a representative selection of the most common DPMs in a single library makes it possible to combine
them in a straightforward manner, a common practice in deep learning research nowadays, which however is still missing
for certain class of models. 
Moreover, **DeeProb-kit** provides high-quality fully-documented APIs, and it will help the community to accelerate research
on DPMs as well as improve experiments' reproducibility.

## Features

- Inference algorithms for SPNs. <sup>[1](#r1) [4](#r4)</sup>
- Learning algorithms for SPNs structure. <sup>[1](#r1) [2](#r2) [3](#r3) [4](#r4) [5](#r5)</sup>
- Chow-Liu Trees (CLT) as SPN leaves. <sup>[12](#r12) [13](#r13)</sup>
- Batch Expectation-Maximization (EM) for SPNs with arbitrarily leaves. <sup>[14](#r14) [15](#r15)</sup>
- Structural marginalization and pruning algorithms for SPNs.
- High-order moments computation for SPNs.
- JSON I/O operations for SPNs and CLTs. <sup>[4](#r4)</sup>
- Plotting operations based on NetworkX for SPNs and CLTs. <sup>[4](#r4)</sup>
- Randomized And Tensorized SPNs (RAT-SPNs). <sup>[6](#r6)</sup>
- Deep Generalized Convolutional SPNs (DGC-SPNs). <sup>[11](#r11)</sup>
- Masked Autoregressive Flows (MAFs). <sup>[7](#r7)</sup>
- Real Non-Volume-Preserving (RealNVP) flows. <sup>[8](#r8)</sup>
- Non-linear Independent Component Estimation (NICE) flows. <sup>[9](#r9)</sup>

The collection of implemented models is summarized in the following table.
The supported data dimensionality for each model is showed in the *Input Dimensionality* column.
Moreover, the *Supervised* column tells which model is suitable for a supervised learning task,
other than density estimation task.

Legend — D: one-dimensional size, C: channels, H: height, W: width.

| Model      | Description                                        | Input Dimensionality | Supervised |
|------------|----------------------------------------------------|:--------------------:|:----------:|
| Binary-CLT | Binary Chow-Liu Tree (CLT)                         |           D          |      ❌     |
| SPN        | Vanilla Sum-Product Network                        |           D          |      ✔     |
| MSPN       | Mixed Sum-Product Network                          |           D          |      ✔     |
| XPC        | Random Probabilistic Circuit                       |           D          |      ✔     |
| RAT-SPN    | Randomized and Tensorized Sum-Product Network      |           D          |      ✔     |
| DGC-SPN    | Deep Generalized Convolutional Sum-Product Network |       (C, D, D)      |      ✔     |
| MAF        | Masked Autoregressive Flow                         |           D          |      ❌     |
| NICE       | Non-linear Independent Components Estimation Flow  |   D and (C, H, W)    |      ❌     |
| RealNVP    | Real-valued Non-Volume-Preserving Flow             |   D and (C, H, W)    |      ❌     |

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

<b id="r1">1.</b> Peharz et al. [*On Theoretical Properties of Sum-Product Networks*][Peharz2015]. AISTATS (2015).

<b id="r2">2.</b> Poon and Domingos. [*Sum-Product Networks: A New Deep Architecture*][PoonDomingos2011]. UAI (2011).

<b id="r3">3.</b> Molina, Vergari et al. [*Mixed Sum-Product Networks: A Deep Architecture for Hybrid Domains*][MolinaVergari2018]. AAAI (2018).

<b id="r4">4.</b> Molina, Vergari et al. [*SPFLOW : An easy and extensible library for deep probabilistic learning using Sum-Product Networks*][MolinaVergari2019]. CoRR (2019).

<b id="r5">5.</b> Di Mauro et al. [*Sum-Product Network structure learning by efficient product nodes discovery*][DiMauro2018]. AIxIA (2018).

<b id="r6">6.</b> Peharz et al. [*Probabilistic Deep Learning using Random Sum-Product Networks*][Peharz2020a]. UAI (2020).

<b id="r7">7.</b> Papamakarios et al. [*Masked Autoregressive Flow for Density Estimation*][Papamakarios2017]. NeurIPS (2017).
   
<b id="r8">8.</b> Dinh et al. [*Density Estimation using RealNVP*][Dinh2017]. ICLR (2017).

<b id="r9">9.</b> Dinh et al. [*NICE: Non-linear Independent Components Estimation*][Dinh2015]. ICLR (2015).
   
<b id="r10">10.</b> Papamakarios, Nalisnick et al. [*Normalizing Flows for Probabilistic Modeling and Inference*][PapamakariosNalisnick2021]. JMLR (2021).
   
<b id="r11">11.</b> Van de Wolfshaar and Pronobis. [*Deep Generalized Convolutional Sum-Product Networks for Probabilistic Image Representations*][VanWolfshaarPronobis2020]. PGM (2020).

<b id="r12">12.</b> Rahman et al. [*Cutset Networks: A Simple, Tractable, and Scalable Approach for Improving the Accuracy of Chow-Liu Trees*][Rahman2014]. ECML-PKDD (2014).

<b id="r13">13.</b> Di Mauro, Gala et al. [*Random Probabilistic Circuits*][DiMauroGala2021]. UAI (2021).

<b id="r14">14.</b> Desana and Schnörr. [*Learning Arbitrary Sum-Product Network Leaves with Expectation-Maximization*][DesanaSchnörr2016]. CoRR (2016).
    
<b id="r15">15.</b> Peharz et al. [*Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits*][Peharz2020b]. ICML (2020).

[Peharz2015]: http://proceedings.mlr.press/v38/peharz15.pdf
[PoonDomingos2011]: https://arxiv.org/pdf/1202.3732.pdf
[MolinaVergari2018]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16865/16619
[MolinaVergari2019]: https://arxiv.org/pdf/1901.03704.pdf
[DiMauro2018]: http://www.di.uniba.it/~ndm/pubs/dimauro18ia.pdf
[Peharz2020a]: http://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf
[Papamakarios2017]: https://proceedings.neurips.cc/paper/2017/file/6c1da886822c67822bcf3679d04369fa-Paper.pdf
[Dinh2017]: https://arxiv.org/pdf/1605.08803v3.pdf
[Dinh2015]: https://arxiv.org/pdf/1410.8516.pdf
[PapamakariosNalisnick2021]: https://www.jmlr.org/papers/volume22/19-1028/19-1028.pdf
[VanWolfshaarPronobis2020]: http://proceedings.mlr.press/v138/wolfshaar20a/wolfshaar20a.pdf
[Rahman2014]: https://link.springer.com/content/pdf/10.1007%2F978-3-662-44851-9_40.pdf
[DiMauroGala2021]: https://openreview.net/pdf?id=xzn1RVTCyB
[DesanaSchnörr2016]: https://arxiv.org/pdf/1604.07243.pdf
[Peharz2020b]: http://proceedings.mlr.press/v119/peharz20a/peharz20a.pdf
