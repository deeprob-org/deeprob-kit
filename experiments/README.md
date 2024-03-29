# Experiments

A collection of 29 binary datasets, which most of them are used in *Probabilistic Circuits* literature,
can be found at [UCLA-StarAI-Binary-Datasets](https://github.com/UCLA-StarAI/Density-Estimation-Datasets).
Moreover, a collection of 5 continuous datasets, commonly present in works regarding *Normalizing Flows*,
can be found at [MAF-Continuous-Datasets](https://zenodo.org/record/1161203#.Wmtf_XVl8eN).

In order to run the experiments, it is necessary to clone the repository.
After downloading the datasets, they must be stored in the `experiments/datasets` directory to be able to
run the experiments.
Finally, install the development packages from the root directory as follows.
```bash
pip install -e .[develop]
```

The experiments scripts are available in the `experiments` directory and can be launched using the command line
by specifying the dataset and hyperparameters.
The following table shows the available experiments scripts.

|      Experiment      | Description                                                                     |
|----------------------|---------------------------------------------------------------------------------|
| energy.py            | Fit Sum-Product Networks (SPNs) and Normalizing Flows on energy functions. [^1] |
| spn.py               | Experiments for Sum-Product Networks (SPNs).                                    |
| ratspn.py            | Experiments for Randomized And Tensorized Sum-Product Networks (RAT-SPNs).      |
| dgcspn.py            | Experiments for Deep Generalized Convolutional Sum-Product Networks (DGC-SPNs). |
| flows.py             | Experiments for several Normalizing Flows models.                               |

[^1]: Rezende and Mohamed. [*Variational Inference with Normalizing Flows*](http://proceedings.mlr.press/v37/rezende15.pdf). ICML (2015).
