## Benchmark

The `benchmark` directory contains benchmark scripts of models and algorithms implemented in the library.
All the scripts can be launched by command line.
Every script will print an estimation of the time required to run each algorithm (in seconds) to a JSON file.
The following table contains a description about the available benchmark scripts.
Please install the packages in `requirements.txt` to be able to run the scripts.

|       Benchmark      |                        Description                        |
|----------------------|-----------------------------------------------------------|
| run_deeprob_clt.py   | Benchmark on DeeProb-kit of Binary Chow-Liu Trees (CLTs). |
| run_deeprob_spn.py   | Benchmark on DeeProb-kit of Sum-Product Networks (SPNs).  |
| run_spflow_clt.py    | Benchmark on SPFlow of Binary Chow-Liu Trees (CLTs).      |
| run_spflow_spn.py    | Benchmark on SPFlow of Sum-Product Networks (SPNs).       |
