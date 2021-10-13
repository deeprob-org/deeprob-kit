## Benchmark

The `benchmark` directory contains benchmark scripts of models and algorithms implemented in the library.
All the scripts can be launched by command line.
Every script will print out an estimation of the time required to run an algorithm (expressed in milliseconds).

The following table contains a description about the benchmark scripts and the external libraries used for comparison.
Please install the packages in `requirements.txt` to be able to run the scripts.

|       Benchmark      |                                    Description                               |  Compared Libraries |
|----------------------|------------------------------------------------------------------------------|:-------------------:|
| clt_queries.py       | Benchmark on Binary Chow-Liu Trees (CLTs): learning, inference and sampling. | [*SPFlow*][SPFlow]  |      
| spn_queries.py       | Benchmark on Sum-Product Networks (SPNs): inference and sampling.            | [*SPFlow*][SPFlow]  |

[SPFlow]: https://github.com/SPFlow/SPFlow 
