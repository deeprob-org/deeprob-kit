#!/bin/bash
export PYTHONPATH=..

python run_spflow.py spn msweb --algs evi.mar.mpe.csampling
python run_spflow.py spn binarized_mnist --algs evi.mar.mpe.csampling
python run_spflow.py spn ad --algs evi.mar.mpe.csampling

python run_spflow.py binary-clt msweb --algs evi.mar.mpe
python run_spflow.py binary-clt binarized_mnist --algs evi.mar.mpe
python run_spflow.py binary-clt ad --algs evi.mar.mpe
