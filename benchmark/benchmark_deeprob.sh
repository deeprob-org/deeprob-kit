#!/bin/bash
export PYTHONPATH=..

python run_deeprob.py spn msweb --algs evi.mar.mpe.csampling
python run_deeprob.py spn binarized_mnist --algs evi.mar.mpe.csampling
python run_deeprob.py spn ad --algs evi.mar.mpe.csampling

python run_deeprob.py spn msweb --algs evi.mar.mpe.csampling --n-jobs -1
python run_deeprob.py spn binarized_mnist --algs evi.mar.mpe.csampling --n-jobs -1
python run_deeprob.py spn ad --algs evi.mar.mpe.csampling --n-jobs -1

python run_deeprob.py binary-clt msweb --algs evi.mar.mpe.csampling
python run_deeprob.py binary-clt binarized_mnist --algs evi.mar.mpe.csampling
python run_deeprob.py binary-clt ad --algs evi.mar.mpe.csampling
