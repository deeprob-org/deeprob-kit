import numpy as np

from typing import Union, Type, List

from deeprob.spn.structure.leaf import LeafType, Leaf
  

def entropy_cols(
        data: np.ndarray, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        random_state: np.random.RandomState,
        e: float = 0.3, 
        a: float = 1.0
) -> np.ndarray:
    '''
    Compute Entropy based splitting
    
    param data : the data
    param distributions : distributions of the features
    param domains : range of values of the features
    param e : threshold of the considered entropy to be signficant
    param a : laplacian alpha to apply at frequence
    return : a partitioning of features
    '''
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)
    
    # compute entropy for each variable
    for i in range(n_features):
        entropy = compute_entropy(data, i, distributions, np.array(domains), a)
        
        # add to cluster if entropy less than treshold
        if entropy < e :
            partition[i] = 1
        
    return partition

def entropy_adaptive_cols(
        data: np.ndarray, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        random_state: np.random.RandomState, 
        e: float = 0.3, 
        a: float = 1.0, 
        size: int = None
) -> np.ndarray:
    '''
    Compute Adaptive Entropy based splitting 
    
    param data : the data
    param distributions : distributions of the features
    param domains : range of values of the features
    param e : threshold of the considered entropy to be signficant
    param a : laplacian alpha to apply at frequence
    param size : size of whole dataset 
    return : a partitioning of features
    '''
    assert size is not None, "No size in entropy adaptive"
    
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)
    
    # compute entropy for each variable
    for i in range(n_features):
        entropy = compute_entropy(data, i, distributions, np.array(domains), a)
        
        # adaptive_entropy
        e = max(e * (data.shape[0] / size), 1e-07)
        
        # add to cluster if entropy less than treshold
        if entropy < e :
            partition[i] = 1
        
    return partition


def compute_entropy(
        data: np.ndarray, 
        idx: int, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        a: float
) -> float:
    '''
    Computes Entropy of a feature 
    
    param data : the data
    param idx : index of the feature
    param domains : domain of the feature (numpy array)
    param a : laplacian alpha to apply at frequence

    return: value of the entropy

    '''
    
    if distributions[idx].LEAF_TYPE == LeafType.DISCRETE: # binary
    
        one_counts = np.sum(data[:, idx])
        zero_counts = len(data[:, idx]) - one_counts
        smoth_freq = np.array([one_counts, zero_counts]) + a
        
        probs = smoth_freq / (data.shape[0] + (domains[idx] * a))
        log_probs = np.log2(probs)
        
        ent = -(probs * log_probs).sum()
    
    else: # continue

        bins = np.ceil(np.cbrt(data[:, idx].shape[0])).astype(np.int)
        hist, bin_edges = np.histogram(data[:, idx], bins=bins)
        smoth_freq = np.array(hist) + a
      
        probs = smoth_freq / (data.shape[0] + (bin_edges[1:] * a))
        log_probs = np.log2(probs)
        
        ent = - (probs * log_probs).sum() / np.log2(bins)
    
    if ent >1: ent = 1.0
    if ent <0: ent = 0.0
    
    return ent