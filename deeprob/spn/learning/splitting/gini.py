import numpy as np

from typing import Union, Type, List

from deeprob.spn.structure.leaf import LeafType, Leaf

def gini_cols(
        data: np.ndarray, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        random_state: np.random.RandomState,
        e: float = 0.3, 
        a: float = 1.0
) -> np.ndarray:
    '''
    Compute Gini's index based splitting
    
    param data : the data
    param distributions : distributions of the features
    param domains : range of values of the features
    param e : threshold of the considered gini to be signficant
    param a : laplacian alpha to apply at frequence
    return : a partitioning of features
    '''
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)
    
    # compute entropy for each variable
    for i in range(n_features):
        gini = compute_gini(data, i, distributions, np.array(domains), a)
        
        # add to cluster if entropy less than treshold
        if gini < e :
            partition[i] = 1
        
    return partition

def gini_adaptive_cols(
        data: np.ndarray, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        random_state: np.random.RandomState, 
        e: float = 0.3, 
        a: float = 1.0, 
        size: int = None
) -> np.ndarray:
    '''
    Compute Adaptive Gini's index based splitting 
    
    param data : the data
    param distributions : distributions of the features
    param domains : range of values of the features
    param e : threshold of the considered entropy to be signficant
    param a : laplacian alpha to apply at frequence
    param size : size of whole dataset 
    return : a partitioning of features
    '''
    assert size is not None, "No size in gini adaptive"
    
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=int)
    
    # compute entropy for each variable
    for i in range(n_features):
        gini = compute_gini(data, i, distributions, np.array(domains), a)
        
        # adaptive gini
        e = max(e * (data.shape[0] / size), 1e-07)
        
        # add to cluster if gini is less than treshold
        if gini < e :
            partition[i] = 1
        
    return partition
      

def compute_gini(
        data: np.ndarray, 
        idx: int, 
        distributions: List[Type[Leaf]], 
        domains: List[Union[list, tuple]], 
        a: float
) -> float:
    '''
    Computes Gini value of a feature
    
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
        
        gini = 1 - np.sum(probs**2)        
        
    else: # continue

        bins = np.ceil(np.cbrt(data[:, idx].shape[0])).astype(np.int)
        hist, bin_edges = np.histogram(data[:, idx], bins=bins)
        smoth_freq = np.array(hist) + a
      
        probs = smoth_freq / (data.shape[0] + (bin_edges[1:] * a))
        
        gini = 1 - np.sum(probs**2)
    
    if gini > 1: gini = 1.0 
    if gini < 0: gini = 0.0  
    
    return gini