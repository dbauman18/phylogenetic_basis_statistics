'''
This module contains functions for computed the mean, variance, and covariance of the daughter number basis under the Yule model of evolution. 
The daughter number basis is a set of quantities representing the number of nodes with daughters subtrees of size (i,j) in a phylogenetic tree.
The distribution of these quantities is determined by the distribution of trees induced by the Yule model of evolution. 

The main functions are:
- daughter_mean(n): Returns the mean vector E[D_{11}(T), ..., D_{1,n-1}(T), ..., D_{n/2,n/2}(T)] of the daughter number basis
- daughter_cov_matrix(n): Returns the covariance matrix Sigma_D of the daughter number basis
- daughter_inv_cov_matrix(n): Returns the pseudo-inverse of the covariance matrix Sigma_D of the daughter number basis
- sample_daughter_num(n): Generates a randomly sampled daughter number for a Yule generated tree
- indices(n): Returns a list by the convention ordering of the indices (i,j) for the daughter number basis
'''

from .clade_basis import M, V, C
import math
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

##########################################################################################

# Main functions: daughter_mean, daughter_cov_matrix, daughter_inv_cov_matrix, sample_daughter_num

##########################################################################################

def daughter_mean(n: int) -> NDArray[np.float64]:
    '''Compute the mean vector of the daughter number basis under the Yule model.
    This is a vector E[D_{11}(T), ..., D_{1,n-1}(T), ..., D_{n/2,n/2}(T)] of the expected number of nodes with daughters of each size.
    
    Args:
        n (int): The size of the tree

    Returns:
        np.array: The mean vector E[D_{11}(T), ..., D_{1,n-1}(T), ..., D_{n/2,n/2}(T)] of the daughter number basis
    
    Raises:
        ValueError: If n is less than 2
    '''
    if n < 2:
        raise ValueError("n must be at least 2 to define the daughter number basis and compute its mean")
    ind = indices(n)
    return np.array([expected_daughter_num(n, i[0], i[1]) for i in ind])


def daughter_cov_matrix(n: int, PARALLEL_CUTOFF: float = 50) -> NDArray[np.float64]:
    '''Compute the covariance matrix of the daughter number basis under the Yule model.
    If n is less than PARALLEL_CUTOFF, the naive implementation is used. Otherwise, a parallel implementation is used to improve performance.
    
    Args:
        n (int): The size of the tree
        PARALLEL_CUTOFF (int): The size of the tree at which to switch to the parallel implementation (default 50)
    
    Returns:
        np.array: The covariance matrix Sigma_D of the daughter number basis
    
    Raises:
        ValueError: If n is less than 2'''
    
    if n < 2:
        raise ValueError("n must be at least 2 to define the daughter number basis and compute its covariance matrix")
    
    if n < PARALLEL_CUTOFF:
        return sigma_d_naive(n)
    else:   
        return sigma_d_parallel(n)


def daughter_inv_cov_matrix(n: int, CUTOFF: float = 1e-5) -> NDArray[np.float64]:
    '''Compute the pseudo-inverse of the covariance matrix of the daughter number basis under the Yule model.
    Uses the cutoff value CUTOFF to determine which singular values to include in the pseudo-inverse.
    
    Args:
        n (int): The size of the tree
        CUTOFF (float): The cutoff value for singular values below which to truncate (default 1e-5)
        
    Returns:
        np.array: The pseudo-inverse of the covariance matrix Sigma_D of the daughter number basis
        
    Raises:
        ValueError: If n is less than 4 (since the covariance matrix is singular for n < 4)'''

    if n < 4:
        raise ValueError("n must be at least 4 to compute the pseudo-inverse covariance matrix")
    
    # Perform SVD to get pseudo-inverse
    sigma = daughter_cov_matrix(n)
    U, S, V = np.linalg.svd(sigma)
    S = S[S > CUTOFF]
    k = len(S)

    # Theoretical prediction is that k = floor((n-2)/2)*ceil((n-2)/2)
    expected_k = math.floor((n-2)/2)*math.ceil((n-2)/2)
    if k != expected_k:
        print(f"Warning: Trunkated to {k} singular values (expected {expected_k})")

    return V[:k, :].T @ np.diag(1/S) @ U[:, :k].T


def sample_daughter_num(n: int) -> NDArray[np.float64]:
    '''Generate a random daughter number for a Yule generated tree of size n.

    Args:
        n (int): The size of the tree

    Returns:
        np.array: A randomly sampled daughter number for a Yule generated tree of size n
    
    Raises:
        ValueError: If n is less than 2
    '''

    if n < 2:
        raise ValueError("n must be at least 2 to generate a random daughter number")
    
    # Helper function recursively returns list of daughter numbers for a random Yule tree of size n
    def gen_daughters(k):
        if k <= 1:
            return []
        i = np.random.randint(1, k)
        return [(i, k-i)] + gen_daughters(i) + gen_daughters(k-i)
    
    # Count daughter numbers
    D = [0]*(math.floor(n/2)*math.ceil(n/2))
    daughters = gen_daughters(n)
    for i, j in daughters:
        D[ij_to_k(n, i, j)] += 1
    return np.array(D)

##########################################################################################

# Linear Index Conversion Functions
# These functions convert between the linear index k and the pair of indices (i,j) for the daughter basis

##########################################################################################


def ij_to_k(n: int, i: int, j: int) -> int:
    '''Convert a pair of indices (i,j) to a linear index k for the daughter number basis.
    Convention is (1,1), (1,2), ..., (1,n-1), (2,2), ..., (2,n-2), ..., ([n/2],[n/2]). 
    Switches i and j if i > j to ensure i <= j.
    
    Args:
        n (int): The size of the tree
        i (int): The first index i
        j (int): The second index j
        
    Returns:
        int: The linear index k corresponding to the pair of indices (i,j)
        
    Raises:
        ValueError: If i or j are out of bounds (less than 1 or greater than n or i+j > n)'''
    # Check for out-of-bounds
    validate_index(n, i, j)
    
    i, j = min(i, j), max(i, j)
    return int(i * (n + 1 - i) + j - n - 1)


def k_to_ij(n: int, k: int) -> Tuple[int, int]:
    '''Convert a linear index k to a pair of indices (i,j) for the daughter number basis.
    Convention is (1,1), (1,2), ..., (1,n-1), (2,2), ..., (2,n-2), ..., ([n/2],[n/2]).

    Args:
        n (int): The size of the tree
        k (int): The linear index k

    Returns:
        Tuple[int, int]: The pair of indices (i,j) corresponding to the linear index k with i <= j
    
    Raises:
        ValueError: If k is out of bounds (less than 0 or greater than floor(n/2)*ceil(n/2))'''
    # Check for out-of-bounds
    if k < 0 or k >= math.floor(n/2)*math.ceil(n/2):
        raise ValueError("k must be at least 0 and less than floor(n/2)*ceil(n/2), the total number of daughter numbers")
    
    # Find correct i and j
    i = 1
    while k > n - 2*i:
        k -= n + 1 - 2*i
        i += 1
    return i, k+i


def indices(n: int) -> List[Tuple[int, int]]:
    '''Generate a list of all pairs of indices (i,j) for the daughter number basis.
    Convention is (1,1), (1,2), ..., (1,n-1), (2,2), ..., (2,n-2), ..., ([n/2],[n/2]).

    Args:
        n (int): The size of the tree

    Returns:
        List[Tuple[int, int]]: A list of all pairs of indices (i,j) for the daughter number basis
        
    Raises:
        ValueError: If n is less than 2'''
    
    if n < 2:
        raise ValueError("n must be at least 2 to define the daughter number basis")
    
    return [k_to_ij(n, k) for k in range(math.floor(n/2)*math.ceil(n/2))]


def validate_index(n: int, i: int, j: int) -> None:
    '''Helper function to validate that a pair of indices (i,j) is valid for the daughter number basis.
    Requires that i, j >= 1 and i+j <= n.
    
    Args:
        n (int): The size of the tree
        i (int): The first index i
        j (int): The second index j'''
    
    if i < 1 or j < 1 or i + j > n:
        raise ValueError(f"Indices ({i},{j}) are invalid for a tree of size {n}")

##########################################################################################

# Helper functions: pr, expected_daughter_num, variance_daughter_num, covariance_daughter_num 
# (help find individual components of matrices above)

##########################################################################################


# Probability of a root split (r, n-r) under Yule model
def pr(n: int, r: int) -> float:
    '''Compute the probability of a root split (r, n-r) in a Yule tree of size n.

    Args:
        n (int): The size of the tree
        r (int): The size of the left daughter clade

    Returns:
        float: The probability of a root split (r, n-r) in a Yule tree of size n

    Raises:
        ValueError: If r is less than 1 or greater than n'''
    if r < 1 or r > n:
        raise ValueError("r must be at least 1 and at most n to compute the split probability")
    
    if r == n / 2.0:
        return 1 / (n-1)
    else:
        return 2 / (n-1)

# E[D_{ij}(T)] under Yule model
def expected_daughter_num(n: int, i: int, j: int) -> float:
    '''Compute the expected value of the number of nodes with daughters of size (i,j) in a Yule tree of size n.
    
    Args:
        n (int): The size of the tree
        i (int): The size of the left daughter clade
        j (int): The size of the right daughter clade
        
    Returns:
        float: The expected number of nodes with daughters of size (i,j) in a Yule tree of size n
    
    Raises:
        ValueError: If i or j are out of bounds (less than 0 or greater than n or i+j > n)'''
    # Check for out-of-bounds
    validate_index(n, i, j)

    return pr(i+j, i) * M(n, i+j)


# Var[D_{ij}(T)] under Yule model
def variance_daughter_num(n: int, i: int, j: int) -> float:
    '''Compute the variance of the number of nodes with daughters of size (i,j) in a Yule tree of size n.

    Args:
        n (int): The size of the tree
        i (int): The size of the left daughter clade
        j (int): The size of the right daughter clade

    Returns:
        float: The variance of the number of nodes with daughters of size (i,j) in a Yule tree of size n

    Raises:
        ValueError: If i or j are out of bounds (less than 0 or greater than n or i+j > n)'''
    
    # Check for out-of-bounds
    validate_index(n, i, j)
    
    # Compute variance
    p = pr(i+j, i)
    return p * (1-p) * M(n,i+j) + p * p * V(n,i+j)

# Cov[D_{ij}(T),D_{kl}(T)] under Yule model
def covariance_daughter_num(n: int, i: int, j: int, k: int, l: int) -> float:
    '''Compute the covariance between the number of nodes with daughters of size (i,j) and (k,l) in a Yule tree of size n.

    Args:
        n (int): The size of the tree
        i (int): The size of the left daughter clade of the first pair
        j (int): The size of the right daughter clade of the first pair
        k (int): The size of the left daughter clade of the second pair
        l (int): The size of the right daughter clade of the second pair

    Returns:
        float: The covariance between the number of nodes with daughters of size (i,j) and (k,l) in a Yule tree of size n

    Raises:
        ValueError: If any of the indices are out of bounds (less than 0 or greater than n or i+j > n or k+l > n)'''
    
    # Check for out-of-bounds
    validate_index(n, i, j)
    validate_index(n, k, l)
    
    # First case: if i + j = k + l:
    if i + j == k + l:
        if i == k:
            return variance_daughter_num(n,i,j)
        else:
            return pr(i+j, i) * pr(i+j, k) * (V(n,i+j) - M(n,i+j))
        
    # Otherwise, relabel so that i + j < k + l
    if i + j > k + l:
        i, j, k, l = k, l, i, j

    r = i+j
    s = k + l
    
    # Final case: i + j < k + l
    C1 = C(n, r, s)
    C2 = -M(s, r)
    if k >= r:
        C2 += M(k, r)
    if l >= r:
        C2 += M(l, r)
    return pr(r, i) * pr(s, k) * (C1 + M(n, s)*C2)



##########################################################################################

# Implementations of computing covariance matrix Sigma_D 
# These are just private helper functions for daughter_cov_matrix

##########################################################################################

# Original, naive version (unoptimized except recognizing symmetry)
def sigma_d_naive(n: int) -> NDArray[np.float64]:
    '''Compute the covariance matrix of the daughter number basis under the Yule model using a naive implementation.
    This implementation is not parallelized and is used for small trees (n < 50).
    
    Args:
        n (int): The size of the tree
        
    Returns:
        np.array: The covariance matrix Sigma_D of the daughter number basis
        
    Raises:
        ValueError: If n is less than 2'''
    # Check for valid input
    if n < 2:
        raise ValueError("n must be at least 2 to define the daughter number basis and compute its covariance matrix")
    
    ind = indices(n)
    sigma = np.zeros((len(ind), len(ind)))
    for i in range(len(ind)):
        for j in range(i, len(ind)):
            sigma[i, j] = covariance_daughter_num(n, ind[i][0], ind[i][1], ind[j][0], ind[j][1])
            if i != j:
                sigma[j, i] = sigma[i, j]
    return sigma

# parallel implementation of sigma_d_naive
from multiprocessing import Pool
import os

def compute_row(args):
    '''Helper function to compute a row of the covariance matrix Sigma_D in parallel.
    
    Args:
        args: A tuple (n, ind, row_idx) containing the tree size n, list of indices ind, and the row index row_idx to compute
        
    Returns:
        Tuple[np.array, int]: A tuple containing the row of the covariance matrix Sigma_D and the row index
'''
    n, ind, row_idx = args
    size = len(ind)
    row = np.zeros(size)
    i, j = ind[row_idx]
    
    # Only compute elements from row_idx onwards (upper triangle)
    for col_idx in range(row_idx, size):
        k, l = ind[col_idx]
        row[col_idx] = covariance_daughter_num(n, i, j, k, l)
    
    return row, row_idx  # Return the row_idx so we know where to place it

def sigma_d_parallel(n, num_processes=None):
    '''Compute the covariance matrix of the daughter number basis under the Yule model using a parallel implementation.
    This implementation is used for large trees (n >= 50) to improve performance.

    Args:
        n (int): The size of the tree
        num_processes (int): The number of processes to use (default is os.cpu_count()-1)

    Returns:
        np.array: The covariance matrix Sigma_D of the daughter number basis

    Raises:
        ValueError: If n is less than 2'''
    # Check for valid input
    if n < 2:
        raise ValueError("n must be at least 2 to define the daughter number basis and compute its covariance matrix")
    
    # Use all but one CPU by default
    if num_processes is None:
        num_processes = os.cpu_count()-1
    
    ind = indices(n)
    size = len(ind)
    sigma = np.zeros((size, size))
    
    with Pool(processes=num_processes) as pool:
        # Only create args for upper triangle
        args = [(n, ind, row_idx) for row_idx in range(size)]
        results = pool.map(compute_row, args)
        
        # Fill in the matrix
        for row, row_idx in results:
            # Fill upper triangle
            sigma[row_idx, row_idx:] = row[row_idx:]
            # Fill lower triangle
            sigma[row_idx+1:, row_idx] = row[row_idx+1:]
            
    return sigma

