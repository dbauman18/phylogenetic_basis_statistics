'''
This module contains functions for computed the mean, variance, and covariance of the clade number basis under the Yule model of evolution. 
The clade number basis is a set of random variables representing the number of clades of different sizes in a phylogenetic tree
which has been generated under the Yule model. 

The main functions are:
- clade_mean(n): Returns the mean vector E[C_2(T), ..., C_n(T)] of the clade number basis
- clade_cov_matrix(n): Returns the covariance matrix Sigma of the clade number basis
- clade_inv_cov_matrix(n): Returns the pseudo-inverse of the covariance matrix Sigma of the clade number basis
- sample_clade_num(n): Generates a randomly sampled clade number for a Yule generated tree
'''

import numpy as np
from numpy.typing import NDArray


##########################################################################################

# Main functions: clade_mean, clade_cov_matrix, clade_inv_cov_matrix, sample_clade_num

##########################################################################################

def clade_mean(n: int) -> NDArray[np.float64]:
    '''Compute the mean vector of the clade number basis under the Yule model.
    
    Args:
        n (int): The size of the tree

    Returns:
        np.array: The mean vector E[C_2(T), ..., C_n(T)] of the clade number basis

    Raises:
        ValueError: If n is less than 2
    '''
    if n < 2:
        raise ValueError("n must be at least 2 to define the clade number basis and compute its mean")
    
    return np.array([M(n, r) for r in range(2, n+1)])

def clade_cov_matrix(n: int) -> NDArray[np.float64]:
    '''Compute the covariance matrix of the clade number basis under the Yule model.
    
    Args:
        n (int): The size of the tree

    Returns:
        np.array: The covariance matrix Sigma of the clade number basis

    Raises:
        ValueError: If n is less than 2
    '''
    if n < 2:
        raise ValueError("n must be at least 2 to define the clade number basis and compute its covariance matrix")
    
    sigma = np.zeros((n-1, n-1))
    for i in range(n-1):
        for j in range(i, n-1):
            sigma[i, j] = C(n, i+2, j+2)
            if i != j:
                sigma[j, i] = sigma[i, j]
    return sigma


def clade_inv_cov_matrix(n: int, DEFAULT_CUTOFF: float = 1e-5) -> NDArray[np.float64]:
    '''Compute the pseudo-inverse of the covariance matrix of the clade number basis under the Yule model.
    
    Args:
        n (int): The size of the tree
        DEFAULT_CUTOFF (float): The cutoff value for singular values below which to truncate
        
    Returns:
        np.array: The pseudo-inverse of the covariance matrix Sigma of the clade number basis
        
    Raises:
        ValueError: If n is less than 4 (since the covariance matrix is singular for n < 4)'''
    if n < 4:
        raise ValueError("n must be at least 2 to define the clade number basis and compute its inverse covariance matrix")
    
    sigma = clade_cov_matrix(n)
    U, S, V = np.linalg.svd(sigma)
    S = S[S > DEFAULT_CUTOFF]
    k = len(S)

    # Theoretical prediction is that k = n-3
    expected_k = n-3
    if k != expected_k:
        print(f"Warning: Truncated to {k} singular values (expected {expected_k})")

    return V[:k, :].T @ np.diag(1/S) @ U[:, :k].T


def sample_clade_num(n: int) -> NDArray[np.float64]:
    '''Generate a random clade number for a Yule generated tree of size n.
    
    Args:
        n (int): The size of the tree
        
    Returns:
        np.array: A random clade number for a Yule generated tree of size n
        
    Raises:
        ValueError: If n is less than 2'''
    if n < 2:
        raise ValueError("n must be at least 2 to define the clade number basis and generate a random clade number")
    
    # Helper function recursively returns a list of the clade sizes for a random Yule tree of size n
    def gen_clade_num(k):
        if k <= 1:
            return []
        i = np.random.randint(1, k)
        return [k] + gen_clade_num(i) + gen_clade_num(k-i)
    
    # Count clade numbers
    C = [0]*(n-1)
    clades = gen_clade_num(n)
    for i in clades:
        C[i-2] += 1
    return np.array(C)



##########################################################################################

# Helper functions: M, V, C (corresponding to these expressions in main text of paper)

##########################################################################################

# M(n,r) = E[C_r(T)] under Yule model
def M(n: int, r: int) -> float:
    '''Compute the expected value of the number of clades of size r in a Yule tree of size n.
    
    Args:
        n (int): The size of the tree
        r (int): The size of the clade
        
    Returns:
        float: The expected number of clades of size r in a Yule tree of size n
    
    Raises:
        ValueError: If r is less than 1 or greater than n'''
    if r < 1 or r > n:
        raise ValueError("r must be at least 1 and at most n to compute the expected clade number")
    
    # Edge cases
    if r == 1:
        return n
    if r == n:
        return 1
    
    # Main formula for expected value
    if r > 1 and r < n:
        return 2 * n / r / (r+1)
    
# V(n,r) = Var[C_r(T)] under Yule model
def V(n: int, r: int) -> float:
    '''Compute the variance of the number of clades of size r in a Yule tree of size n.

    Args:
        n (int): The size of the tree
        r (int): The size of the clade

    Returns:
        float: The variance of the number of clades of size r in a Yule tree of size n

    Raises:
        ValueError: If r is less than 2 or greater than n'''
    if r < 2 or r > n:
        raise ValueError("r must be at least 2 and at most n to compute the variance")
    
    # Edge case (r = n)
    if r == n:
        return 0
    
    # Formulas for variance
    if r < n/2:
        return 2 * (4*r**2 - 3*r - 4) * (r-1) * n / r / (r+1)**2 / (2*r-1) / (2*r+1)
    if r == n/2:
        return (5*r - 7) * (r-1) * n / r / (r+1)**2 / (2*r-1)
    if r > n/2:
        return 2 * (r**2 + r - 2*n) * n / r**2 / (r+1)**2
    
# C(n,r,s) = Cov[C_r(T),C_s(T)] under Yule model
def C(n: int, r: int, s: int) -> float:
    '''Compute the covariance between the number of clades of size r and s in a Yule tree of size n.

    Args:
        n (int): The size of the tree
        r (int): The size of the first clade
        s (int): The size of the second clade

    Returns:
        float: The covariance between the number of clades of size r and s in a Yule tree of size n

    Raises:
        ValueError: If r or s is less than 1 or greater than n'''
    
    # Valid input check
    if r < 0 or r > n or s < 0 or s > n:
        raise ValueError("r and s must be at least 1 and at most n to compute the covariance")
    
    # Edge cases
    if r == s:
        return V(n, r)
    r, s = min(r,s), max(r,s)
    if r == 1 or r == n or s == 1 or s == n:
        return 0
    
    # Main formula for covariance
    non_disjoint_part = 4 * n * (s-n) / r / (r+1) / s / (s+1)
    if r + s == n:
        return non_disjoint_part + (2 / (n-1))
    if r + s > n:
        return non_disjoint_part
    else:
        return -4 * n * (r-1) * (r + 2*s) / s / (s+1) / (r+s) / (r+s+1) / (r+s-1)
