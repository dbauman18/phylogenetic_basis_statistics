'''
This module contains the Tree class, which represents a rooted binary tree topology.
The class provides methods for merging trees, generating random trees, and computing
various balance indices for the tree.
'''

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import math


class Tree:
    '''A class to represent a unlabeled, rooted, binary tree topology

    This class represents unique tree topologies and implements various
    balance indices calculations and tree operations like merging two trees. 
    To be used in conjunction with the UnlabelledTree class to generate and analyze
    a set of trees.

    Note: if you create a tree object directly by passing in inputs to the constructor,
    the class will not guarantee that the tree is a valid rooted binary tree beyond a few
    small checks that each of the parameters are self-consistent. 

    Attributes
    ----------
        n (int) Number of nodes in the tree
        path_lengths (List[int]) List of path lengths from root to each leaf
        internal_path_lengths (List[int]) List of all path lengths from root to each internal node
        daughter_sizes (List[Tuple[int, int]]) List of daughter subtree sizes for each node
        height (List[int]) List of heights of each node
        sym_nodes (int) Number of symmetric nodes in the tree
        colijn_plazzota (int) Colijn-Plazzota rank of the tree
        furnas (int) Furnas rank of the tree (only computed when all trees are generated in UnlabelledTree class)
    '''

    def __init__(self, 
                 n: int = 1, 
                 path_lengths: List[int] = [0], 
                 internal_path_lengths: List[int] = [],
                 daughter_sizes: List[Tuple[int, int]] = [], 
                 height: List[int] = [], 
                 sym_nodes: int = 0, 
                 colijn_plazzota: int = 1,
                 newick: str = "") -> None:
        '''Initialize a Tree object with the given attributes

        Args:
            n (int): Number of nodes in the tree (default 1)
            path_lengths (List[int]): List of path lengths from root to each node (default [0])
            daughter_sizes (List[Tuple[int, int]]): List of daughter subtree sizes for each node (default [])
            height (List[int]): List of heights of each node (default [])
            sym_nodes (int): Number of symmetric nodes in the tree (default 0)
            colijn_plazzota (int): Colijn-Plazzota rank of the tree (default 1)

        Raises:
            ValueError: If n is less than 1
        '''

        if n < 1:
            raise ValueError("Tree must have at least 1 node")
        
        self.n = n
        self.path_lengths = path_lengths
        self.internal_path_lengths = internal_path_lengths
        self.daughter_sizes = daughter_sizes
        self.height = height
        self.sym_nodes = sym_nodes
        self.colijn_plazzota = colijn_plazzota
        self.newick = newick
        self.furnas = 1 # Furnas rank (only computed when all trees are generated in UnlabelledTree class)
        self._validate_tree() 

    def _validate_tree(self) -> None:
        '''Private method to perform basic check if the tree attributes could represent a valid tree

        This does not actually create/inspect the structural components of the tree, i.e. if the Colijn-Plazzota
        rank corresponds to the tree with the desired attributes. It only checks if the attributes are consistent
        with each other and the number of leaves.

        Raises:
            ValueError: If the tree attributes are invalid
        '''
        if len(self.path_lengths) != self.n:
            raise ValueError("Invalid path lengths: there should be n path lengths for a tree with n nodes")
        if len(self.daughter_sizes) != self.n - 1:
            raise ValueError("Invalid daughter sizes: there should be n-1 daughter sizes for a tree with n nodes (1 for each internal node)")
        if len(self.height) != self.n - 1:
            raise ValueError("Incorrect heights for tree with " + str(self.n) + " leaves: there should be n - 1 heights (1 for each internal node)")
        if self.sym_nodes < 0 or self.sym_nodes > self.n - 1:
            raise ValueError("Invalid number of symmetric nodes: must be between 0 and n-1")
        if self.colijn_plazzota < 1:
            raise ValueError("Invalid Colijn-Plazzota rank: must be at least 1")
        if sum(self.path_lengths) != sum([sum(pair) for pair in self.daughter_sizes]):
            raise ValueError("Valid rooted binary trees satisfy the equation: sum(path_lengths for leaves) = sum(subtree sizes for internal nodes)")
        
    def __repr__(self) -> str:
        '''String representation of the Tree object.
        
        Returns:
            str: A string representation of the Tree object, including the number of nodes and Colijn-Plazzota rank
        '''
        return f"Tree(n={self.n}, colijn_plazzota={self.colijn_plazzota})"
        

##########################################################################################

# Core tree operations

##########################################################################################

    def merge(self, other) -> 'Tree':
        '''Merge two trees by joining both to the root node

        Args:
            other (Tree): The other tree to merge with self

        Returns:
            Tree: A new Tree object representing the merged tree

        Raises:
            TypeError: If other is not a Tree object
            ValueError: If either tree has an invalid node count
        '''

        if not isinstance(other, Tree):
            raise TypeError("Can only merge Tree objects")
        if self.n < 1 or other.n < 1:
            raise ValueError("Cannot merge trees with invalid node counts")
        
        n = self.n + other.n
        path_lengths = self.path_lengths + other.path_lengths
        path_lengths = [x + 1 for x in path_lengths]
        internal_path_lengths = self.internal_path_lengths + other.internal_path_lengths
        internal_path_lengths = [x + 1 for x in internal_path_lengths]
        internal_path_lengths += [0] # Add the root node
        daughter_sizes = self.daughter_sizes + other.daughter_sizes + [(min(self.n, other.n), max(self.n, other.n))]
        height = self.height + other.height + [max(path_lengths)]
        sym_nodes = self.sym_nodes + other.sym_nodes
        if self == other:
            sym_nodes += 1
        colijn_plazzota_l = max(self.colijn_plazzota, other.colijn_plazzota)
        colijn_plazzota_r = min(self.colijn_plazzota, other.colijn_plazzota)
        colijn_plazzota = colijn_plazzota_l*(colijn_plazzota_l - 1)/2 + 1 + colijn_plazzota_r
        if self.furnas < other.furnas:
            newick = "(" + self.newick + "," + other.newick + ")"
        else:
            newick = "(" + other.newick + "," + self.newick + ")"
        return Tree(n, path_lengths, internal_path_lengths, daughter_sizes, height, sym_nodes, colijn_plazzota, newick)
    
    def random_tree(self, k: int) -> 'Tree':
        '''Generate a random tree of size k by recursively merging smaller trees with random splits

        Args:
            k (int): The size of the tree to generate

        Returns:
            Tree: A new Tree object representing the random tree

        Raises:
            ValueError: If k is less than 1
        '''
        if k < 1:
            raise ValueError("Cannot generate trees with less than 1 node")
        if k == 1:
            return Tree()
        i = np.random.randint(1, k) # Randomly choose a size for the left subtree
        i = min(i, k-i)
        return self.random_tree(i).merge(self.random_tree(k-i))
    
##########################################################################################

# Basic tree properties

##########################################################################################

    def __eq__(self, other) -> bool:
        '''Check if two trees are equal by comparing their Furnas and Colijn-Plazzota ranks

        Args:
            other (Tree): The other tree to compare with self

        Returns:
            bool: True if the trees are equal, False otherwise

        Raises:
            TypeError: If other is not a Tree object
        '''
        if not isinstance(other, Tree):
            raise TypeError("Can only compare Tree objects")
        return self.furnas == other.furnas and self.colijn_plazzota == other.colijn_plazzota
    
    def get_newick(self) -> str:
        ''' Returns a string representation of the tree in Newick format
        Returns:
            str: The Newick representation of the tree
        '''
        return str(self.newick + ";")
    
    def yule(self) -> float:
        '''Compute the probability of the tree under the Yule (ERM) model
        
        Returns:
            float: The Yule probability of the tree
        '''
        C = np.array([sum(pair) for pair in self.daughter_sizes])
        P = np.sum(np.log(C - 1))
        val = np.log(2)*(self.n - 1 - self.sym_nodes) - P
        return np.exp(val)

    def uniform(self) -> float:
        '''Compute the probability of the tree under the Uniform (PDA) model
        
        Returns:
            float: The uniform probability of the tree
        '''
        return 1/2**(self.sym_nodes)

##########################################################################################

# Basis statistic calculations for clade/daughter basis

##########################################################################################

    def clade_number_basis(self) -> np.ndarray:
        '''Get the clade number basis for the tree
        
        Returns:
            np.ndarray: The clade number basis for the tree
        '''
        C = np.zeros(self.n - 1)
        for pair in self.daughter_sizes:  
            C[sum(pair) - 2] += 1
        return C
    
    def ij_to_k(self, i: int, j: int) -> int:
        '''Translate a pair of indices (i, j) to the corresponding linear index k for the daughter number basis.
        If i > j, the pair is swapped
        
        Args:
            i (int): The first index
            j (int): The second index
            
        Returns:
            int: The linear index k corresponding to the pair (i, j)
        
        Raises:
            ValueError: If i or j are invalid values (less than 1 or greater than n or i + j > n)
        '''
        if i < 1 or j < 1 or i > self.n or j > self.n or i + j > self.n:
            raise ValueError("i and j must be at least 1 and at most n with i + j <= n")
        i, j = min(i, j), max(i, j)
        return int(i*(self.n+1-i) + j - self.n - 1)
    
    def daughter_number_basis(self) -> np.ndarray:
        '''Get the daughter number basis for the tree, with the ordering given by the ij_to_k function. 
        The convention is that we count D_11, D_12, ..., D_{1,(n-1)}, D_{22}, ..., D_{2,n-2},...,D([n/2], [n/2])

        Returns:
            np.ndarray: The daughter number basis for the tree
        '''

        max_k = self.ij_to_k(np.floor(self.n/2), np.ceil(self.n/2))
        D = np.zeros(max_k + 1)
        for pair in self.daughter_sizes:
            D[self.ij_to_k(pair[0], pair[1])] += 1
        return D
    
##########################################################################################

# Balance index calculations

##########################################################################################

    def get_balance_indices(self) -> List:
        '''Returns an array of the following balance indices for the tree:
        1. Sackin index
        2. Average Leaf Depth
        3. Variance of Leaf Depths
        4. Cophenetic Index
        5. s-shape
        6. B1
        7. B2
        8. Colless
        9. Corrected Colless
        10. I2
        11. Quadratic Colless
        12. Total-I
        13. Mean-I
        14. Total-I'
        15. Mean-I'
        16. Symmetry Nodes
        17. Rogers J
        18. Rooted Quartet Index
        19. Colijn-Plazzotta Rank
        20. Furnas Rank
        21. Total Internal Path Length
        22. Total Path Length
        23. Average Vertex Depth
        24. Maximum Depth
        25. Maximum Width
        26. Modified Maximum Difference in Widths
        27. Maximum Width over Maximum Depth
        28. stairs1
        29. stairs2 

        Returns:
            np.ndarray: An array of all the balance indices for the tree
        '''

        if self.n == 1:
            ans = [0]*len(self.get_balance_names())
            ans[15] = 1 # Furnas rank
            return ans
        
        return [self.sackin(), self.average_leaf_depth(), self.variance_of_leaf_depths(), self.cophenetic_index(), 
                self.s_hat(), self.b1(), self.b2(), self.colless(), self.corrected_colless(), self.I2(), 
                self.quadratic_colless(), self.total_I(), self.mean_I(), self.total_I_prime(), self.mean_I_prime(),
                self.symmetry_nodes(), self.rogers_J(), self.rQIB(), self.colijn_plazzota_rank(), self.furnas_rank(), 
                self.total_internal_path_length(), self.total_path_length(), self.average_vertex_depth(),
                self.maximum_depth(), self.maximum_width(), self.modified_maximum_difference_in_widths(),
                self.maximum_width_over_maximum_depth(), self.stairs1(), self.stairs2()]

    def get_balance_names(self) -> List[str]:
        '''Returns a list of the names of the balance indices that can be computed for the tree

        Returns:
            List[str]: A list of the names of the balance indices that can be computed for the tree
        '''
        return ["Sackin index", "Average leaf depth", "Variance of leaf depths", "Total cophenetic index", "$\\hat{s}$-shape", "B1 index", 
                "B2 index", "Colless index", "Corrected Colless index", "$I_2$ index", "Quadratic Colless index", "Total I index", "Mean I index", 
                "Total I' index", "Mean I' index", "Symmetry nodes index", 
                "Rogers J index", "Rooted Quartet index for binary trees", "Colijn-Plazzotta rank", "Furnas rank", "Total internal path length",
                "Total path length", "Average vertex depth", "Maximum depth", "Maximum width",
                "Modified maximum difference in widths", "Maximum width over maximum depth", "stairs1 index", "stairs2 index"]

    def balance_dict(self) -> Dict[str, float]:
        '''Returns a dictionary of the balance indices for the tree

        Returns:
            Dict[str, float]: A dictionary of the balance indices for the tree and their values
        '''
        return dict(zip(self.get_balance_names(), self.get_balance_indices()))


    def get_indices(self, indices: List[str]) -> Dict[str, float]:
        '''Returns a dictionary of the specified balance indices for the tree, if a method has been implemented
        to compute them and they are named correctly. To see the list of computable indices, call Tree().get_balance_names()

        Args:
            indices (List[str]): A list of the names of the balance indices to compute

        Returns:
            Dict[str, float]: A dictionary of the specified balance indices for the tree and their values
            If an index is not computable, the value will be None

        '''

        ans = {}
        for index in indices:
            if index in self.get_balance_names():
                ans[index] = self.get_balance_indices()[self.get_balance_names().index(index)]
            else:
                # if the index is not found, then say it is not computable
                ans[index] = None
        return ans

    
##########################################################################################

# Individual index calculations

##########################################################################################
    
    def balance(self) -> List[int]:
        '''Compute the balance of each node in the tree, defined as the absolute difference in the sizes of the daughter subtrees
        
        Returns:
            List[int]: A list of the balance of each node in the tree
        '''
        return [abs(pair[0] - pair[1]) for pair in self.daughter_sizes]
        
    def sackin(self) -> int:
        '''Compute the Sackin index of the tree
        
        Returns:
            int: The Sackin index of the tree
        '''
        return int(np.sum(self.path_lengths))

    def average_leaf_depth(self) -> float:
        '''Compute the average leaf depth of the tree.
        
        Returns:
            float: The average leaf depth of the tree
        '''
        return float(self.sackin()/self.n)

    def variance_of_leaf_depths(self) -> float:
        '''Compute the variance of the leaf depths of the tree

        Returns:
            float: The variance of the leaf depths of the tree
        '''
        return float(np.var(self.path_lengths))

    def cophenetic_index(self) -> int:
        '''Compute the total cophenetic index of the tree

        Returns:
            int: The total cophenetic index of the tree
        '''

        total = 0
        for pair in self.daughter_sizes:
            k = sum(pair)
            if k < self.n:
                total += (k * (k - 1)) / 2
        return int(total)

    def s_hat(self) -> float:
        '''Compute the s-hat shape index of the tree

        Returns: 
            float: the s-hat shape index of the tree
        '''
        total = 0
        for pair in self.daughter_sizes:
            nv = sum(pair)
            total += np.log(nv - 1)
        return float(total)
    
    def b1(self) -> float:
        '''Compute the B1 index of the tree
        
        Returns:
            float: the B1 index of the tree
        '''
        return float(np.sum(1/np.array(self.height[:-1]))) # don't include height of root node

    def b2(self) -> float:
        '''Compute the B2 index of the tree
        
        Returns:
            float: the B2 index of the tree
        '''
        pl = np.array(self.path_lengths)
        return float(np.sum(pl*(1/2)**pl))
    
    def colless(self) -> int:
        '''Compute the Colless index of the tree
        
        Returns:
            int: the Colless index of the tree'''
        return int(sum(self.balance()))
        
    def corrected_colless(self) -> float:
        '''Compute the Corrected Colless index of the tree
        
        Returns:
            float: the Corrected Colless index of the tree
            '''
        return float(2 * self.colless() / (self.n - 1) / (self.n - 2))

    def I2(self) -> float:
        '''Compute the I2 index of the tree (equal weights Colless)
        
        Returns:
            float: the I2 index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            nv = pair[0] + pair[1]
            bal = abs(pair[0] - pair[1])
            if nv > 2:
                tot += bal/(nv - 2)
        return float(tot/(self.n-2))

    def quadratic_colless(self) -> int:
        '''Compute the Quadratic Colless index of the tree
        
        Returns:
            int: the Quadratic Colless index of the tree
        '''
        return int(np.sum(np.array(self.balance())**2))
    
    def total_I(self) -> float:
        '''Compute the Total-I index of the tree

        Returns:
            float: the Total-I index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            nv = pair[0] + pair[1]
            n_max = max(pair)
            if nv >= 4:
                tot += (n_max - np.floor(nv/2))/((nv-1) - np.floor(nv/2))
        return float(tot)
    
    def mean_I(self) -> float:
        ''' Compute the mean-I index of the Tree

        Returns:
            float: the mean-I index of the tree
        '''
        sum = 0
        denominator = 0
        for pair in self.daughter_sizes:
            nv = pair[0] + pair[1]
            n_max = max(pair)
            if nv >= 4:
                denominator += 1
                sum += (n_max - np.floor(nv/2))/((nv-1) - np.floor(nv/2))
        if denominator == 0:
            return 0
        return float(sum / denominator)
    
    def total_I_prime(self) -> float:
        '''Compute the Total-I' index of the tree

        Returns:
            float: the Total-I' index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            nv = pair[0] + pair[1]
            n_max = max(pair)
            if nv >= 4:
                Iv = (n_max - np.floor(nv/2))/((nv-1) - np.floor(nv/2))
                if nv % 2 == 0:
                    tot += Iv * (nv - 1) / nv
                else:
                    tot += Iv
        return float(tot)
    
    def mean_I_prime(self) -> float:
        '''Compute the mean-I' index of the tree

        Returns:
            float: the mean-I' index of the tree
        '''
        sum = 0
        denominator = 0
        for pair in self.daughter_sizes:
            nv = pair[0] + pair[1]
            n_max = max(pair)
            if nv >= 4:
                denominator += 1
                Iv = (n_max - np.floor(nv/2))/((nv-1) - np.floor(nv/2))
                if nv % 2 == 0:
                    sum += Iv * (nv - 1) / nv
                else:
                    sum += Iv
        if denominator == 0:
            return 0
        return float(sum / denominator)
    
    def symmetry_nodes(self) -> int:
        '''Compute the symmetry nodes index of the tree

        Returns:
            int: the symmetry nodes index of the tree
        '''
        return int(self.n - 1 - self.sym_nodes)
    
    def rogers_J(self) -> int:
        '''Compute the Rogers J index of the tree

        Returns:
            int: the Rogers J index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            if pair[0] != pair[1]:
                tot += 1
        return int(tot)

    def rQIB(self) -> int:
        '''Compute the Rooted Quartet Index (rQIB) of the tree

        Returns:
            int: the Rooted Quartet Index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            tot += math.comb(pair[0], 2)  * math.comb(pair[1], 2)
        return int(tot)
    
    def colijn_plazzota_rank(self) -> float:
        '''Compute the Colijn-Plazzotta rank of the tree
        
        Returns:
            float: the Colijn-Plazzotta rank of the tree (may overflow for not too large trees and be infinite)
        '''
        return float(self.colijn_plazzota)
    
    def furnas_rank(self) -> int:
        '''Compute the Furnas rank of the tree. If this has not been computed, it will return 1
        (the default value).

        Returns:
            int: the Furnas rank of the tree
        '''
        return int(self.furnas)
    
    def total_internal_path_length(self) -> int:
        '''Compute the total internal path length of the tree
        
        Returns:
            int: the total internal path length of the tree
        '''
        return int(np.sum(self.internal_path_lengths))
    
    def total_path_length(self) -> int:
        '''Compute the total path length of the tree
        
        Returns:
            int: the total path length of the tree
        '''
        return int(self.total_internal_path_length() + self.sackin())
    
    def average_vertex_depth(self) -> float:
        '''Compute the average vertex depth of the tree
        
        Returns:
            float: the average vertex depth of the tree
        '''
        return float(self.total_path_length() / (2 * self.n - 1))
    
    def maximum_depth(self) -> int:
        '''Compute the maximum depth of the tree
        
        Returns:
            int: the maximum depth of the tree
        '''
        return int(max(self.path_lengths))
    
    def maximum_width(self) -> int:
        '''Compute the maximum width of the tree
        
        Returns:
            int: the maximum width of the tree
        '''
        max_width = 0
        all_path_lengths = self.path_lengths + self.internal_path_lengths
        for i in range(self.maximum_depth() + 1):
            w_i = len([x for x in all_path_lengths if x == i])
            if w_i > max_width:
                max_width = w_i
        return max_width
    
    def modified_maximum_difference_in_widths(self) -> int:
        '''Compute the modified maximum difference in widths of the tree
        
        Returns:
            int: the modified maximum difference in widths of the tree
        '''
        all_path_lengths = self.path_lengths + self.internal_path_lengths
        widths = [len([x for x in all_path_lengths if x == i]) for i in range(self.maximum_depth() + 1)]
        max_dif = max([widths[i] - widths[i-1] for i in range(1, len(widths))])
        return max_dif

    def maximum_width_over_maximum_depth(self) -> float:
        '''Compute the maximum width over maximum depth of the tree
        
        Returns:
            float: the maximum width over maximum depth of the tree
        '''
        return float(self.maximum_width() / self.maximum_depth()) if self.maximum_depth() > 0 else 0.0
    
    def stairs1(self) -> float:
        '''Compute the stairs1 index of the tree
        
        Returns:
            float: the stairs1 index of the tree
        '''
        return float(self.rogers_J()/(self.n - 1))

    def stairs2(self) -> float:
        '''Compute the stairs2 index of the tree

        Returns:
            float: the stairs2 index of the tree
        '''
        tot = 0
        for pair in self.daughter_sizes:
            tot += min(pair)/max(pair)
        return float(tot/(self.n - 1))
    
    