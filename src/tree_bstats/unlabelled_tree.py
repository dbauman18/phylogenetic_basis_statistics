''' 
This module contains the UnlabelledTree class, which generates all unlabelled trees of size up to n.
It directly builds on the Tree class, which represents a single unlabelled tree.
'''

import numpy as np
from typing import List, Tuple, Dict, Any
from .tree import Tree


class UnlabelledTree:
    '''
    Class to generate all unlabelled trees of size up to n. Directly builds on the Tree class.
    
    Attributes
    ----------
        n (int): the largest size of trees to generate
        r (int): tracks the total number of trees generated
        all_trees (dict): stores all trees of size i for i <= n
        balance_index_names (list): names of all balance indices it can compute
    '''

##########################################################################################

# Initialization

##########################################################################################

    def __init__(self, n: int) -> None:
        '''Initialize the UnlabelledTree object by generating all trees up to size n
        
        Args: 
            n (int):The size of trees to generate
            
        Raises: 
            ValueError: If n is less than 1'''
        if n < 1:
            raise ValueError("Tree size must be at least 1")
        
        if n >= 20:
            print("Warning: Requires huge amount memory to generate all trees of size up to " + str(n))
        
        self.n = n
        self.r = 1
        self.all_trees = {1: [Tree()]}
        self.balance_index_names = Tree().get_balance_names()
        self._generate_all_trees(n)
        print("Trees generated with counts ", self.count_trees())

    def _generate_size_k(self, k: int) -> None:
        '''
        Private helper function to generate all trees of size k given all trees of size i for i < k
        
        Args:
            k (int): the size of trees to generate

        Raises:
            KeyError: If trees of size i for i < k have not been generated yet 
        '''

        for i in range(1, k):
            if i not in self.all_trees:
                raise KeyError("Trees of size " + str(i) + " have not been generated yet")

        self.all_trees[k] = []
        if k%2 == 1:
            for i in range(1, k//2 + 1):
                for left in self.all_trees[i]:
                    for right in self.all_trees[k-i]:
                        self.r += 1
                        new_tree = left.merge(right)
                        new_tree.furnas = self.r
                        self.all_trees[k].append(new_tree)
        else:
            for i in range(1, k//2):
                for left in self.all_trees[i]:
                    for right in self.all_trees[k-i]:
                        self.r += 1
                        new_tree = left.merge(right)
                        new_tree.furnas = self.r
                        self.all_trees[k].append(new_tree)
            # Corner case of k being even--can't add duplicates
            for i in range(len(self.all_trees[k//2])):
                for j in range(i, len(self.all_trees[k//2])):
                    self.r += 1
                    new_tree = self.all_trees[k//2][i].merge(self.all_trees[k//2][j])
                    new_tree.furnas = self.r
                    self.all_trees[k].append(new_tree)
    
    def _generate_all_trees(self, n: int) -> None:
        '''Private helper function to generate all trees of size n

        Args:
            n (int): the size of trees to generate
            
        '''
        for i in range(2, self.n+1):
            self._generate_size_k(i)
        return
    
    def __repr__(self) -> str:
        """Return string representation of UnlabelledTree object.
        
        Returns:
            str: String representation including size and number of trees
        """
        return f"UnlabelledTree(n={self.n}, total_trees={sum(self.count_trees())})"
    
##########################################################################################

# Data extraction

##########################################################################################

    def get_data(self, k):
        '''Returns the clade numbers, daughter numbers, balance indices, and weights of all trees of size k

        Args:
            k (int): the size of trees to generate

        Returns:
            Tuple[List, List, List, List]: clade numbers, daughter numbers, balance indices, and weights of all trees of size k

        Raises:
            KeyError: If trees of size k have not been generated yet

        '''
        if k not in self.all_trees:
            raise KeyError("Trees of size " + str(k) + " have not been generated yet")
        
        clade_numbers = []
        daughter_numbers = []
        balance_indices = []
        weights = []
        for tree in self.all_trees[k]:
            clade_numbers.append(tree.clade_number_basis())
            daughter_numbers.append(tree.daughter_number_basis())
            balance_indices.append(tree.get_balance_indices())
            weights.append(tree.yule())
        return clade_numbers, daughter_numbers, balance_indices, weights
    
    def get_uniform_weights(self, k: int) -> np.ndarray:
        '''Returns the uniform weights of all trees of size n
        
        Args: 
            k (int): the size of trees to generate

        Returns:
            np.ndarray: the uniform weights of all trees of size n

        Raises:
            KeyError: If trees of size k have not been generated yet
        '''
        if k not in self.all_trees:
            raise KeyError("Trees of size " + str(k) + " have not been generated yet")
        
        weights = np.array([tree.uniform() for tree in self.all_trees[k]])
        return weights/np.sum(weights)
    
    def get_yule_weights(self, k: int) -> np.ndarray:
        '''Returns the Yule weights of all trees of size k

        Args:
            k (int): the size of trees to generate

        Returns:
            np.ndarray: the Yule weights of all trees of size k

        Raises:
            KeyError: If trees of size k have not been generated yet
        '''

        if k not in self.all_trees:
            raise KeyError("Trees of size " + str(k) + " have not been generated yet")

        weights = np.array([tree.yule() for tree in self.all_trees[k]])
        return weights/np.sum(weights)

    # Returns the count of trees of each size
    def count_trees(self) -> List[int]:
        '''Returns the count of trees of each size which have been generated

        Returns:
            List: the count of trees of each size which have been generated

        '''
        return [len(self.all_trees[i]) for i in range(1, self.n+1)]
    