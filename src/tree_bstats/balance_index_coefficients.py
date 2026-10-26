'''
This module generates the coefficients for 17 balance indices in the clade basis and daughter basis.
The indices whose coefficients are calculated are:
- Sackin index
- Average leaf depth
- Total cophenetic index
- s-hat shape
- Colless index
- Corrected Colless index
- I_2 index
- Quadratic Colless index
- Total I index
- Total I' index
- Rogers J index
- Rooted quartet index for binary trees
- Total internal path length
- Total path length
- Average vertex depth
- stairs1 index
- stairs2 index

The main function is calculate_all_statistics(n), which creates dictionaries of the coefficients
of the clade basis and daughter basis for all listed statistics above, for trees of size n.

'''

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List, Optional
from .daughter_basis import indices




class BalanceIndexCoefficients:
    '''Class to generate the coefficients for the clade basis and daughter basis for a set of balance indices. The balance indices included are:
    - Sackin index
    - Average leaf depth
    - Total cophenetic index
    - s-hat shape
    - Colless index
    - Corrected Colless index
    - I_2 index
    - Quadratic Colless index
    - Total I index
    - Total I' index
    - Rogers J index
    - Rooted quartet index for binary trees
    - Total internal path length
    - Total path length
    - Average vertex depth
    - stairs1 index
    - stairs2 index

    Attributes:
    ----------
        n (int): the size of the tree (number of leaves)
        clade_k (List[int]): the order of the clade basis coefficients: [2, ..., n]
        daughter_ij (List[Tuple[int, int]]): the order of the daughter basis coefficients: [(1, 1), (1, 2), ..., (floor(n/2), ceil(n/2))]
        clade_coefficients (Dict[str, NDArray[np.float64]]): the coefficients for the clade basis for each statistic
        daughter_coefficients (Dict[str, NDArray[np.float64]]): the coefficients for the daughter basis for each statistic

    Main Methods:
    -------------
        get_clade_labels(): Returns a list of labels for the clade basis coefficients which this class generates.
        get_daughter_labels(): Returns a list of labels for the daughter basis coefficients which this class generates.
        get_statistic_coefficient(statistic: str, basis: str) -> NDArray[np.float64]: Returns the coefficients for a given statistic and basis (clade or daughter).
        get_clade_coefficients(): Returns the coefficients for all statistics which are represented by the clade basis in a dictionary.
        get_daughter_coefficients(): Returns the coefficients for all statistics which are represented by the daughter basis in a dictionary.
        calculate_clade_basis_represented_balance_indices(clade_num: NDArray[np.float64]): Calculates the balance indices represented by the clade basis for a given clade number.
        calculate_daughter_basis_represented_balance_indices(daughter_num: NDArray[np.float64]): Calculates the balance indices represented by the daughter basis for a given daughter number.

    '''

    def __init__(self, n: int = 3) -> None:
        '''Initializes the BalanceIndexCoefficients object.

        Args:
            n (int): the size of the tree (number of leaves). Default is 3.

        Raises:
            ValueError: if n < 3 (since basis statistics are not defined for trees of size 1 and some statistics are not defined for trees of size 2).

        '''
        if n < 3:
            raise ValueError("The basis statistics (and corresponding balance index coefficients) are not all defined for n < 3.")
        
        self.n = n
        self.clade_k, self.daughter_ij = self._get_coefficient_order(n)
        self.clade_coefficients, self.daughter_coefficients = self._calculate_all_statistics()

    
    def __repr__(self) -> str:
        '''Return string representation of BalanceIndexCoefficients object.
        
        Returns:
            str: String representation including size of tree
        '''
        return f"BalanceIndexCoefficients(n={self.n})"


    def _get_coefficient_order(self, n: int) -> Tuple[List[int], List[Tuple[int, int]]]:
        '''Private helper function to get the coefficient order of the clade basis and daughter basis for a given n using the conventions established.
        
        Args:
            n (int): the size of the tree
            
        Returns:
            Tuple[List[int], List[Tuple[int, int]]]: 
                A tuple of two numpy arrays. The first array contains the order of the clade basis coefficients: [2, ..., n]
                The second array contains the order of the daughter basis coefficients: [(1, 1), (1, 2), ..., (floor(n/2), ceil(n/2))]
        
        Raises: 
            ValueError: if n < 2 (since these statistics are not defined for trees of size 1)'''
        if n < 2:
            raise ValueError("The statistics are not defined for n < 2.")
        
        clade_k = list(range(2, n+1))                       # order for Clade Basis
        daughter_ij = indices(n)                            # order for Daughter Basis (calls indices function from daughter_basis.py)
        return clade_k, daughter_ij


    def _calculate_all_statistics(self) -> Tuple[Dict[str, NDArray[np.float64]], 
                                              Dict[str, NDArray[np.float64]]]:
        """Private helper function to calculate coefficients for all statistics.
        
        Returns:
            Tuple[Dict[str, NDArray[np.float64]], Dict[str, NDArray[np.float64]]]:
                A tuple of two dictionaries. The first dictionary contains the coefficients of the clade basis
                for each statistic which is represented by the clade basis. The second dictionary contains the coefficients of the daughter basis 
                for each statistic which is represented by the daughter basis.
        """

        clade_coef = {}
        daughter_coef = {}

        # Map of statistic names to their calculation methods
        calculations = {
            'Sackin index': self._calculate_sackin_coef,
            'Average leaf depth': self._calculate_avg_leaf_depth_coef,
            'Total cophenetic index': self._calculate_tot_cophenetic_coef,
            '$\\hat{s}$-shape statistic': self._calculate_s_hat_shape_coef,
            'Colless index': self._calculate_colless_coef,
            'Corrected Colless index': self._calculate_corrected_colless_coef,
            '$I_2$ index': self._calculate_I2_coef,
            'Quadratic Colless index': self._calculate_quad_colless_coef,
            'Total $I$ index': self._calculate_total_I_coef,
            'Total $I\'$ index': self._calculate_total_I_prime_coef,
            'Rogers $J$ index': self._calculate_rogers_J_coef,
            'Rooted quartet index for binary trees': self._calculate_rooted_quartet_coef,
            'Total internal path length': self._calculate_tot_internal_path_length_coef,
            'Total path length': self._calculate_tot_path_length_coef,
            'Average vertex depth': self._calculate_avg_vertex_depth_coef,
            'stairs1 index': self._calculate_stairs1_coef,
            'stairs2 index': self._calculate_stairs2_coef,
        }

        for name, calc_method in calculations.items():
            a_k, b_ij = calc_method()
            if a_k is not None:
                clade_coef[name] = a_k
            daughter_coef[name] = b_ij

        return clade_coef, daughter_coef
    
##########################################################################################

# Main public methods

##########################################################################################

    def get_clade_labels(self) -> List[str]:
        '''Returns the labels for the clade basis coefficients which this class generates.
        
        Returns:
            List[str]: List of labels for the clade basis coefficients.
        '''
        return ['Sackin index', 'Average leaf depth', 'Total cophenetic index', '$\\hat{s}$-shape statistic', 'Quadratic Colless index',
                'Total internal path length', 'Total path length', 'Average vertex depth',]
    

    def get_daughter_labels(self) -> List[str]:
        '''Returns the labels for the daughter basis coefficients which this class generates.
        
        Returns:
            List[str]: List of labels for the daughter basis coefficients.
        '''
        return ['Sackin index', 'Average leaf depth', 'Total cophenetic index', '$\\hat{s}$-shape statistic', 'Colless index', 
                'Corrected Colless index', '$I_2$ index', 'Quadratic Colless index', 'Total $I$ index', 'Total $I\'$ index', 
                'Rogers $J$ index', 'Rooted quartet index for binary trees', 'Total internal path length', 'Total path length', 
                'Average vertex depth', 'stairs1 index', 'stairs2 index']

    def get_statistic_coefficient(self, statistic: str, basis: str) -> NDArray[np.float64]:
        '''Returns the coefficients for a given statistic and basis (clade or daughter).
        
        Args:
            statistic (str): The name of the statistic. Must be one of the statistics in get_clade_labels() or get_daughter_labels().
            basis (str): The basis to return coefficients for ('clade' or 'daughter').
            
        Returns:
            NDArray[np.float64]: The coefficients for the specified statistic and basis.
        
        Raises:
            ValueError: If the statistic or basis is not recognized.
        '''
        if basis == 'clade':
            if statistic in self.clade_coefficients:
                return self.clade_coefficients[statistic]
            else:
                raise ValueError(f"Statistic '{statistic}' not found in clade basis.")
        elif basis == 'daughter':
            if statistic in self.daughter_coefficients:
                return self.daughter_coefficients[statistic]
            else:
                raise ValueError(f"Statistic '{statistic}' not found in daughter basis.")
        else:
            raise ValueError("Basis must be either 'clade' or 'daughter'.")

    def get_clade_coefficients(self) -> Dict[str, NDArray[np.float64]]:
        '''Returns the coefficients for all statistics which are represented by the clade basis.
        
        Returns:
            Dict[str, NDArray[np.float64]]: Dictionary of the coefficients for the clade basis for each statistic.
        '''
        return self.clade_coefficients


    def get_daughter_coefficients(self) -> Dict[str, NDArray[np.float64]]:
        '''Returns the coefficients for all statistics which are represented by the daughter basis.
        
        Returns:
            Dict[str, NDArray[np.float64]]: Dictionary of the coefficients for the daughter basis for each statistic.
        '''
        return self.daughter_coefficients
    

    def calculate_clade_basis_represented_balance_indices(self, clade_num: NDArray[np.float64]) -> Dict[str, float]:
        '''Calculates the balance indices represented by the clade-number basis for a given clade number value. 
        Use .get_clade_labels() to see which balance indices are represented by the clade basis.

        Args:
            clade_num (NDArray[np.float64]): The clade numbers for a tree of size n
            
        Returns:
            Dict[str, float]: A dictionary of the calculated balance indices for the clade basis.

        Raises:
            ValueError: If the length of clade_num is not equal to n-1.
        '''
        if len(clade_num) != len(self.clade_k): 
            raise ValueError("The length of clade_num must be equal to n-1.")
        
        clade_stats = {}
        for name, coef in self.clade_coefficients.items():
            clade_stats[name] = np.dot(coef, clade_num)
        return clade_stats
    

    def calculate_daughter_basis_represented_balance_indices(self, daughter_num: NDArray[np.float64]) -> Dict[str, float]:
        '''Calculates the balance indices represented by the daughter-number basis for a given daughter number value. 
        Use .get_daughter_labels() to see which balance indices are represented by the daughter basis.
        
        Args:
            daughter_num (NDArray[np.float64]): The daughter numbers for a tree of size n
            
        Returns:
            Dict[str, float]: A dictionary of the calculated balance indices for the daughter basis.

        Raises:
            ValueError: If the length of daughter_num is not equal to the number of daughter basis coefficients.
        '''
        if len(daughter_num) != len(self.daughter_ij):
            raise ValueError("The length of daughter_num must be equal to the number of daughter basis coefficients (floor(n/2)*ceil(n/2)).")
        
        daughter_stats = {}
        for name, coef in self.daughter_coefficients.items():
            daughter_stats[name] = np.dot(coef, daughter_num)
        return daughter_stats



##########################################################################################

# Individual balance index functions for calculating coefficients

##########################################################################################

    # Sackin index
    def _calculate_sackin_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the Sackin Index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''
 
        a_k = np.array(self.clade_k)                             # Coefficients for clade basis
        b_ij = np.array([sum(ij) for ij in self.daughter_ij])    # Coefficients for daughter basis
        return a_k, b_ij

    # Average leaf depth
    def _calculate_avg_leaf_depth_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the average leaf depth for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.

        '''
        
        a_k, b_ij = self._calculate_sackin_coef()
        return a_k/self.n, b_ij/self.n

    # Total cophenetic index
    def _calculate_tot_cophenetic_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the Total Cophenetic Index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = np.array([k*(k-1)/2 for k in self.clade_k])
        a_k[-1] = 0
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j < self.n:
                b_ij[k] = ((i + j) * (i + j - 1)) / 2
        return a_k, b_ij

    # s-hat shape statistic
    def _calculate_s_hat_shape_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the s-hat shape statistic for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = np.array([np.log(k - 1) for k in self.clade_k])
        b_ij = np.array([np.log(sum(ij) - 1) for ij in self.daughter_ij])
        return a_k, b_ij

    # Colless index
    def _calculate_colless_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the Colless index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = None                                  # Not represented in clade basis
        b_ij = np.array([(ij[1] - ij[0]) for ij in self.daughter_ij])
        return a_k, b_ij

    # Corrected Colless index
    def _calculate_corrected_colless_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the corrected Colless index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''     

        a_k, b_ij = self._calculate_colless_coef()
        return a_k, b_ij * 2 / (self.n-1) / (self.n-2)

    # I_2 index
    def _calculate_I2_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the I_2 index fora tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''
        
        a_k = None
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j > 2:
                b_ij[k] = (j - i) / (i + j - 2) / (self.n - 2)
        return a_k, b_ij

    # Quadratic colless index
    def _calculate_quad_colless_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the quadratic colless statistic for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: 
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = np.array([k**2 for k in self.clade_k])
        a_k[-1] = self.n * (2 - self.n)
        b_ij = np.array([(ij[1] - ij[0])**2 for ij in self.daughter_ij])
        return a_k, b_ij

    # Total I index
    def _calculate_total_I_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the Total-I index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = None
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j >= 4:
                b_ij[k] = (j - np.ceil((i + j) / 2)) / (i + j - 1 - np.ceil((i + j) / 2))
        return a_k, b_ij

    # Total I' index
    def _calculate_total_I_prime_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the Total-I' index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = None
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j >= 4:
                b_ij[k] = (j - np.ceil((i + j) / 2)) / (i + j - 1 - np.ceil((i + j) / 2))
                if (i + j) % 2 == 0:
                    b_ij[k] *= (i + j - 1) / (i + j)
        return a_k, b_ij
    
    # Rogers J index
    def _calculate_rogers_J_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the Rogers J index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = None
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j > 2 and i < j:
                b_ij[k] = 1
        return a_k, b_ij

    # Rooted quartet index
    def _calculate_rooted_quartet_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the rooted quartet index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = None
        b_ij = np.array([(ij[0] * (ij[0] - 1) * ij[1] * (ij[1] - 1)) / 4 for ij in self.daughter_ij])
        return a_k, b_ij
    
    # Total internal path length
    def _calculate_tot_internal_path_length_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the total internal path length for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = np.arange(2, self.n + 1)
        a_k[-1] = (2 - self.n)
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i, j = self.daughter_ij[k]
            if i + j < self.n:
                b_ij[k] = i + j
            else:
                b_ij[k] = 2 - self.n
        return a_k, b_ij
    
    # Total path length
    def _calculate_tot_path_length_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the total path length for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k = 2 * np.arange(2, self.n + 1)
        a_k[-1] = 2
        b_ij = np.zeros(len(self.daughter_ij))
        for k in range(len(self.daughter_ij)):
            i,j = self.daughter_ij[k]
            if i + j < self.n:
                b_ij[k] = 2 * (i + j) 
            else:
                b_ij[k] = 2
        return a_k, b_ij
    
    # Average vertex depth
    def _calculate_avg_vertex_depth_coef(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Calculates the coefficients for the average vertex depth for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''

        a_k, b_ij = self._calculate_tot_path_length_coef()
        return a_k / (2 * self.n - 1), b_ij / (2 * self.n - 1)

    # stairs1 index
    def _calculate_stairs1_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the stairs1 index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''
        a_k, b_ij = self._calculate_rogers_J_coef()
        return a_k, b_ij / (self.n-1)

    # stairs2 index
    def _calculate_stairs2_coef(self) -> Tuple[Optional[NDArray[np.float64]], NDArray[np.float64]]:
        '''Calculates the coefficients for the stairs2 index for a tree of size n.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]:
                A tuple of two numpy arrays. The first array contains the coefficients for the clade basis.
                The second array contains the coefficients for the daughter basis.
        '''
    
        a_k = None
        b_ij = np.array([(ij[0] / ij[1] / (self.n - 1)) for ij in self.daughter_ij])
        return a_k, b_ij