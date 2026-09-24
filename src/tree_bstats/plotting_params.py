'''
This module generates standard plotting parameters to use for balance indices
for all 29 statistics we consider in the Tree class.
'''

from .tree import Tree
from .indices import nameFormulas
import matplotlib.cm as cm
import numpy as np
from typing import Dict, Union, List

balance_index_names = Tree().get_balance_names()
index_formulas = nameFormulas
colors = cm.gist_earth(np.linspace(0, 0.8, len(balance_index_names)))
markers = ['o', '^', 's', '>', 'D', 'X']
linestyles = ['-', '--', ':']

def plot_params(statistic: str) -> Dict[str, Union[str, List[float]]]:
    '''
    Returns the plotting parameters for a statistic
    
    Args:
        statistic (str): The name of the statistic.
        
    Returns:
        Dict[str, str]: The plotting parameters of the statistic, e.g.:
                {
            'label'     : 'Sackin index,
            'formula'   : '$S(T)$',
            'color'     : [0.2, 0.5, 0.4],
            'marker'    : 'x',
            'linestyle' : '-'
            }
    '''
    if statistic in balance_index_names:
        params = {}
        params['label'] = statistic
        params['formula'] = index_formulas[statistic]
        i = balance_index_names.index(statistic)
        params['color'] = colors[i]
        params['marker'] = markers[i % len(markers)]
        params['linestyle'] = linestyles[i % len(linestyles)]
        return params
    else:
        raise ValueError(f"Statistic '{statistic}' not found.")
