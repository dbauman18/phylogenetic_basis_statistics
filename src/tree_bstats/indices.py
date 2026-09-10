'''
This module contains the names of balance indices and their formula names
for all 29 statistics we consider in the Tree class.
'''

nameFormulas = {
    "Sackin index": "$S(T)$",
    "Average leaf depth": "$\\bar{N}(T)$",
    "Variance of leaf depths": "$\\sigma^2_{N}(T)$",
    "Total cophenetic index": "$\\Phi(T)$",
    "$\\hat{s}$-shape statistic": "$\\hat{s}(T)$",
    "$B_1$ index": "$B_1(T)$",
    "$B_2$ index": "$B_2(T)$",
    "Colless index": "$C(T)$",
    "Corrected Colless index": "$I_C(T)$",
    "$I_2$ index": "$I_2(T)$",
    "Quadratic Colless index": "$QC(T)$",
    "Mean $I$ index": "$\\bar{I}(T)$",
    "Total $I$ index": "$\\Sigma I(T)$",
    "Mean $I\'$ index": "$\\bar{I\'}(T)$",
    "Total $I\'$ index": "$\\Sigma I\'(T)$",
    "Symmetry nodes index": "$SNI(T)$",
    "Rogers $J$ index": "$J(T)$",
    "Rooted quartet index for binary trees": "$rQIB(T)$",
    "Colijn-Plazzotta rank": "$CP(T)$",
    "Furnas rank": "$F(T)$",
    "Total internal path length": "$TIP(T)$",
    "Total path length": "$TPL(T)$",
    "Average vertex depth": "$AVD(T)$",
    "Maximum depth": "$mD(T)$",
    "Maximum width": "$mW(T)$",
    "Modified maximum difference in widths": "$mdelW(T)$",
    "Maximum width over maximum depth": "$W/D(T)$",
    "stairs1 index": "$st1(T)$",
    "stairs2 index": "$st2(T)$",
}

def get_formula_name(statistic: str) -> str:
    '''
    Returns the formula name for a given statistic.
    
    Args:
        statistic (str): The name of the statistic.
        
    Returns:
        str: The formula name of the statistic.
    '''
    if statistic in nameFormulas:
        return nameFormulas[statistic]
    else:
        raise ValueError(f'Statistic "{statistic}" not found.')