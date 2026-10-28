# phylogenetic_basis_statistics

Code accompanying the paper **Basis statistics for phylogenetic tree shape: \\ tree balance formulas beyond expectations** (Bauman, Daniel J and Rosenberg, Noah A, Submitted, 2026).

The repository computes tree balance indices for rooted binary tree topologies. It represents them as linear combinations of the **clade-number basis** and the **daughter-number basis**, and uses those representations to get exact means, variances, covariances and correlations of the indices under the Yule model. The notebooks check these results against brute-force enumeration of all trees and produce the figures in the paper.

## Repository layout

```
phylogenetic_basis_statistics/
├── src/tree_bstats/                    # Python package
│   ├── tree.py                         # Tree: one unlabeled rooted binary topology + 29 balance indices
│   ├── unlabelled_tree.py              # UnlabelledTree: enumerates all topologies up to size n, with Yule/uniform weights
│   ├── clade_basis.py                  # Mean / covariance of clade-number vectors under the Yule model
│   ├── daughter_basis.py               # Mean / covariance of daughter-number vectors under the Yule model
│   ├── balance_index_coefficients.py   # Basis coefficients for the 17 indices represented by the bases
│   ├── indices.py                      # Index names and their LaTeX formula labels
│   └── plotting_params.py              # Plot styling for each index
├── notebooks/
│   ├── plot_figures.ipynb              # Makes every figure in the paper (main text and appendix)
│   ├── statisticMeans.ipynb            # Checks the means against enumeration and closed-form formulas
│   ├── statisticCorrelations.ipynb     # Checks the covariance/correlation matrix against enumeration
│   └── appendixC.ipynb                 # Shows that mean I and mean I' are not in the daughter-number basis (Appendix C)
├── mathematica/
│   └── MeanCalculations.nb             # Symbolic derivation of closed-form means (equations A.1–A.29)
├── plots/                              # Figures written by plot_figures.ipynb
├── pyproject.toml                      # Package metadata for tree_bstats
└── requirements.txt                    # Pinned versions used for the paper
```

## Installation

You need Python 3.10 or newer. The code was developed with Python 3.14.

1. Clone the repository:

   ```bash
   git clone https://github.com/dbauman18/phylogenetic_basis_statistics.git
   cd phylogenetic_basis_statistics
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS/Linux, activate it with `source .venv/bin/activate`. On Windows, use `.venv\Scripts\activate`.

3. Install the dependencies, the `tree_bstats` package and Jupyter:

   ```bash
   pip install -r requirements.txt jupyterlab
   ```

   `requirements.txt` pins the exact versions of `numpy`, `scipy`, `sympy`, `pandas`, `matplotlib` and `scikit-learn` that produced the paper's results. It also installs `tree_bstats` from `src/` in editable mode, so changes to the source take effect without reinstalling.

   If you only need the package, with whatever recent dependency versions pip picks, run `pip install .` instead.

## Running the code

Start Jupyter from the **repository root**:

```bash
jupyter lab
```

Then open a notebook in `notebooks/` and run all cells. If you use VS Code or another editor, select the `.venv` interpreter as the notebook kernel. `plot_figures.ipynb` saves its figures to `../plots/` relative to its own folder, so keep it in `notebooks/`.

| Notebook | What it does |
|---|---|
| `plot_figures.ipynb` | Makes Figures 2, 3 and C1–C3 and saves them to `plots/` |
| `statisticMeans.ipynb` | Computes index means three ways (basis representation, enumerating all trees, closed-form formulas from `MeanCalculations.nb`) and compares them for n ≤ 18 |
| `statisticCorrelations.ipynb` | Computes the correlation matrix by enumerating all trees and from the basis representations, and confirms they match |
| `appendixC.ipynb` | Uses exact rational row reduction (SymPy) to show that the mean I and mean I′ indices have no daughter-number basis representation |

Figure 3 uses n = 100 and the appendix figures enumerate every tree up to n = 16, so `plot_figures.ipynb` can take several minutes. For n > 50 the daughter-basis covariance matrix is computed with `multiprocessing`, which uses all available CPU cores.

`mathematica/MeanCalculations.nb` needs Wolfram Mathematica, or the free Wolfram Player if you only want to view it. None of the Python code depends on it.

## Using the package directly

Once installed, `tree_bstats` can be imported from anywhere:

```python
from tree_bstats.unlabelled_tree import UnlabelledTree
from tree_bstats.balance_index_coefficients import BalanceIndexCoefficients
from tree_bstats.daughter_basis import daughter_mean, daughter_cov_matrix

n = 20
coefs = BalanceIndexCoefficients(n)

# Exact Yule-model means of the 17 represented indices
means = coefs.calculate_daughter_basis_represented_balance_indices(daughter_mean(n))
print(means["Sackin index"])

# Variance of an index: a^T Σ a, where a holds its daughter-basis coefficients
a = coefs.get_statistic_coefficient("Sackin index", "daughter")
print(a @ daughter_cov_matrix(n) @ a)

# Enumerate all tree topologies up to size 8 and compute their balance indices
trees = UnlabelledTree(8)
print(trees.count_trees())   # [1, 1, 1, 2, 3, 6, 11, 23]
```

## Citation

If you use this code, please cite:

```
[CITATION]
```

## License

This project is released under the [MIT License](LICENSE).
