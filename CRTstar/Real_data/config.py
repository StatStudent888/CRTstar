import numpy as np

# Configuration parameters for CRT procedure
CRT = 'in'                     # Splitting mode: 'in' (use all data for learning), 'out' (train-test split)
       
is_pool = False                # Whether to pool source and external domain data: True (pooled statistic), False (weighted statistic)
k_in = 0.5                     # Source domain split proportion/amount: 0.5 (50% for statistic calculation, 50% for learning)
k_E = 0.5                      # External domain split proportion/amount: 0.5 (50% for statistic calculation, 50% for learning)
m = 200                        # Number of pseudo-samples per original sample
       
A = None                       # List of indices indicating unlabeled data samples that have a similar conditional distribution to the internal data. If None, 'trans_lasso' is used; otherwise, 'Oracle_lasso' is used.
AE = None                      # List of indices indicating unlabeled data samples that have a similar conditional distribution to the external data. If None, 'trans_lasso' is used; otherwise, 'Oracle_lasso' is used.
sigma_select = 'not1'          # Noise variance selection method: '1' (use fixed variance = 1), 'not1' (use estimated variance)
generate_X0_name = 'bootstrap' # Pseudo-data generation method: 'sigma' (generate using normal distribution with estimated variance), 'bootstrap' (bootstrap residual method)
learn_name = 'Trans_lasso'     # Conditional distribution learning method: 'Trans_lasso' (transfer learning with lasso), 'pool_lasso' (pool source and unlabeled data with lasso)

lam_betahat = '10'             # Regularization parameter for beta coefficient estimation: numeric value (e.g., '10') or sequence (e.g., np.linspace(1, 0.1, 10))
lam_const = '10'               # Regularization parameter for constant term in CRT statistic: numeric value (e.g., '10') or sequence (e.g., np.linspace(1, 0.1, 10))

params = {
'CRT': CRT,
'A': A,
'AE': AE,
'sigma_select' : sigma_select,
'learn_name' : learn_name,
'generate_X0_name' : generate_X0_name,
'is_pool' : is_pool,
"k_in": k_in,
"k_E": k_E,
"m": m,
'lam_betahat' : lam_betahat if isinstance(lam_betahat, str) else lam_betahat,
'lam_const' : lam_const if isinstance(lam_const, str) else lam_const
}