import numpy as np
data_type = np.float16

# Configuration parameters for CRT procedure
CRT = 'in'                             # Splitting mode: 'in' (use all data for learning), 'out' (train-test split)
is_pool = False                        # Whether to pool source and external domain data: True (pooled statistic), False (weighted statistic)
k_in = 0.5                             # Source domain split proportion/amount: 0.5 (50% for statistic calculation, 50% for learning)
k_E = 0.5                              # External domain split proportion/amount: 0.5 (50% for statistic calculation, 50% for learning)
m = 200                                # Number of pseudo-samples per original sample
A = None                               # List of indices indicating unlabeled data samples that have a similar conditional distribution to the internal data. If None, 'trans_lasso' is used; otherwise, 'Oracle_lasso' is used.
AE = None                              # List of indices indicating unlabeled data samples that have a similar conditional distribution to the external data. If None, 'trans_lasso' is used; otherwise, 'Oracle_lasso' is used.
sigma_select = 'not1'                  # Noise variance selection method: '1' (use fixed variance = 1), 'not1' (use estimated variance)
generate_X0_name = 'bootstrap'         # Pseudo-data generation method: 'sigma' (generate using normal distribution with estimated variance), 'bootstrap' (bootstrap residual method)
learn_name = 'Trans_lasso'             # Conditional distribution learning method: 'Trans_lasso' (transfer learning with lasso), 'pool_lasso' (pool source and unlabeled data with lasso)
lam_betahat = np.linspace(1, 0.1, 10)  # Regularization parameter for beta coefficient estimation: numeric value (e.g., '10') or sequence (e.g., np.linspace(1, 0.1, 10))
lam_const = np.linspace(1, 0.1, 10)    # Regularization parameter for constant term in CRT statistic: numeric value (e.g., '10') or sequence (e.g., np.linspace(1, 0.1, 10))


# Parameters for generating simulated data
item = 1000                            # Number of iterations
model_y = 'l'                          # Linearity of Y with respect to z: 'l' (linear), 'n' (nonlinear)
model_yE = 'l'                         # Linearity of external Y with respect to z: 'l' (linear), 'n' (nonlinear)
n = 100                                # Internal sample size
nE = 0                                 # External sample sizes (multiple values for comparison)
N = 100                                # Test sample sizes
N_un = 100                             # Unlabeled high-heterogeneity data size (number of unlabeled samples)
p = 200                                # Number of covariates/features
c = 0                                  # Conditional independence of internal Y and X given z (confounding level)
cE = 0                                 # Conditional independence of external YE and XE given zE (confounding level)
sigz = 'identity matrix'               # Covariance structure for z
K = 20                                 # Number of unlabeled datasets (number of auxiliary studies)
Hk_num = 3                             # Number of heterogeneous features
s = 16                                 # Sparsity level (number of non-zero coefficients)


# Coefficients for each dataset
beta = [0.3] * s + [0] * (p - s)                   # Initialize Beta
beta_E = beta                                      # External Beta

np.random.seed(24)
gamma_0 = np.random.normal(0, 1, s)
gamma = np.zeros(p)                                
gamma[int(s/2):(int(s/2)+s)] = gamma_0             # Initialize gamma
np.random.seed(48)
gamma_0E = np.random.normal(0, 1, s)
gammaE = gamma                                     # External gamma
a = np.vstack((beta, gamma))       
aE = np.vstack((beta_E, gammaE))   
aU = np.zeros((K, p))
A_real = np.arange(12)    
beta_0 = beta

A_set = set(A_real)
for i in range(K):                                 # Unlabel Beta
    if i in A_set:
        Hk = np.random.choice(range(p), Hk_num, replace=False)
        aU[i] = beta.copy()
        for j in Hk:
            aU[i, j] -= 0.3
    else:
        Hk = np.random.choice(range(p), 4 * s, replace=False)
        aU[i] = beta_0.copy()
        for j in Hk:
            aU[i, j] -= 0.6

params = {
'data_type': str(data_type),
'CRT': CRT,
'is_pool': is_pool,
"k_in": k_in,
"k_E": k_E,
"m": m,
"A": A,
"AE": AE,
'sigma_select' : sigma_select,
'generate_X0_name' : generate_X0_name,
'learn_name' : learn_name,
'lam_betahat' : lam_betahat if lam_betahat is None else lam_betahat.tolist(),
'lam_const' : lam_const if lam_const is None else lam_const.tolist(),

'model_y' : model_y,
'model_yE' : model_yE,
"n": n,
"nE": nE,
"N": N,
"N_un":N_un,
"p": p,
'sigz':sigz,
"item": item,
"cE": cE,
"c": c,
"K": K,
"Hk_num": Hk_num,
"s": s,
"beta": beta,
"beta_E": beta_E,
"beta_0": beta_0,
"gamma": gamma.tolist(),
'gammaE':gammaE.tolist(),
"A_real": A_real.tolist(),
"AE_real": np.arange(2, 12).tolist(),
}