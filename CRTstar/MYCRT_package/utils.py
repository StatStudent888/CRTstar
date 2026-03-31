import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso , LassoCV
import h5py
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

def DF_CRT_all(data, seed=None, params=None, path=None, show=True):
    """
    Perform Conditional Independence Test with CRT*.

    This function orchestrates the entire CRT* pipeline: splitting data, learning conditional
    distributions, generating pseudo-data, performing the CRT* test, and saving results.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility throughout the entire process.
    
    data : dict
        Dictionary containing the generated data with keys:
        'z', 'zE', 'zU', 'X', 'XE', 'XU', 'Y', 'YE'.
        
        Expected formats and dimensions:
        - 'z' : numpy.ndarray, shape (n, p)
            Source domain latent variables, where n is number of source samples, p is latent dimension.
        - 'zE' : numpy.ndarray, shape (n_E, p)
            External domain latent variables, where n_E is number of external samples.
        - 'zU' : list of numpy.ndarray, each shape (n_U_i, p)
            Unlabeled domain latent variables. List of arrays from multiple unlabeled sources.
            The total number of unlabeled samples is sum(n_U_i).
        - 'X' : numpy.ndarray, shape (n,)
            Source domain covariates. One-dimensional array of length n.
        - 'XE' : numpy.ndarray, shape (n_E,)
            External domain covariates. One-dimensional array of length n_E.
        - 'XU' : list of numpy.ndarray, each shape (n_U_i,)
            Unlabeled domain covariates. List of one-dimensional arrays, must have same structure as zU.
        - 'Y' : numpy.ndarray, shape (n,)
            Source domain outcomes.
        - 'YE' : numpy.ndarray, shape (n_E,)
            External domain outcomes.
        
        Note: 
        - All source domain arrays (z, X, Y) must have consistent sample size n.
        - All external domain arrays (zE, XE, YE) must have consistent sample size n_E.
        - For unlabeled domain, XU and zU must have matching list structure (same number of elements),
          and within each list element, the sample sizes must be consistent.
    
    params : dict
        Dictionary containing configuration parameters for the CRT* procedure.
        
        Required parameters:
        - lam_const : float
            Regularization parameter lambda for the constant term in the CRT* statistic.
        - lam_betahat : float
            Regularization parameter lambda for estimating beta coefficients.
        - k_in : float or int
            If CRT == 'out', specifies the proportion (float between 0 and 1) or absolute number (int)
            of samples to allocate to the statistic calculation set from the source domain.
        - k_E : float or int
            If CRT == 'out', specifies the proportion (float between 0 and 1) or absolute number (int)
            of samples to allocate to the statistic calculation set from the external domain.
        - learn_name : str
            Name of the learning method for conditional distribution estimation.
            Used to select the appropriate learning function via get_learn_X0_function.
        - generate_X0_name : str
            Name of the pseudo-data generation method.
            Used to select the appropriate generation function via get_learn_X0_function.
        - sigma_select : str or None
            Selection method for the noise variance sigma.
        - m : int
            Number of pseudo-samples to generate for each original sample.
        - is_pool : bool
            Whether to combine source and external domain data when computing the test statistic.
            If True, pools all data together and computes a single statistic;
            If False, computes statistics separately for each domain and combines them using a weighted approach.
        - CRT : str
            Splitting mode. If 'out', performs train-test split; otherwise, uses all data as conditional
            distribution learning set.
        - A : list or None
            List of indices indicating unlabeled data samples that have a similar conditional 
            distribution to the internal data. If None, 'trans_lasso' is used; otherwise, 
            'Oracle_lasso' is used.
        - AE : list or None
            List of indices indicating unlabeled data samples that have a similar conditional 
            distribution to the external data. If None, 'trans_lasso' is used; otherwise, 
            'Oracle_lasso' is used.
    
    path : str
        File path for saving output results (HDF5 format).
    
    show : bool, optional (default=True)
        If True, prints progress messages at each stage of the CRT* pipeline.

    Returns
    -------
    result : dict
        Dictionary containing complete results from the CRT* procedure with the following keys:
        
        - 'bata_hat_l' : list
            Learned coefficients [a_hat, aE_hat] for source and external domains respectively.
        
        - 'epsilon_mean_var' : list
            Mean and variance of residuals: [epsilon_mean, epsilon_var, epsilonE_mean, epsilonE_var]
        
        - 'epsilon_l' : list
            Residuals from the learned model: [epsilon, epsilonE]
        
        - 'an_l' : list
            Sample sizes used for learning: [an, anE]
        
        - 'sigma_l' : list
            Estimated noise variances: [sigma_hat, sigmaE_hat]

        - 'ktheta' : float or array
            Estimated ktheta parameter.
        
        - 'lam_const_l' : list
            Regularization constants used: [lam_const_in, lam_const_E] when is_pool=False,
            or [lam_const_in_m] when is_pool=True.
        
        - 'P' : list of p-values
            P-values returned in the following order:
            - Optimal weighting
            - Using only internal data
            All two p-values are returned when is_pool=False.
            When is_pool=True, returns a single p-value from the pooled data.
            
        - 'W' : float or int
            Optimal weight parameter for combining p-values. Non-zero only when is_pool=False
            and both domains have data.

    Notes
    -----
    The function executes the following steps:
    1. Set random seed for reproducibility.
    2. Split data into training (conditional distribution learning) and statistic calculation sets.
    3. Learn conditional distribution using the specified learning method.
    4. Generate pseudo-data based on the learned conditional distribution.
    5. Perform CRT* using the generated pseudo-data.
    6. Save results to HDF5 file and output file.
    
    The function also prints progress messages if show=True.
    """
    if params is None:
        params = {}
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed(318)
    CRT = params.get('CRT', 'in')
    is_pool = params.get('is_pool', False)
    k_in = params.get('k_in', 0.5)
    k_E = params.get('k_E', 0.5)
    m = params.get('m', 200)
    A = params.get('A', None)
    AE = params.get('AE', None)
    sigma_select = params.get('sigma_select', 'not1')
    generate_X0_name = params.get('generate_X0_name', 'bootstrap')
    learn_name = params.get('learn_name', 'Trans_lasso')
    lam_betahat = params.get('lam_betahat', '10')
    lam_const = params.get('lam_const', '10')
    
    learn_function, generate_X0_function = get_learn_X0_function(learn_name, generate_X0_name)

    data_train, data_stat = Split_the_data(seed, CRT, data, k_in, k_E)
    if show: print('Data splitting completed!')
    
    result = learn_function(data_train, sigma_select, A, AE, lam_betahat)
    if show: print('Conditional distribution learning completed!')  

    data_stat = generate_X0_function(data_stat, result, m)
    if show: print('Pseudo-data generation completed!')  

    result = Df_CRT(data_stat, result, m, lam_const, is_pool)
    if show: print('Df_CRT completed!')  
    
    if path:
        save_output_hdf5(path, params, result)
        save_out(result, path)
    return result

def Split_the_data(seed, CRT, data, k_in, k_E):
    """
    Split the generated data into conditional distribution learning set and statistic calculation set.

    This function partitions the source domain data and external domain data into conditional
    distribution learning subset and statistic calculation subset based on the specified splitting mode.
    The unlabeled data remains untouched and is included only in the conditional distribution learning set.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility of the split.
    CRT : str
        Splitting mode. If 'out', performs train-test split; otherwise, uses all data as conditional
        distribution learning set.
    data : dict
        Dictionary containing the generated data, expected to have keys:
        'z', 'zE', 'zU', 'X', 'XE', 'XU', 'Y', 'YE'.
        
        Expected formats and dimensions:
        - 'z' : numpy.ndarray, shape (n, p)
            Source domain latent variables, where n is number of source samples, p is latent dimension.
        - 'zE' : numpy.ndarray, shape (n_E, p)
            External domain latent variables, where n_E is number of external samples.
        - 'zU' : list of numpy.ndarray, each shape (n_U_i, p) OR single numpy.ndarray, shape (n_U, p)
            Unlabeled domain latent variables. Can be a list of arrays from multiple unlabeled sources,
            or a single array. The total number of unlabeled samples is sum(n_U_i) if list, else n_U.
        - 'X' : numpy.ndarray, shape (n, d_X)
            Source domain covariates.
        - 'XE' : numpy.ndarray, shape (n_E, d_X)
            External domain covariates.
        - 'XU' : list of numpy.ndarray, each shape (n_U_i, d_X) OR single numpy.ndarray, shape (n_U, d_X)
            Unlabeled domain covariates. Must have same structure (list or array) as zU.
        - 'Y' : numpy.ndarray, shape (n,)
            Source domain outcomes.
        - 'YE' : numpy.ndarray, shape (n_E,)
            External domain outcomes.
        
        Note: 
        - All source domain arrays (z, X, Y) must have consistent sample size n.
        - All external domain arrays (zE, XE, YE) must have consistent sample size n_E.
        - For unlabeled domain, XU and zU must have matching structure (both list or both array),
          and within each list element, the sample sizes must be consistent.
    k_in : float or int
        If CRT == 'out', specifies the proportion (float between 0 and 1) or absolute number (int) 
        of samples to allocate to the statistic calculation set from the source domain.
        Passed to train_test_split as test_size parameter.
    k_E : float or int
        If CRT == 'out', specifies the proportion (float between 0 and 1) or absolute number (int)
        of samples to allocate to the statistic calculation set from the external domain.
        Passed to train_test_split as test_size parameter. If external domain is empty, this parameter is ignored.

    Returns
    -------
    data_train : dict
        Dictionary containing conditional distribution learning set data with keys:
        'z', 'X', 'zE', 'XE', 'zU', 'XU'.
        
        When CRT == 'out':
        - 'z' : shape (n_train, p), where n_train = n - k_in if k_in is int, or n * (1 - k_in) if k_in is float
        - 'X' : shape (n_train, d_X)
        - 'zE' : shape (n_E_train, p), where n_E_train = n_E - k_E if k_E is int, or n_E * (1 - k_E) if k_E is float
        - 'XE' : shape (n_E_train, d_X)
        - 'zU' : same structure as input (list or array), unchanged
        - 'XU' : same structure as input (list or array), unchanged
        
        When CRT != 'out':
        - All arrays keep their original shapes (no splitting performed)
    data_stat : dict
        Dictionary containing statistic calculation set data with keys:
        'z', 'X', 'Y', 'zE', 'XE', 'YE'.
        
        When CRT == 'out':
        - 'z' : shape (n_stat, p), where n_stat = k_in if k_in is int, or n * k_in if k_in is float
        - 'X' : shape (n_stat, d_X)
        - 'Y' : shape (n_stat,)
        - 'zE' : shape (n_E_stat, p), where n_E_stat = k_E if k_E is int, or n_E * k_E if k_E is float
        - 'XE' : shape (n_E_stat, d_X)
        - 'YE' : shape (n_E_stat,)
        
        When CRT != 'out':
        - All arrays are identical to input (no splitting performed, all data used as statistic set)
    """
    z0, zE0, zU = data['z'], data['zE'], data['zU']
    X0, XE0, XU = data['X'], data['XE'], data['XU']
    Y0, YE0 = data['Y'], data['YE']
    _ , p = z0.shape
    ne , _ = zE0.shape
    if CRT == 'out':
        X_train, X, _, Y, z_train, z = train_test_split(X0, Y0, z0, test_size=k_in, random_state=seed)
        if ne == 0:
            XE = YE = XE_train = np.empty((0,))
            zE = zE_train = np.empty((0, p))
        else:
            XE_train, XE, _, YE, zE_train, zE = train_test_split(XE0, YE0, zE0, test_size=k_E, random_state=seed)
        data_train = {'z': z_train ,'X': X_train ,'zE': zE_train ,'XE': XE_train ,'zU': zU ,'XU': XU}
        data_stat = {'z': z ,'X': X ,'Y': Y ,'zE': zE ,'XE': XE ,'YE': YE}
    else:
        data_train = {'z': z0 ,'X': X0 ,'zE': zE0 ,'XE': XE0 ,'zU': zU ,'XU': XU}
        data_stat = {'z': z0 ,'X': X0 ,'Y': Y0 ,'zE': zE0 ,'XE': XE0 ,'YE': YE0}
    return data_train, data_stat

def learn_distribution(data_train,sigma_select,A,AE,lam_betahat,data_type=np.float16):
    """
    Learn the conditional distributions using trans_lasso.

    This function estimates the parameters of the conditional distributions using the
    conditional distribution learning set. It computes coefficient estimates, noise variances,
    residuals, and other statistical quantities required for distribution learning.

    Parameters
    ----------
    data_train : dict
        Dictionary containing the conditional distribution learning set data with keys:
        'z', 'zE', 'zU', 'X', 'XE', 'XU'.
    sigma_select : array-like
        Selection matrix for variance estimation.
    lam_betahat : float
        Regularization parameter for coefficient estimation.
    data_type : dtype, default=np.float16
        Data type for generated arrays.
    A : list or None
        List of indices indicating unlabeled data samples that have a similar conditional 
        distribution to the internal data. If None, 'trans_lasso' is used; otherwise, 
        'Oracle_lasso' is used.
    AE : list or None
        List of indices indicating unlabeled data samples that have a similar conditional 
        distribution to the external data. If None, 'trans_lasso' is used; otherwise, 
        'Oracle_lasso' is used.

    Returns
    -------
    result : dict
        Dictionary containing learned distribution parameters with keys:
        - 'bata_hat_l': List of estimated coefficients [a_hat, aE_hat]
        - 'epsilon_mean_var': List [epsilon_mean, epsilon_var, epsilonE_mean, epsilonE_var]
        - 'epsilon_l': List of residuals [epsilon, epsilonE]
        - 'an_l': List of sample sizes [an, anE]
        - 'sigma_l': List of estimated noise variances [sigma_hat, sigmaE_hat]
        - 'ktheta': Estimated ktheta parameter
        - 'lam_const_l': List of regularization constants [lam]
    """
    z_train, zE_train, zU = data_train['z'], data_train['zE'], data_train['zU']
    X_train, XE_train, XU = data_train['X'], data_train['XE'], data_train['XU']
  
    if A:
        a_hat, sigma_hat, epsilon, an, epsilon_list, lam = get_betahat(X_train,XU,z_train,zU,A,sigma_select,lam_betahat)
        A_hat = []
    else:
        a_hat, sigma_hat, epsilon, an, epsilon_list, A_hat = get_betahat_unknown(X_train,XU,z_train,zU,sigma_select,lam_betahat = lam_betahat)
        lam = 0

    if AE:
        aE_hat, sigmaE_hat, epsilonE, anE, epsilonE_list, lamE = get_betahat(XE_train,XU,zE_train,zU,AE,sigma_select,lam_betahat)
        AE_hat = []
    else:
        aE_hat, sigmaE_hat, epsilonE, anE, epsilonE_list, AE_hat = get_betahat_unknown(XE_train,XU,zE_train,zU,sigma_select,lam_betahat = lam_betahat)
        lamE = 0

    epsilon_mean = epsilon.mean()
    epsilonE_mean = epsilonE.mean()
    epsilon -= epsilon_mean
    epsilonE -= epsilonE_mean
    result = {
        'bata_hat_l': [a_hat,aE_hat],
        'epsilon_mean_var': np.array([epsilon_mean,epsilon.var(),epsilonE_mean,epsilonE.var()]),
        'epsilon_l': [epsilon, epsilonE],
        'epsilon_list': [epsilon_list, epsilonE_list],
        'an_l': [an, anE],
        'sigma_l': [sigma_hat, sigmaE_hat],
        'ktheta': [A_hat, AE_hat],
        'lam_const_l': [lam, lamE],
    }
    return result


def learn_distribution_lasso(data_train,sigma_select,A,AE,lam_betahat,data_type=np.float16):
    """
    Directly mix unlabeled data to learn conditional distributions
    """
    z_train, zE_train, zU = data_train['z'], data_train['zE'], data_train['zU']
    X_train, XE_train, XU = data_train['X'], data_train['XE'], data_train['XU']

    n, p = z_train.shape
    ne, _ = zE_train.shape    

    if A :
        zU_A = np.concatenate([zU[i] for i in list(A)], axis=0)
        XU_A = np.concatenate([XU[i] for i in list(A)])
    else:
        zU_A = np.vstack(zU)
        XU_A = np.concatenate(XU)
    if AE :
        zU_AE = np.concatenate([zU[i] for i in list(A)], axis=0)
        XU_AE = np.concatenate([XU[i] for i in list(A)])
    else:
        zU_AE = np.vstack(zU)
        XU_AE = np.concatenate(XU)

    z_train = np.concatenate((z_train, zU_A),axis = 0)
    zE_train = np.concatenate((zE_train, zU_AE),axis = 0)
    X_train = np.concatenate((X_train, XU_A),axis = 0)
    XE_train = np.concatenate((XE_train, XU_AE),axis = 0)

    a_hat , lam_const, s = catulate_s(X_train, z_train,lam_betahat)
    n_all , _ = z_train.shape
    an = 0

    epsilon = X_train - z_train.dot(a_hat)
    sigma_hat = np.sum(epsilon**2)/n_all

    if ne != 0 :
        aE_hat , lamE_const, _ = catulate_s(XE_train, zE_train,lam_betahat)
        ne_all , _ = zE_train.shape
        anE = 0
        
        epsilonE = XE_train - zE_train.dot(aE_hat)
        sigmaE_hat = np.sum(epsilonE**2)/ne_all
    else: 
        aE_hat = np.zeros(p)
        lamE_const = 0
        sigmaE_hat = 1
        epsilonE = np.array([])
        anE = 1
    epsilon_mean = epsilon.mean()
    epsilonE_mean = epsilonE.mean()
    epsilon -= epsilon_mean
    epsilonE -= epsilonE_mean
    result = {
        'bata_hat_l': [a_hat,aE_hat],
        'sigma_l': [sigma_hat, sigmaE_hat],
        'ktheta': [[], []],
        'epsilon_mean_var': [epsilon_mean,epsilon.var(),epsilonE_mean,epsilonE.var()],
        'epsilon_l': [epsilon, epsilonE],
        'an_l': [an, anE],
        'lam_const_l': [lam_const, lamE_const],
    }
    return result

def generate_X0(data_stat,result,m,data_type=np.float16):
    """
    Generate synthetic X0 data based on estimated parameters.

    This function generates synthetic samples X00 and X0E for source and external domains
    respectively, using the estimated coefficients and noise variances from the learned
    conditional distributions. The generation assumes Gaussian noise with estimated variance.

    Parameters
    ----------
    data_stat : dict
        Dictionary containing statistical evaluation set data with keys 'z' and 'zE'.
    result : dict
        Dictionary containing learned distribution parameters with keys 'bata_hat_l' and 'sigma_l'.
    m : int
        Number of synthetic samples to generate per original sample.
    data_type : dtype, default=np.float16
        Data type for generated arrays.

    Returns
    -------
    data_stat : dict
        Updated dictionary with added keys 'X00' and 'X0E' containing generated synthetic data.
    """
    z, zE= data_stat['z'], data_stat['zE']
    a_hat, aE_hat = result['bata_hat_l']
    sigma_hat, sigmaE_hat = result['sigma_l']
    n , _ = z.shape
    ne , _ = zE.shape
    X0_eps = np.random.normal(0,np.sqrt(sigma_hat),m*n).astype(data_type).reshape((m,n))
    X0E_eps = np.random.normal(0,np.sqrt(sigmaE_hat),m*ne).astype(data_type).reshape((m,ne)) if ne != 0 else np.empty((m,0))
    X00 = X0_eps + np.sum(z*a_hat, axis=1)
    X0E = X0E_eps + np.sum(zE*aE_hat, axis=1)
    data_stat['X00'] = X00
    data_stat['X0E'] = X0E
    return data_stat

def generate_X0_bootstrap(data_stat,result,m,data_type=np.float16):
    """
    Generate synthetic X0 data using bootstrap resampling of residuals.

    This function generates synthetic samples X00 and X0E for source and external domains
    respectively using a bootstrap approach. It resamples residuals from the learned
    conditional distributions and adds Gaussian noise scaled by the estimated standard deviations.

    Parameters
    ----------
    data_stat : dict
        Dictionary containing statistical evaluation set data with keys 'z' and 'zE'.
    result : dict
        Dictionary containing learned distribution parameters with keys 'bata_hat_l',
        'epsilon_l', and 'an_l'.
    m : int
        Number of synthetic samples to generate per original sample.
    data_type : dtype, default=np.float16
        Data type for generated arrays.

    Returns
    -------
    data_stat : dict
        Updated dictionary with added keys 'X00' and 'X0E' containing generated synthetic data.
    """
    z, zE= data_stat['z'], data_stat['zE']
    a_hat, aE_hat = result['bata_hat_l']
    epsilon, epsilonE = result['epsilon_l']
    an, anE = result['an_l']

    n , _ = z.shape
    ne , _ = zE.shape
    epsilon_X0 = np.random.choice(epsilon, size=m*n, replace=True).reshape((m,n))
    X0_eps = np.random.normal(0,1,m*n).astype(data_type).reshape((m,n))
    X00 = epsilon_X0 + an*X0_eps + np.sum(z*a_hat, axis=1)
    # X00 = epsilon_X0 + 0*an*X0_eps + np.sum(z*a_hat, axis=1)  ## No smoothing terms are used.

    epsilon_XE0 = np.random.choice(epsilonE, size=m*ne, replace=True).reshape((m,ne)) if ne != 0 else np.empty((m,0))
    X0E_eps = np.random.normal(0,1,m*ne).astype(data_type).reshape((m,ne)) if ne != 0 else np.empty((m,0))
    X0E = epsilon_XE0 + anE*X0E_eps + np.sum(zE*aE_hat, axis=1)
    # X0E = epsilon_XE0 + 0*anE*X0E_eps + np.sum(zE*aE_hat, axis=1)  ## No smoothing terms are used.

    data_stat['X00'] = X00
    data_stat['X0E'] = X0E
    return data_stat

def Df_CRT (data_stat, result_part, m, lam_const, is_pool, data_type=np.float16):
    """
    Perform Distribution-free Conditional Randomization Test.

    This function computes test statistics and p-values for the CRT procedure using
    the learned conditional distribution and generated pseudo-data.

    Parameters
    ----------
    data_stat : dict
        Dictionary containing statistic calculation set data with keys:
        'z', 'X', 'Y', 'zE', 'XE', 'YE', 'X00', 'X0E'.
    result_part : dict
        Dictionary containing intermediate results from previous steps.
        Expected to have key 'bata_hat_l' which contains learned coefficients:
        - 'bata_hat_l' : tuple (a_hat, aE_hat) or list
            Learned coefficients for source and external domains respectively.
    m : int
        Number of pseudo-samples generated for each original sample.
    lam_const : float
        Regularization parameter lambda for the constant term in the CRT* statistic.
    is_pool : bool
        Whether to combine source and external domain data when computing the test statistic.
        If True, pools all data together and computes a single statistic; 
        If False, computes statistics separately for each domain and combines them using a weighted approach.
    data_type : numpy.dtype, optional (default=np.float16)
        Data type to use for storing statistics to save memory.

    Returns
    -------
    result_part : dict
        Updated result dictionary with additional keys:
        - 'P' : list of p-values
            P-values returned in the following order:
            - Optimal weighting
            - Using only internal data
            All two p-values are returned when is_pool=False.
            When is_pool=True, returns a single p-value from the pooled data.
        - 'lam_const0_l' : list
            Regularization parameters used for each domain.
        - 'stat' : numpy.ndarray
            Test statistics array. Shape (2, 1+m) when is_pool=False (source and external),
            or (1, 1+m) when is_pool=True. First column contains the observed statistic,
            remaining columns contain pseudo-statistics.
        - 'W' : float or int
            Weight parameter for combining p-values. Non-zero only when is_pool=False
            and both domains have data.
        
        The 'epsilon_l' key is removed from result_part before returning.
    
    Notes
    -----
    The function handles two cases:
    1. is_pool=False: Computes statistics separately for source and external domains,
       then combines p-values using a weighted approach via calculate_P.
    2. is_pool=True: Pools all data together and computes a single test statistic.
    
    The function also estimates weights W when combining statistics from multiple domains.
    """
    z, zE = data_stat['z'], data_stat['zE']
    X, XE = data_stat['X'], data_stat['XE']
    Y, YE = data_stat['Y'], data_stat['YE']
    X00, X0E = data_stat['X00'], data_stat['X0E']
    n , _ = z.shape
    ne , _ = zE.shape
    if not is_pool: 
        a_hat, aE_hat = result_part['bata_hat_l']
        stat , lam_const_in, stat0, l  = dCRT_statistics(X, X00, Y, z ,lam_const, a_hat)

        if ne == 0:
            statE0 = np.zeros(m)
            statE = 0
            lam_const_E = 0
            lE = None
            W = 0
        else:
            statE , lam_const_E, statE0, lE = dCRT_statistics(XE,X0E, YE, zE ,lam_const, aE_hat)
            W = Estimate_W_hat(n, ne, l, lE)
        lam_const0_l = [lam_const_in,lam_const_E]
        stat_all = np.array([[stat,*stat0.astype(data_type)],[statE,*statE0.astype(data_type)]])
        P_local = calculate_P(n,ne,m,stat,statE,stat0,statE0,W)
    else:
        XXE = X if ne == 0 else np.concatenate((X, XE))
        YYE = Y if ne == 0 else np.concatenate((Y, YE))
        zzE = z if ne == 0 else np.concatenate((z, zE),axis = 0)
        X00X0E = X00 if ne == 0 else np.concatenate((X00, X0E),axis = 1)
        stat_m , lam_const_in_m, stat0_m, l = dCRT_statistics(XXE, X00X0E, YYE, zzE ,lam_const)
        lam_const0_l = [lam_const_in_m]
        stat_all = np.array([[stat_m,*stat0_m.astype(data_type)]])
        W = 0
        P_local = [calculate_P(n+ne,0,m,stat_m,None,stat0_m,None,None)[0]]

    result = {
        'P': P_local,
        'lam_const0_l': lam_const0_l,
        'stat' : stat_all,
        'W' : W,
    }
    result_part.update(result)
    del result_part['epsilon_l']
    return result_part

def calculate_P (n,ne,m,stat,statE,stat0,statE0,W):
    if ne == 0:
        rho_W = stat
        rho0_W = stat0
    else:
        rho_W = (1-W)*stat+W*(statE) 
        rho0_W = (1- W)*stat0+W*(statE0)  

    pv0 = (1+np.sum(np.abs(stat0)>=np.abs(stat)))/(1+m)
    pvW = (1+np.sum(np.abs(rho0_W)>=np.abs(rho_W)))/(1+m)

    # pv0 = (1+np.sum((stat0)>=(stat)))/(1+m)
    # pvW = (1+np.sum((rho0_W)>=(rho_W)))/(1+m)
    
    result_pv = np.stack([pvW,pv0], axis=0)
    return result_pv

def Estimate_W_hat(n, ne, l, lE):
    return l/(n/ne*lE+l)

def Oracle_betahat(X,XU_A,z,zU_A,lam_betahat = None):
    n , p = z.shape
    nu , _ = zU_A.shape
    if n != 0:
        if nu == 0:
            a_hat , lam_const = fit_lasso_model(z, X,lam_betahat)
        else:
            aU_hat , lam_const = fit_lasso_model(zU_A, XU_A,lam_betahat)
            lasso_model = Lasso(alpha=lam_const*np.sqrt(2*np.log(p)/n),fit_intercept=False)
            lasso_model.fit(z, X-np.sum(z*aU_hat,axis=1))
            a_hat = aU_hat+lasso_model.coef_
    else:
        a_hat = np.zeros(p)
        lam_const = None
    return a_hat ,lam_const

def generate_unique_index_sets_ordered(Rhat):
    ranks = rankdata(Rhat, method='average')
    kk_list = np.unique(ranks)
    
    Tset = [[]]
    seen = {}
    for kk in kk_list:
        indices = tuple(np.where(ranks <= kk)[0].tolist())
        if indices not in seen:
            seen[indices] = True
            Tset.append(list(indices))
    
    return Tset

def agg_fun(B, X_test, y_test, total_step=10, selection=False):
    B = np.array(B)
    if np.all(B == 0):
        return {'theta': None, 'beta': np.zeros(X_test.shape[1]), 'beta_ew': None}
    K, _ = B.shape
    errors = np.sum((y_test - np.dot(B,X_test.T)) ** 2, axis=1)
    khat = np.argmin(errors)
    if selection:
        theta_hat = np.zeros(K)
        theta_hat[khat] = 1
        beta = B[khat,:]
        beta_ew = None
    else:
        theta_hat = np.exp(-errors / 2)
        theta_hat /= np.sum(theta_hat)
        theta_old = theta_hat.copy()
        beta = np.dot(theta_hat,B)
        beta_ew = beta.copy()

        for step in range(total_step):
            residuals = np.dot(X_test , beta) - np.dot(B,X_test.T)
            residuals_sq = np.sum(residuals ** 2, axis=1)
            theta_hat = np.exp(-errors / 2 + residuals_sq / 8)
            theta_hat /= np.sum(theta_hat)
            beta_new = np.dot(theta_hat,B) * 0.25 + 0.75 * beta
            if np.sum(np.abs(theta_hat - theta_old)) < 1e-3:
                beta = beta_new
                break
            beta = beta_new
            theta_old = theta_hat.copy()
        beta_ew = beta.copy()
    return {'theta': theta_hat, 'beta': beta, 'beta_ew': beta_ew, 'khat': khat}

def Trans_Lasso(X0,XU,z0,zU,lam_betahat):
    n_all , p = z0.shape
    X, X_til, z, z_til = train_test_split(X0, z0, test_size=1/3, random_state=42)
    n_only , _ = z.shape

    Xtz = []
    for xu_i in range(len(XU)):    
        res = np.dot(zU[xu_i].T, XU[xu_i])/len(XU[xu_i]) - np.dot(z.T, X)/n_only
        Xtz.append(res)
    Xtz = np.vstack(Xtz)

    t_star = round(1/3*n_only)
    Xtz_part = np.sort(np.abs(Xtz), axis=1)[:, -t_star:]
    max_vals = np.max(np.abs(Xtz_part), keepdims=True)
    scaled = Xtz_part.astype(np.float64)  / (max_vals + 1e-100)
    Rhat = np.sum(scaled**2, axis=1) 

    Gl_list = generate_unique_index_sets_ordered(Rhat)
    beta_list = []
    seen = set()
    for gl in Gl_list:
        zU_A = np.concatenate([zU[i] for i in list(gl)], axis=0) if gl else np.empty((0, p))
        XU_A = np.concatenate([XU[i] for i in list(gl)]) if gl else np.empty((0,))
        beta_hat , _ =  Oracle_betahat(X,XU_A,z,zU_A,lam_betahat)
        beta_tuple = tuple(beta_hat)
        if beta_tuple not in seen:
            seen.add(beta_tuple)
            beta_list.append(list(beta_hat))
    beta_agg = agg_fun(beta_list, z_til , X_til)
    a_hat = beta_agg['beta_ew']
    k_hat = beta_agg['khat']
    A_hat_k = Gl_list[k_hat]
    epsilon_all = X0 - z0.dot(a_hat)
    return a_hat , A_hat_k, epsilon_all, n_all

def catulate_s(X, z, lam_betahat):
    if z.shape[0] == 0:
        return None, None, -1
    a , lam_const = fit_lasso_model(z, X,lam_betahat)
    return a, lam_const, np.sum(a!=0)

def catulate_an(s,p,n_min,nu_all):
    return 0.8*(s*np.log(p)/(n_min+nu_all))**(1/4) if n_min != 0 else 1

def get_betahat(X_train,XU,z_train,zU,A,sigma_select,lam_betahat,data_type=np.float16):
    n, p = z_train.shape

    nu_l = [zU[i].shape[0] for i in list(A) if zU[i].shape[0] != 0]
    n_min = min(nu_l + [n])
    nu_all = np.sum(nu_l) if nu_l else 0

    _, _, s = catulate_s(X_train, z_train, lam_betahat)
    sU = []
    for i in A:
        _, _, su = catulate_s(XU[i], zU[i], lam_betahat)
        if su != -1:
            sU.append(su)
    s_max= np.max(np.array(sU)) if sU else -1
    s = max(s,s_max)
    an = catulate_an(s,p,n_min,nu_all)

    zU_A = np.concatenate([zU[i] for i in list(A)], axis=0)
    XU_A = np.concatenate([XU[i] for i in list(A)])
    a_hat, lam = Oracle_betahat(X_train, XU_A, z_train, zU_A,lam_betahat = lam_betahat)

    epsilon = X_train - z_train.dot(a_hat)
    epsilon_list = [epsilon]
    if sigma_select == '1':
        sigma_hat = 1
        an = 1
    else:
        sigma_hat = np.sum(epsilon**2)
        n_nu = n
        for i in A:
            if n != 0 :
                X_train_w, z_train_w = XU[i], zU[i]
                zU_A_w = np.concatenate([zU[j] for j in list(A) if j != i] + [z_train], axis=0)
                XU_A_w = np.concatenate([XU[j] for j in list(A) if j != i] + [X_train])
                a_w_hat, _ = Oracle_betahat(X_train_w, XU_A_w, z_train_w, zU_A_w,lam_betahat = lam_betahat)
                epsilon_u = X_train_w - z_train_w.dot(a_w_hat)
                epsilon_list.append(epsilon_u)
                sigma_hat += np.sum(epsilon_u**2)
                epsilon = np.concatenate([epsilon, epsilon_u])
        n_nu += nu_all
        sigma_hat /= n_nu
    return a_hat, sigma_hat, epsilon, an, epsilon_list, lam

def get_betahat_unknown(X,XU,z,zU,sigma_select,lam_betahat = None):
    n , p = z.shape
    n_max = max([zU[i].shape[0] for i in range(len(zU))])
    A_hat = []
    if n != 0:
        if n_max == 0:
            a_hat, _, _ = catulate_s(X, z, lam_betahat)
            an = 0
            epsilon = X - z.dot(a_hat)
            sigma_hat = np.sum(epsilon**2)/n
            epsilon_list = [epsilon]
        else:
            a_hat , A_hat, epsilon_all_0, n_all_0 = Trans_Lasso(X,XU,z,zU,lam_betahat)
            sigma_hat , epsilon, an, epsilon_list = get_sigmahat(X,XU,z,zU,A_hat,epsilon_all_0, n_all_0, sigma_select,lam_betahat)
    else:
        an = 1
        a_hat = np.zeros(p)
        sigma_hat = 1
        epsilon = np.array([])

    return np.array(a_hat), sigma_hat, epsilon, an, epsilon_list, A_hat

def get_sigmahat(X,XU,z,zU,A_hat,epsilon_all, n_all, sigma_select,lam_betahat=None):
    n, p = z.shape

    nu_l = [zU[i].shape[0] for i in list(A_hat) if zU[i].shape[0] != 0]
    n_min = min(nu_l + [n])
    nu_all = np.sum(nu_l) if nu_l else 0

    _, _, s = catulate_s(X, z, lam_betahat)
    sU = []
    for i in A_hat:
        _, _, su = catulate_s(XU[i], zU[i], lam_betahat)
        if su != -1:
            sU.append(su)
    s_max= np.max(np.array(sU)) if sU else -1
    s = max(s,s_max)
    an = catulate_an(s,p,n_min,nu_all)
    
    err_list = [epsilon_all]
    err_all = np.sum(epsilon_all**2)
    
    if sigma_select == '1':
        err_all = 1
        epsilon_all = np.array([0]*n_all,dtype=np.float64)
        err_list = [epsilon_all]
        an = 1
    else:
        for xu_hat_i in A_hat:
            x_i = XU[xu_hat_i]
            z_i = zU[xu_hat_i]
            zu_i = [zU[j] for j in list(A_hat) if j != xu_hat_i] + [z]
            xu_i = [XU[j] for j in list(A_hat) if j != xu_hat_i] + [X]
            _ , _, epsilon_all_i, n_all_i = Trans_Lasso(x_i,xu_i,z_i,zu_i,lam_betahat)
            err_list.append(epsilon_all_i)
            epsilon_all = np.concatenate([epsilon_all, epsilon_all_i])
            err_all_i = np.sum(epsilon_all_i**2)
            err_all += err_all_i
            n_all += n_all_i
        err_all /= n_all
    return err_all, epsilon_all, an, err_list

def get_learn_X0_function(learn_name, generate_X0_name):
    generate_X0_functions = {
        'sigma': generate_X0,
        'bootstrap': generate_X0_bootstrap,
    }
    generate_X0_function = generate_X0_functions.get(generate_X0_name)
    learn_functions = {
        'Trans_lasso': learn_distribution,
        'pool_lasso': learn_distribution_lasso,
    }
    learn_function = learn_functions.get(learn_name)
    return learn_function, generate_X0_function

def fit_lasso_model(X, y, lam_const=None):
    """
    Perform Lasso regression of y on X
    """
    X = X.copy()
    y = y.copy()
    n, p = X.shape
    if isinstance(lam_const, str):
        lasso_model = LassoCV(cv=4, fit_intercept=False, random_state=42, n_alphas = int(lam_const))
    elif np.isscalar(lam_const):
        lasso_model = Lasso(alpha=lam_const*np.sqrt(2*np.log(p)/n), fit_intercept=False)
    else:
        lasso_model = LassoCV(cv=4, alphas=np.atleast_1d(lam_const)*np.sqrt(2*np.log(p)/n), fit_intercept=False, random_state=42)

    lasso_model.fit(X, y)
    return lasso_model.coef_, lasso_model.alpha_/(np.sqrt(2*np.log(p)/(n))) if hasattr(lasso_model, 'alpha_') else lam_const/(np.sqrt(2*np.log(p)/(n)))

def dCRT_statistics (X, X0, Y, z ,lam_const, coef = None):
    """
    Calculate dCRT statistic
    """
    n , _ = z.shape
    coefy , _ = fit_lasso_model(z,Y,lam_const)
    dy = Y-np.dot(z,coefy)
    coef , lam_const1 = fit_lasso_model(z,X,lam_const)
    dx = X-np.dot(z,coef)
    stat = np.mean(dx*dy)
    if X0 is None:
        stat0 = 0
    else:
        m, _ = X0.shape
        dx0 = np.zeros((m,n))
        lam_const0 = np.zeros(m)
        for mm in range(m):
            coef0 , lam_const0[mm] = fit_lasso_model(z,X0[mm],lam_const1)
            dx0[mm] = X0[mm]-np.dot(z,coef0)
        stat0 = np.dot(dx0,dy)/n
    dy_norm = np.mean(dy**2)
    return stat , lam_const1, stat0, dy_norm

def save_output_hdf5(output_dir, params, data_dict):
    os.makedirs(output_dir, exist_ok=True)
    pickle_items = ['epsilon_list', 'A_hat_lists']
    
    for item in pickle_items:
        if item in data_dict:
            with open(os.path.join(output_dir, f"{item}.pkl"), 'wb') as f:
                pickle.dump(data_dict[item], f)
    hdf5_file_path = os.path.join(output_dir, f'output.h5')
    with h5py.File(hdf5_file_path, 'w') as h5f:
        for key, value in data_dict.items():
            if key in pickle_items:
                continue
            try:
                if isinstance(value, (np.ndarray, list)):
                    value_array = np.array(value, dtype=object) if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list) else np.array(value)
                    h5f.create_dataset(key, data=value_array)
                else:
                    h5f.create_dataset(key, data=value)
            except Exception as e:
                with open(os.path.join(output_dir, f"{key}.pkl"), 'wb') as f:
                    pickle.dump(value, f)
    
    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        f.write('{\n')
        num_items = len(params)
        for i, (key, value) in enumerate(params.items()):
            if isinstance(value, list):
                f.write(f'    "{key}": {json.dumps(value)}')
            else:
                f.write(f'    "{key}": {json.dumps(value)}')
            if i < num_items - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write('}\n')

def save_out(result, path_out):    
    plt.hist(result['stat'][0,1:], bins=10, edgecolor='black')
    mean_value = np.mean(result['stat'][0,1:])
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    plt.title(f"Mean_in = {mean_value:.5f}")
    plt.savefig(os.path.join(path_out, "pseudo_sample_statistics_in.png"), bbox_inches='tight')
    plt.close()
    if(result['stat'].shape[0]>1):
        plt.hist(result['stat'][1,1:], bins=10, edgecolor='black')
        mean_value = np.mean(result['stat'][1,1:])
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
        plt.title(f"Mean_E = {mean_value:.5f}")
        plt.savefig(os.path.join(path_out, "pseudo_sample_statistics_E.png"), bbox_inches='tight')
        plt.close()
    
    with open(os.path.join(path_out,'result_output.txt'), "w", encoding="utf-8") as f:
        print('-----------------Results-------------------', file=f)
        print('P-values: ', result['P'][0], '   ------optimal weight | combined', file=f)
        print('Wop weight: ', result['W'], file=f)
        print('Statistics: ', result['stat'][:,0], '   ------internal, external | combined', file=f)
        print('Hyperparameters: ', result['lam_const0_l'], '   ------internal, external | combined', file=f)
        print('-------------------------------------------', file=f)
        print('X-Z distribution (internal): ', np.round(result['bata_hat_l'][0],2), file=f)
        print('-------------------------------------------', file=f)
        print('X-Z distribution (external): ', np.round(result['bata_hat_l'][1],2), file=f)
        print('-------------------------------------------', file=f)
        print('X-Z variance: ', result['sigma_l'], '   ------internal, external', file=f)
        print('Selected unlabeled set (internal): ', result['ktheta'][0], file=f)
        print('Selected unlabeled set (external): ', result['ktheta'][1], file=f)
    print(f"Results saved to {path_out}")