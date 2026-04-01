import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from MYCRT_package.utils import *

def Split_the_data_realerr(seed,CRT,data,k_in,k_E):
    z0, zE0, zU = data['z'], data['zE'], data['zU']
    X0, XE0, XU = data['X'], data['XE'], data['XU']
    Y0, YE0 = data['Y'], data['YE']

    X_eps, XE_eps, XU_all_eps = data['X_eps'], data['XE_eps'], data['XU_all_eps']

    n , p = z0.shape
    ne , _ = zE0.shape
    if CRT == 'out':
        X_train, X, _, Y, z_train, z, X_eps_train, _ = train_test_split(X0, Y0, z0, X_eps, test_size=k_in, random_state=42)
        if ne == 0:
            XE = YE = XE_train = XE_eps_train = np.empty((0,))
            zE = zE_train = np.empty((0, p))
        else:
            XE_train, XE, _, YE, zE_train, zE, XE_eps_train, _ = train_test_split(XE0, YE0, zE0, XE_eps, test_size=k_E, random_state=42)
        data_train = {'z': z_train ,'X': X_train ,'zE': zE_train ,'XE': XE_train ,'zU': zU ,'XU': XU, 'X_eps': X_eps_train, 'XE_eps' : XE_eps_train, 'XU_eps' : XU_all_eps}
        data_stat = {'z': z ,'X': X ,'Y': Y ,'zE': zE ,'XE': XE ,'YE': YE}
    else:
        data_train = {'z': z0 ,'X': X0 ,'zE': zE0 ,'XE': XE0 ,'zU': zU ,'XU': XU, 'X_eps': X_eps, 'XE_eps' : XE_eps, 'XU_eps' : XU_all_eps}
        data_stat = {'z': z0 ,'X': X0 ,'Y': Y0 ,'zE': zE0 ,'XE': XE0 ,'YE': YE0}
    return data_train, data_stat

def DF_CRT_all_realerr(data, seed=None, params=None, path=None, show=True):
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

    data_train, data_stat = Split_the_data_realerr(seed, CRT, data, k_in, k_E)
    if show: print('Data splitting completed!')
    
    result = learn_function(data_train, sigma_select, A, AE, lam_betahat)
    if show: print('Conditional distribution learning completed!')  
    
    X_eps, XE_eps, XU_eps = data_train['X_eps'], data_train['XE_eps'], data_train['XU_eps']
    epsilon = np.concatenate((X_eps, XU_eps))
    epsilonE = np.concatenate((XE_eps, XU_eps))
    epsilon_mean = epsilon.mean()
    epsilonE_mean = epsilonE.mean()
    epsilon -= epsilon_mean
    epsilonE -= epsilonE_mean
    result['epsilon_l'] = [epsilon, epsilonE]
    result['epsilon_mean_var'] = [epsilon_mean,epsilon.var(),epsilonE_mean,epsilonE.var()]

    data_stat = generate_X0_function(data_stat, result, m)
    if show: print('Pseudo-data generation completed!')  

    result = Df_CRT(data_stat, result, m, lam_const, is_pool)
    if show: print('Df_CRT completed!')  
    
    if path:
        save_output_hdf5(path, params, result)
        save_out(result, path)
    return result