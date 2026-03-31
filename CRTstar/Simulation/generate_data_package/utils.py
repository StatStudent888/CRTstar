import numpy as np
import warnings
warnings.filterwarnings("ignore")

def generate_XYZ(seed,c,cE,a,aE,aU,n,p,nE,nu,A,AE,N_un,sigz,K,model_y='l',model_yE='l',data_type=np.float16):
    """
    Generate synthetic data for multi-source learning with latent variables.
 
    This function generates labeled data (X, Y) from source domains, external labeled data (XE, YE)
    from auxiliary domains, and unlabeled data (XU) from multiple domains. The data generation follows a latent factor model where observations are
    linear combinations of latent factors plus noise.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    a : array-like, shape (2, p)
        Coefficient matrix for the source domain: a[0,:] for X, a[1,:] for Y.
    aE : array-like, shape (2, p)
        Coefficient matrix for the external domain: aE[0,:] for XE, aE[1,:] for YE.
    aU : list of array-like, length K
        Coefficient vectors for each unlabeled domain, where aU[i] has shape (p,).
    n : int
        Number of source domain samples.
    p : int
        Dimension of latent variable Z.
    nE : int
        Number of external labeled samples. If 0, no external data is generated.
    nu : int
        Number of unlabeled samples per domain in the union set (A ∪ AE).
    A : set or list
        Indices of domains for which unlabeled data is generated with nu samples each.
    AE : set or list
        Additional domain indices for unlabeled data generation (union with A).
    N_un : int
        Number of unlabeled samples per domain not in A ∪ AE.
    sigz : array-like or str
        Covariance matrix for latent variables Z. If 'identity matrix', uses identity matrix.
    K : int
        Total number of unlabeled domains.
    model_y : {'l', 'nonlinear'}, default='l'
        Model type for source domain Y: 'l' for linear, 'nonlinear' for custom nonlinear function.
    model_yE : {'l', 'nonlinear'}, default='l'
        Model type for external domain YE: 'l' for linear, 'nonlinear' for custom nonlinear function.
    data_type : dtype, default=np.float16
        Data type for generated arrays (e.g., np.float16, np.float32, np.float64).

    Returns
    -------
    data : dict
        Dictionary containing generated data with keys:
        - 'z': Latent variables for source domain, shape (n, p)
        - 'X': Source domain features, shape (n,)
        - 'Y': Source domain labels, shape (n,)
        - 'zE': Latent variables for external domain, shape (nE, p) (empty if nE=0)
        - 'XE': External domain features, shape (nE,) (empty if nE=0)
        - 'YE': External domain labels, shape (nE,) (empty if nE=0)
        - 'zU': List of latent variables for unlabeled domains, each shape (m, p) where m is nu or N_un
        - 'XU': List of unlabeled features for each domain, each shape (m,)
    """
    data = {}
    muz = np.zeros(p)
    if sigz == 'identity matrix':
        sigz = np.eye(p)
    np.random.seed(seed)

    z = np.random.multivariate_normal(muz,sigz,n).astype(data_type)
    X_eps = np.random.normal(0,1,n).astype(data_type)
    # X_eps = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=n).astype(data_type)  # Generate residuals using a uniform distribution.

    Y_eps = np.random.normal(0,1,n).astype(data_type)
    X = X_eps+np.dot(z,a[0,:]).astype(data_type)
    if model_y == 'l':
        Y = c*X+Y_eps+np.dot(z,a[1,:]).astype(data_type)
    else :
        Y = c*X+Y_eps+YI(z)  
    
    if nE == 0:
        XE = YE = np.empty((0,))
        zE = np.empty((0, p))
    else :
        zE = np.random.multivariate_normal(muz,sigz,nE).astype(data_type)
        XE_eps = np.random.normal(0,1,nE).astype(data_type)
        # XE_eps = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=nE).astype(data_type) # Generate residuals using a uniform distribution.

        YE_eps = np.random.normal(0,1,nE).astype(data_type)
        XE = XE_eps+np.dot(zE,aE[0,:]).astype(data_type)
        if model_yE == 'l':
            YE = cE*XE+YE_eps+np.dot(zE,aE[1,:]).astype(data_type)
        else :
            YE = cE*XE+YE_eps+YI(zE) 

    union = set(A) | set(AE)
    union_un = set(range(K)) - (set(A) | set(AE))
    A_num = len(union)
    K_A_num = len(union_un)
    if nu == 0 :
        XU_l = [np.empty(0) for _ in range(K)]
        zU_l = [np.empty((0, p)) for _ in range(K)]
    else:
        XU_l = []
        zU_l = []
        zU = np.random.multivariate_normal(muz,sigz,A_num*nu).astype(data_type).reshape((A_num,nu,-1))
        XU_eps = np.random.normal(0,1,A_num*nu).astype(data_type).reshape((A_num,nu))
        # XU_eps = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=A_num*nu).astype(data_type).reshape((A_num,nu)) # Generate residuals using a uniform distribution.

        zU_un = np.random.multivariate_normal(muz,sigz,K_A_num*N_un).astype(data_type).reshape((K_A_num,N_un,-1))
        XU_un_eps = np.random.normal(0,1,K_A_num*N_un).astype(data_type).reshape((K_A_num,N_un))
        # XU_eps = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=K_A_num*nu).astype(data_type).reshape((K_A_num,nu)) # Generate residuals using a uniform distribution.

        u_num = 0
        u_un_num = 0
        for i in range(K):
            if i in union:
                zU_l.append(zU[u_num])
                XU_l.append(XU_eps[u_num]+np.dot(zU[u_num],aU[i]).astype(data_type))
                u_num += 1
            else:
                zU_l.append(zU_un[u_un_num])
                XU_l.append(XU_un_eps[u_un_num]+np.dot(zU_un[u_un_num],aU[i]).astype(data_type))
                u_un_num += 1
    data = {'z': z ,'X': X ,'Y': Y ,'zE': zE ,'XE': XE ,'YE': YE ,'zU': zU_l ,'XU': XU_l}
    return data

def YI(z):
    if z.shape[-2] == 0:
        return np.empty_like(z, dtype=int)
    I = (z > 0).astype(int)
    I[..., 1] = (z[..., 1] > 0.5).astype(int)
    I[..., 2] = (z[..., 2] > -0.5).astype(int)
    I[..., 3] = (np.abs(z[..., 3]) > 1).astype(int)
    a = 0.4*(I[...,1]+I[...,2])+0.5*(I[...,0]+I[...,3]+I[...,0]*I[...,3]+I[...,1]*I[...,2])+0.15*(I[...,30]+I[...,31]+I[...,32]+I[...,33]+I[...,30]*I[...,31]+I[...,32]*I[...,33])
    return a