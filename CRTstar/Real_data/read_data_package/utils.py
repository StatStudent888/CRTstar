import numpy as np
import pandas as pd
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_ROOT = PACKAGE_ROOT / 'data'

def read_data(call_name, p):
    folder_i = DATA_ROOT / 'Internal_data'
    folder_E = DATA_ROOT / 'External_data'
    folder_U = DATA_ROOT / 'Unlabel_data'
    U_list = ['Basal','Her2','LumA','LumB','Normal']
    Y = pd.read_csv(folder_i / f'InternalD_response_scaled.csv')
    Xz = pd.read_csv(folder_i / f'InternalD_covariate_scaled_p{p}.csv')
    YXz = pd.merge(Y, Xz, on=call_name, how='inner')
    YE = pd.read_csv(folder_E / f'ExternalD_response_scaled.csv')
    XzE = pd.read_csv(folder_E / f'ExternalD_covariate_scaled_p{p}.csv')
    YXzE = pd.merge(YE, XzE, on=call_name, how='inner')
    XzU = []
    for U_name in U_list:
        Xzu = pd.read_csv(folder_U / f'UnlabelD_covariate_{U_name}_scaled_p{p}.csv')
        XzU.append(Xzu)
    return YXz, YXzE, XzU

def data_load(call_name, Y_name, X_name, p):
    YXz, YXzE, XzU = read_data(call_name, p)
    XU, zU = [], []
    for Xzu in XzU:
        XU.append(Xzu[X_name].values)
        zU.append(Xzu.drop(columns=[call_name, X_name]).values)
    data = {}
    data['Y'], data['YE'] = YXz[Y_name].values, YXzE[Y_name].values
    data['X'], data['XE'], data['XU'] = YXz[X_name].values, YXzE[X_name].values, XU
    data['z'], data['zE'], data['zU'] = YXz.drop(columns=[call_name, Y_name, X_name]).values, YXzE.drop(columns=[call_name, Y_name, X_name]).values, zU
    data['Y_name'], data['X_name'], data['z_name'] = Y_name, X_name, YXz.drop(columns=[call_name, Y_name, X_name]).columns
    return data