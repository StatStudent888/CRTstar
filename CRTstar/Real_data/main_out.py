import numpy as np
import pandas as pd
import sys
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm import trange
from read_data_package.utils import *
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from MYCRT_package.utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")

def process_single_id(id, data, params):
    result = DF_CRT_all(data, id, params, None, False)
    P_value = result['P']
    return {
        'id': id, 
        'P_op': P_value[0],
        'P_in': P_value[1]
    }

def parallel_process_with_progress(function, iterable, fixed_args=None, n_jobs=8, **kwargs):
    iterator = iter(iterable)
    total = len(iterable) if hasattr(iterable, '__len__') else None
    
    if fixed_args:
        results = Parallel(n_jobs=n_jobs)(
            delayed(function)(item, *fixed_args) for item in tqdm(iterator, total=total, **kwargs)
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(function)(item) for item in tqdm(iterator, total=total, **kwargs)
        )
    return results

if __name__ == "__main__":
    data = data_load('Unnamed: 0', 'BRCA1P1', 'BRCA1', 200)
    id_l = [47, 115, 117, 119, 129, 130, 136, 140, 142, 151, 156, 166, 173,182, 184, 195, 201, 216, 239, 254, 259, 287, 289, 308, 319, 340,342, 346, 355, 357, 364, 379, 387, 391, 405, 407, 413, 451, 452,456, 458, 466, 472, 480, 486, 496, 497, 504, 514, 542, 558, 566,583, 596, 604, 625, 653, 655, 658, 660, 670, 693, 694, 702, 710,725, 745, 746, 747, 751, 760, 761, 762, 770, 775, 787, 796, 801,806, 809, 824, 831, 835, 836, 849, 851, 852, 855, 866, 871, 886,897, 899, 917, 923, 943, 953, 969, 970, 976]
    results = parallel_process_with_progress(
        process_single_id, 
        id_l, 
        fixed_args=(data, params),
        n_jobs=8
    )
    results_df = pd.DataFrame(results)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, 'output', 'results.csv')
    results_df.to_csv(path, index=False)