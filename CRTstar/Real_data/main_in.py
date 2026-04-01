import sys
import os

from read_data_package.utils import *
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from MYCRT_package.utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = 318
    p = 200

    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, 'output', f'{CRT}_{learn_name}_{sigma_select}_{is_pool}_{seed}')
    
    data = data_load('Unnamed: 0','BRCA1P1','BRCA1',p)

    result = DF_CRT_all(data, seed, params, path)