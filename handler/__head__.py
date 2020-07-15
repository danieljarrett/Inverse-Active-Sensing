import numpy as np
import glob
import os

from typing import Dict, List, Tuple
from pomdp_gym import BasePOMDP

def ary2str(
    ary : np.ndarray,
) -> str:
    return ' '.join([str(e) for e in ary])

def str2ary(
    str : str,
) -> np.ndarray:
    return np.array(list(map(float, str.split(' '))))

def get_latest(
    dir : str      ,
    ext : List[str],
):
    files = [e for batch in [glob.glob(dir + '/*.' + x) for x in ext] for e in batch]

    return max(files, key = os.path.getctime)
