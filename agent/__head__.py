import gym
import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F

from typing import Dict, List, Tuple

STORE = 'volume/'
PNAME = 'decision'

def print_np(
    val : float,
    fmt : str  ,
):
    fmt_str = '{0:' + fmt + '}'
    np.set_printoptions(formatter = {'float': lambda x: fmt_str.format(x)})

    print(val)
