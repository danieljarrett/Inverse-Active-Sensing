import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import List

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
