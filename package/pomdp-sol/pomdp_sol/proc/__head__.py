import subprocess
import sys
import os
import re

from typing import Dict, List

DIR = os.path.dirname(os.path.realpath(__file__))
SEP = '/'
SOL = 'nix-solver/src/pomdp-solve'
SAR = 'nix-sarsop/pomdpsol'
