from .dp_agent import *
from .mutable_agent import *

class MutableDPAgent(DPAgent, MutableAgent):
    def store_value_function(self):
        self.alphas_backup = np.copy(self.alphas)

    def restore_value_function(self):
        self.alphas = self.alphas_backup
