from .rho_belief_mdp import *

class MutableMDP(RhoBeliefMDP):
    def configure_reward(self,
        eta_d : np.ndarray,
    ):
        raise NotImplementedError

    def store_reward(self):
        self.eta_d_backup = np.copy(self.eta_d)

    def restore_reward(self):
        self.eta_d = self.eta_d_backup
