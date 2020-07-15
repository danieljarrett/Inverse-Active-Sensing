from .base_pomdp import *

class MutablePOMDP(BasePOMDP):
    def configure_reward(self,
        theta : np.ndarray,
    ):
        raise NotImplementedError

    def store_reward(self):
        self.reward_backup = np.copy(self.reward_matrix)

    def restore_reward(self):
        self.reward_matrix = self.reward_backup
