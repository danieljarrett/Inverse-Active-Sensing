# NOTE: Consolidate DecisionMDP/DiagnosisMDP

from .mutable_mdp import *

class DiagnosisMDP(MutableMDP):
    def __init__(self,
        pomdp : BasePOMDP  ,
        omega : List[float],
    ):
        self.omega = np.array(omega)

        super(DiagnosisMDP, self).__init__(pomdp)

    def reward(self,
        prev_belief : np.ndarray,
        belief      : np.ndarray,
    ) -> float:
        m = (belief - prev_belief) ** 2
        r = prev_belief * (1 - prev_belief) - belief * (1 - belief)

        psi = r

        return self.omega.dot(psi)

    def configure_reward(self,
        omega : np.ndarray,
    ):
        self.store_reward()

        self.omega = omega
