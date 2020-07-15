# NOTE: Consolidate DecisionMDP/DiagnosisMDP

from .mutable_mdp import *

class DecisionMDP(MutableMDP):
    def __init__(self,
        pomdp : BasePOMDP  ,
        eta_d : List[float],
    ):
        self.eta_d = np.array(eta_d)

        super(DecisionMDP, self).__init__(pomdp)

    def reward(self,
        prev_belief : np.ndarray,
        belief      : np.ndarray,
    ) -> float:
        m = (belief - prev_belief) ** 2
        r = prev_belief * (1 - prev_belief) - belief * (1 - belief)

        psi = r

        return self.eta_d.dot(psi)

    def configure_reward(self,
        eta_d : np.ndarray,
    ):
        self.store_reward()

        self.eta_d = eta_d
