from .hypothesis_pomdp import *
from .mutable_pomdp import *

class MutableHypothesisPOMDP(HypothesisPOMDP, MutablePOMDP):
    def __init__(self,
        p_val : np.ndarray ,
        q_val : np.ndarray ,
        eta_a : List[float],
        eta_b : List[float],
        eta_c : List[float],
        costs : List[float],
    ):
        self.dim_eta_a = len(eta_a)
        self.dim_eta_b = len(eta_b)
        self.dim_eta_c = len(eta_c)

        self.dim_abc = self.dim_eta_a \
                     + self.dim_eta_b \
                     + self.dim_eta_c

        super(MutableHypothesisPOMDP, self).__init__(
            p_val,
            q_val,
            eta_a,
            eta_b,
            eta_c,
            costs,
        )

    def configure_reward(self,
        eta : np.ndarray,
    ):
        self.store_reward()

        a = 0
        b = a + self.dim_eta_a
        c = b + self.dim_eta_b
        d = c + self.dim_eta_c

        eta_a = eta[a:b]
        eta_b = eta[b:c]
        eta_c = eta[c:d]

        super(MutableHypothesisPOMDP, self).__init__(
            self.p_val,
            self.q_val,
            eta_a     ,
            eta_b     ,
            eta_c     ,
            self.costs,
        )
