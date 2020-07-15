from .disease_pomdp import *
from .mutable_pomdp import *

class MutableDiseasePOMDP(DiseasePOMDP, MutablePOMDP):
    def __init__(self,
        tree_factor : int        ,
        tree_height : int        ,
        prob_comp   : List[float],
        prob_fail   : List[float],
        phi_test    : int        ,
        phi_comp    : int        ,
        phi_mark    : int        ,
        theta_test  : List[float],
        theta_comp  : List[float],
        theta_mark  : List[float],
    ):
        self.phi_test = phi_test
        self.phi_comp = phi_comp
        self.phi_mark = phi_mark

        self.dim_theta_test = len(theta_test)
        self.dim_theta_comp = len(theta_comp)
        self.dim_theta_mark = len(theta_mark)

        self.dim_theta = self.dim_theta_test \
                       + self.dim_theta_comp \
                       + self.dim_theta_mark

        super(MutableDiseasePOMDP, self).__init__(
            tree_factor,
            tree_height,
            prob_comp  ,
            prob_fail  ,
            phi_test   ,
            phi_comp   ,
            phi_mark   ,
            theta_test ,
            theta_comp ,
            theta_mark ,
        )

    def configure_reward(self,
        theta : np.ndarray,
    ):
        self.store_reward()

        a = 0
        b = a + self.dim_theta_test
        c = b + self.dim_theta_comp
        d = c + self.dim_theta_mark

        theta_test = theta[a:b]
        theta_comp = theta[b:c]
        theta_mark = theta[c:d]

        super(MutableDiseasePOMDP, self).__init__(
            self.tree_factor,
            self.tree_height,
            self.prob_comp  ,
            self.prob_fail  ,
            self.phi_test   ,
            self.phi_comp   ,
            self.phi_mark   ,
            theta_test ,
            theta_comp ,
            theta_mark ,
        )
