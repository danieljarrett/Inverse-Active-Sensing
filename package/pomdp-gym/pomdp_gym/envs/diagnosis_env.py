from .__head__ import *

from .pomdp import DiagnosisPOMDP
from .mdp import DiagnosisMDP

class DiagnosisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

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
        omega       : List[float],
    ):
        pomdp = DiagnosisPOMDP(
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

        self.mdp = DiagnosisMDP(
            pomdp,
            omega,
        )

        self.action_space = spaces.Discrete(self.mdp.pomdp.num_actions)
        self.observation_space = np.zeros(self.mdp.pomdp.num_states)

    def step(self,
        action_index : int,
    ) -> (np.ndarray, int, bool, Dict):
        _, reward = self.mdp.step(action_index)
        belief    = self.mdp.belief
        done      = self.mdp.pomdp.done

        return belief, reward, done, {}

    def reset(self,
        init_dist : np.ndarray = None,
    ) -> np.ndarray:
        init_dist = self.mdp.reset(init_dist)

        return init_dist

    def render(self, mode = 'human'):
        pass

    def close(self):
        pass
