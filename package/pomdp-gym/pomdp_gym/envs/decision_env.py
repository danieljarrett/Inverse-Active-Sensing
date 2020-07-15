from .__head__ import *

from .pomdp import DecisionPOMDP
from .mdp import DecisionMDP

class DecisionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        p_val : np.ndarray ,
        q_val : np.ndarray ,
        eta_a : List[float],
        eta_b : List[float],
        eta_c : List[float],
        costs : List[float],
        eta_d : List[float],
    ):
        pomdp = DecisionPOMDP(
            p_val,
            q_val,
            eta_a,
            eta_b,
            eta_c,
            costs,
        )

        self.mdp = DecisionMDP(
            pomdp,
            eta_d,
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
