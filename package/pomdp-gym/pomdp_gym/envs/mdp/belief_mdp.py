from .executable_mdp import *

try: # NOTE: Toggle for test use
    from pomdp import BasePOMDP
except ModuleNotFoundError:
    pass

try: # NOTE: Toggle for package use
    from ..pomdp import BasePOMDP
except ValueError:
    pass

class BeliefMDP(ExecutableMDP):
    def __init__(self,
        pomdp : BasePOMDP,
    ):
        self.pomdp = pomdp

        self.reset()

    def step(self,
        action_index : int,
    ) -> (int, float):
        observation_index, reward = self.pomdp.step(action_index)

        self.prev_belief = self.belief
        self.belief = self.U(self.belief, action_index, observation_index)

        return observation_index, reward

    def reset(self,
        init_dist : np.ndarray = None,
    ) -> np.ndarray:
        init_dist = self.pomdp.reset(init_dist)

        self.prev_belief = None
        self.belief = init_dist

        return init_dist

    def U(self,
        belief            : np.ndarray,
        action_index      : int       ,
        observation_index : int       ,
    ) -> np.ndarray:
        P_s_x_b_a = self.P_s_x_b_a(belief, action_index, observation_index)
        P_x_b_a = self.P_x_b_a(belief, action_index, observation_index)

        return P_s_x_b_a / P_x_b_a if P_x_b_a else np.zeros_like(P_s_x_b_a)

    def P_x_b_a(self,
        belief            : np.ndarray,
        action_index      : int       ,
        observation_index : int       ,
    ) -> np.ndarray:
        return np.sum(self.P_s_x_b_a(belief, action_index, observation_index))

    def P_s_x_b_a(self,
        belief            : np.ndarray,
        action_index      : int       ,
        observation_index : int       ,
    ) -> np.ndarray:
        return self.E_x_a_s(action_index, observation_index) * self.P_s_b_a(belief, action_index)

    def E_x_a_s(self,
        action_index      : int,
        observation_index : int,
    ) -> np.ndarray:
        return self.pomdp.emission_matrix[action_index, :, observation_index]

    def P_s_b_a(self,
        belief       : np.ndarray,
        action_index : int       ,
    ) -> np.ndarray:
        return self.pomdp.transition_matrix[:, action_index, :].transpose().dot(belief)
