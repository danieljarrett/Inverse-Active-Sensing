from .belief_mdp import *

class RhoBeliefMDP(BeliefMDP):
    def step(self,
        action_index : int,
    ) -> (int, float):
        observation_index, reward = super(RhoBeliefMDP, self).step(action_index)

        reward += self.reward(self.prev_belief, self.belief)

        return observation_index, reward

    def reward(self,
        prev_belief : np.ndarray,
        belief      : np.ndarray,
    ) -> float:
        raise NotImplementedError
