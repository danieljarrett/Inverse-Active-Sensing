from .base_pomdp import *

class ExecutablePOMDP(BasePOMDP):
    def step(self,
        action_index : int,
    ) -> (int, float):
        self.prev_state_index = self.state_index

        self.state_index = self.transition(self.state_index, action_index)
        observation_index = self.emission(action_index, self.state_index)
        reward = self.reward(self.prev_state_index, action_index, self.state_index)

        return observation_index, reward

    def transition(self,
        state_index  : int,
        action_index : int,
    ) -> int:
        p = self.transition_matrix[state_index, action_index]

        next_state_index = np.random.choice(len(p), p = p)

        return next_state_index

    def emission(self,
        action_index     : int,
        next_state_index : int,
    ) -> int:
        p = self.emission_matrix[action_index, next_state_index]

        next_observation_index = np.random.choice(len(p), p = p)

        return next_observation_index

    def reward(self,
        state_index      : int,
        action_index     : int,
        next_state_index : int,
    ) -> float:
        reward = self.reward_matrix[state_index, action_index, next_state_index]

        return reward
