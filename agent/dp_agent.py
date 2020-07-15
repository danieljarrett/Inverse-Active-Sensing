from .trainable_agent import *

from handler import POMDPHandler
from pomdp_sol import solver, sarsop

class DPAgent(TrainableAgent):
    def __init__(self,
        env     : gym.Env     ,
        verbose : bool        ,
        gamma   : float       ,
        handler : POMDPHandler,
    ):
        super(DPAgent, self).__init__(env, verbose)

        self.gamma = gamma
        self.handler = handler

        self.path = STORE
        self.name = PNAME

        self.alphas = None

    def select_action(self,
        state : np.ndarray,
    ) -> int:
        state_values = [state.dot(alpha[0]) for alpha in self.alphas]

        max_index = state_values.index(max(state_values))

        return self.alphas[max_index][1]

    def state_value(self,
        state : np.ndarray,
    ) -> float:
        state_values = [state.dot(alpha[0]) for alpha in self.alphas]

        return max(state_values)

    def action_value(self,
        state        : np.ndarray,
        action_index : int       ,
    ) -> float:
        list_of_P_x_b_as = []
        list_of_next_state_values = []
        list_of_expected_rewards = []

        for observation_index in range(self.env.mdp.pomdp.num_observations):
            P_x_b_a = self.env.mdp.P_x_b_a(state, action_index, observation_index)

            next_state = self.env.mdp.U(state, action_index, observation_index)
            next_state_value = self.state_value(next_state) * self.gamma

            matrix = self.env.mdp.pomdp.reward_matrix[:, action_index, :]
            expected_reward = np.tensordot(matrix, state, axes = (0, 0)).dot(next_state)

            list_of_P_x_b_as.append(P_x_b_a)
            list_of_next_state_values.append(next_state_value)
            list_of_expected_rewards.append(expected_reward)

        expected_next_state_value = np.sum([P * q for P, q in
            zip(list_of_P_x_b_as, list_of_next_state_values)])

        expected_expected_reward = np.sum([P * r for P, r in
            zip(list_of_P_x_b_as, list_of_expected_rewards)])

        return expected_next_state_value + expected_expected_reward

    def train(self):
        raise NotImplementedError
