from .approx_agent import *

class GreedyAgent(ApproxAgent):
    def select_action(self,
        state : np.ndarray,
    ) -> int:
        if sum(state > 0.85) > 0:
            return list(state).index(max(state)) + 2

        action_values = [self.action_value(state, action_index) for action_index in range(2)]

        return action_values.index(max(action_values))

    def stop_value(self,
        state : np.ndarray,
    ) -> float:
        stop_values = []

        for action_index in [2, 3]:
            stop_values.append(super(GreedyAgent, self).action_value(state, action_index))

        return min(stop_values)

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
            next_state_value = self.stop_value(next_state) * self.gamma

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

    def stop_value(self,
        state : np.ndarray,
    ) -> float:
        stop_values = []

        for action_index in [2, 3]:
            stop_values.append(super(GreedyAgent, self).action_value(state, action_index))

        return min(stop_values)

    def action_value(self,
        state        : np.ndarray,
        action_index : int       ,
    ) -> float:
        if action_index == 2:
            return state[0] - 1 + 0.25

        if action_index == 3:
            return state[1] - 1.25 + 0.25

        list_of_P_x_b_as = []
        list_of_next_state_values = []
        list_of_expected_rewards = []

        for observation_index in range(self.env.mdp.pomdp.num_observations):
            P_x_b_a = self.env.mdp.P_x_b_a(state, action_index, observation_index)

            next_state = self.env.mdp.U(state, action_index, observation_index)
            next_state_value = self.stop_value(next_state) * self.gamma

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
