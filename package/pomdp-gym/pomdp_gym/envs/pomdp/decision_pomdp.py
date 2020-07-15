from .mutable_hypothesis_pomdp import *

class DecisionPOMDP(MutableHypothesisPOMDP):
    def _generate_transition_matrix(self) -> np.ndarray:
        transition_matrix = np.zeros(
            (self.num_states, self.num_actions, self.num_states))

        d = self.done_state_index

        # TEST ACTIONS
        for j in range(self.num_acquisitions):
            for i in range(self.num_hypotheses):
                for k in range(self.num_hypotheses):
        # (1) Hypothesis to Hypothesis
                    transition_matrix[i, j, k] = 1 - self.p_val[i, j] if k == i else 0

        # (2) Hypothesis to Done
                transition_matrix[i, j, d] = self.p_val[i, j]

            for k in range(self.num_hypotheses):
        # (3) Done to Hypothesis [Catch]
                transition_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            transition_matrix[d, j, d] = 1

        # STOP ACTIONS
        for j in range(self.num_acquisitions, self.num_actions):
            for i in range(self.num_hypotheses):
                for k in range(self.num_hypotheses):
        # (1) Hypothesis to Hypothesis
                    transition_matrix[i, j, k] = 0

        # (2) Hypothesis to Done
                transition_matrix[i, j, d] = 1

            for k in range(self.num_hypotheses):
        # (3) Done to Hypothesis [Catch]
                transition_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            transition_matrix[d, j, d] = 1

        return transition_matrix

    def _generate_emission_matrix(self) -> np.ndarray:
        q_val_trans = np.transpose(self.q_val, axes = (1, 0, 2))

        emission_matrix = np.zeros(
            (self.num_actions, self.num_states, self.num_observations))

        emission_matrix[
            :q_val_trans.shape[0],
            :q_val_trans.shape[1],
            :q_val_trans.shape[2],
        ] = q_val_trans

        d = self.done_state_index
        n = self.napp_observation_index
        m = self.done_observation_index

        for j in range(self.num_actions):
            emission_matrix[j, d, m] = 1

        # [Catch]
        for j in range(self.num_acquisitions, self.num_actions):
            for i in range(self.num_hypotheses):
                emission_matrix[j, i, n] = 1

        return emission_matrix

    def _generate_reward_matrix(self) -> np.ndarray:
        reward_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))

        d = self.done_state_index

        # TEST ACTIONS
        for j in range(self.num_acquisitions):
            for i in range(self.num_hypotheses):
                for k in range(self.num_hypotheses):
        # (1) Hypothesis to Hypothesis
                    reward_matrix[i, j, k] = self.costs[j] * self.eta_c[j]

        # (2) Hypothesis to Done
                reward_matrix[i, j, d] = self.costs[j] * self.eta_c[j] + self.eta_b[i]

            for k in range(self.num_hypotheses):
        # (3) Done to Hypothesis [Catch]
                reward_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            reward_matrix[d, j, d] = 0

        # STOP ACTIONS
        for j in range(self.num_acquisitions, self.num_actions):
            for i in range(self.num_hypotheses):
                for k in range(self.num_hypotheses):
        # (1) Hypothesis to Hypothesis [Catch]
                    reward_matrix[i, j, k] = 0

        # (2) Hypothesis to Done
                reward_matrix[i, j, d] = 0 if j == i + self.num_acquisitions else self.eta_a[i]

            for k in range(self.num_hypotheses):
        # (3) Done to Hypothesis [Catch]
                reward_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            reward_matrix[d, j, d] = 0

        return reward_matrix
