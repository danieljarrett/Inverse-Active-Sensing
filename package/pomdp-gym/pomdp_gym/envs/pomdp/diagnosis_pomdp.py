from .mutable_disease_pomdp import *

class DiagnosisPOMDP(MutableDiseasePOMDP):
    def _generate_transition_matrix(self) -> np.ndarray:
        transition_matrix = np.zeros(
            (self.num_states, self.num_actions, self.num_states))

        d = self.done_state_index

        # TEST ACTIONS
        for j in range(self.num_tests):
            for i in range(self.num_diseases):
                for k in range(self.num_diseases):
        # (1) Disease to Disease
                    transition_matrix[i, j, k] = 1 - self.prob_comp[i] if k == i else 0

        # (2) Disease to Done
                transition_matrix[i, j, d] = self.prob_comp[i]

            for k in range(self.num_diseases):
        # (3) Done to Disease [Catch]
                transition_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            transition_matrix[d, j, d] = 1

        # STOP ACTIONS
        for j in range(self.num_tests, self.num_actions):
            for i in range(self.num_diseases):
                for k in range(self.num_diseases):
        # (1) Disease to Disease
                    transition_matrix[i, j, k] = 0

        # (2) Disease to Done
                transition_matrix[i, j, d] = 1

            for k in range(self.num_diseases):
        # (3) Done to Disease [Catch]
                transition_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            transition_matrix[d, j, d] = 1

        return transition_matrix

    def _generate_emission_matrix(self) -> np.ndarray:
        emission_matrix = np.zeros(
            (self.num_actions, self.num_states, self.num_observations))

        d = self.done_state_index
        n = self.napp_observation_index
        m = self.done_observation_index

        # TEST ACTIONS
        for j in range(self.num_tests):
            for k in range(self.num_diseases):
                action = self.index2action(j)
                diagnosis = self.index2action(k + self.num_tests)

                match = True

                for index, level in enumerate(action):
                    if level != diagnosis[index]:
                        match = False

                        break

                if match:
                    observations = list(range(self.num_observations))

                    o = diagnosis[index + 1]

                    observations.remove(n)
                    observations.remove(m)
                    observations.remove(o)

        # (1) Disease to Disease: Proper -> Succeed
                    emission_matrix[j, k, o] = 1 - self.prob_fail[j]

        # (1) Disease to Disease: Proper -> Fail
                    for o_prime in observations:
                        emission_matrix[j, k, o_prime] = \
                            self.prob_fail[j] / (self.num_observations - 3)

                else:
        # (1) Disease to Disease: Improper -> N/A
        # (3) Done to Disease [Catch]
                    # NOTE: Style 1 - Wrong Test gives N/A Result
                    # emission_matrix[j, k, n] = 1

                    # NOTE: Style 2 - Wrong Test is Uninformative
                    for l in range(m - 1):
                        emission_matrix[j, k, l] = 1 / (m - 1)

        # (2) Disease to Done
        # (4) Done to Done [Catch]
            emission_matrix[j, d, m] = 1

        # STOP ACTIONS
        for j in range(self.num_tests, self.num_actions):
            for k in range(self.num_diseases):
        # (1) Disease to Disease [Catch]
        # (3) Done to Disease [Catch]
                emission_matrix[j, k, m] = 1

        # (2) Disease to Done
        # (4) Done to Done [Catch]
            emission_matrix[j, d, m] = 1

        return emission_matrix

    def _generate_reward_matrix(self) -> np.ndarray:
        reward_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))

        d = self.done_state_index

        # TEST ACTIONS
        for j in range(self.num_tests):
            for i in range(self.num_diseases):
                for k in range(self.num_diseases):
        # (1) Disease to Disease
                    reward_matrix[i, j, k] = self.reward_test[j]

        # (2) Disease to Done
                reward_matrix[i, j, d] = self.reward_test[j] + self.reward_comp[i]
                # NOTE: Final index fixed from [k] to [i]

            for k in range(self.num_diseases):
        # (3) Done to Disease [Catch]
                reward_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            reward_matrix[d, j, d] = 0

        # STOP ACTIONS
        for j in range(self.num_tests, self.num_actions):
            for i in range(self.num_diseases):
                for k in range(self.num_diseases):
        # (1) Disease to Disease [Catch]
                    reward_matrix[i, j, k] = 0

        # (2) Disease to Done
                reward_matrix[i, j, d] = 0 if j == i + self.num_tests else self.reward_mark[i]

            for k in range(self.num_diseases):
        # (3) Done to Disease [Catch]
                reward_matrix[d, j, k] = 0

        # (4) Done to Done [Catch]
            reward_matrix[d, j, d] = 0

        return reward_matrix
