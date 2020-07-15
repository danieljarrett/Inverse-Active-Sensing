from .executable_pomdp import *

class HypothesisPOMDP(ExecutablePOMDP):
    def __init__(self,
        p_val : np.ndarray , # \Theta x \Lambda
        q_val : np.ndarray , # \Theta x \Lambda x \Omega
        eta_a : List[float], # \Theta
        eta_b : List[float], # \Theta
        eta_c : List[float], # \Lambda
        costs : List[float], # \Lambda
    ):
        self.p_val = p_val
        self.q_val = q_val
        self.eta_a = eta_a
        self.eta_b = eta_b
        self.eta_c = eta_c
        self.costs = costs

        super(HypothesisPOMDP, self).__init__()

    def step(self,
        action_index : int,
    ) -> (int, float):
        observation_index, reward = super(HypothesisPOMDP, self).step(action_index)

        if self.state_index == self.done_state_index:
            self.done = True

        return observation_index, reward

    def set(self):
        self.num_actions            , \
        self.num_acquisitions       , \
        self.num_decisions          , = self._set_action_vars()
        self.num_states             , \
        self.num_hypotheses         , \
        self.done_state_index       , = self._set_state_vars(self.num_decisions)
        self.num_observations       , \
        self.napp_observation_index , \
        self.done_observation_index , = self._set_observation_vars()

    def reset(self,
        init_dist : np.ndarray = None,
    ) -> np.ndarray:
        init_dist = super(HypothesisPOMDP, self).reset(init_dist)

        self.done = False

        return init_dist

    def default_dist(self) -> np.ndarray:
        dist = np.random.random(self.num_hypotheses)
        dist /= dist.sum()

        return np.array(list(dist) + [0])
        return np.array([1 / self.num_hypotheses] * self.num_hypotheses + [0])

    def _set_action_vars(self) -> (int, int, int):
        num_acquisitions = len(self.eta_c)
        num_decisions = len(self.eta_a)
        num_actions = num_acquisitions + num_decisions

        return num_actions, num_acquisitions, num_decisions

    def _set_state_vars(self,
        num_decisions : int,
    ) -> (int, int, int):
        num_states = num_decisions + 1
        num_hypotheses = num_decisions
        done_state_index = num_states - 1

        return num_states, num_hypotheses, done_state_index

    def _set_observation_vars(self) -> (int, int, int):
        num_observations = self.q_val.shape[2] + 2
        napp_observation_index = num_observations - 2
        done_observation_index = num_observations - 1

        return num_observations, napp_observation_index, done_observation_index
