from .tree_pomdp import *
from .executable_pomdp import *

class DiseasePOMDP(TreePOMDP, ExecutablePOMDP):
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
    ):
        self.prob_comp   = prob_comp
        self.prob_fail   = prob_fail
        self.reward_test = np.array(theta_test) * phi_test
        self.reward_comp = np.array(theta_comp) * phi_comp
        self.reward_mark = np.array(theta_mark) * phi_mark

        super(DiseasePOMDP, self).__init__(
            tree_factor,
            tree_height,
        )

    def step(self,
        action_index : int,
    ) -> (int, float):
        observation_index, reward = super(DiseasePOMDP, self).step(action_index)

        if self.state_index == self.done_state_index:
            self.done = True

        return observation_index, reward

    def set(self):
        self.num_actions            , \
        self.num_tests              , \
        self.num_diagnoses          , = self._set_action_vars()
        self.num_states             , \
        self.num_diseases           , \
        self.done_state_index       , = self._set_state_vars(self.num_diagnoses)
        self.num_observations       , \
        self.napp_observation_index , \
        self.done_observation_index , = self._set_observation_vars()

    def reset(self,
        init_dist : np.ndarray = None,
    ) -> np.ndarray:
        init_dist = super(DiseasePOMDP, self).reset(init_dist)

        self.done = False

        return init_dist

    def default_dist(self) -> np.ndarray:
        dist = np.random.random(self.num_diseases)
        dist /= dist.sum()

        return np.array(list(dist) + [0])
        return np.array([1 / self.num_diseases] * self.num_diseases + [0])

    def _set_action_vars(self) -> (int, int, int):
        max_action = [self.root_action] + [self.tree_factor - 1] * (self.tree_height - 1)
        max_test   = [self.root_action] + [self.tree_factor - 1] * (self.tree_height - 2)

        num_actions   = self.action2index(max_action) + 1
        num_tests     = self.action2index(max_test  ) + 1
        num_diagnoses = self.tree_factor ** (self.tree_height - 1)

        assert num_actions == num_tests + num_diagnoses

        return num_actions, num_tests, num_diagnoses

    def _set_state_vars(self,
        num_diagnoses : int,
    ) -> (int, int, int):
        num_states       = num_diagnoses + 1
        num_diseases     = num_diagnoses
        done_state_index = num_states - 1

        assert num_states == num_diseases + len([done_state_index])

        return num_states, num_diseases, done_state_index

    def _set_observation_vars(self) -> (int, int, int):
        num_observations       = self.tree_factor + 2
        napp_observation_index = self.tree_factor
        done_observation_index = self.tree_factor + 1

        assert num_observations == self.tree_factor + len([napp_observation_index,
            done_observation_index])

        return num_observations, napp_observation_index, done_observation_index
