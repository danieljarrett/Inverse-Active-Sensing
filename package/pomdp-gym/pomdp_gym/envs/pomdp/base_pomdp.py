from .__head__ import *

class BasePOMDP:
    def __init__(self):
        self.set()

        self.transition_matrix = self._generate_transition_matrix()
        self.emission_matrix = self._generate_emission_matrix()
        self.reward_matrix = self._generate_reward_matrix()

        self.reset()

    def set(self):
        raise NotImplementedError

    def reset(self,
        init_dist : np.ndarray = None,
    ) -> np.ndarray:
        if init_dist is None:
            init_dist = self.default_dist()

        self.state_index = np.random.choice(len(init_dist), p = init_dist)
        self.prev_state_index = None

        return init_dist

    def default_dist(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_transition_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_emission_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_reward_matrix(self) -> np.ndarray:
        raise NotImplementedError
