from .__head__ import *

class BaseMDP:
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
        raise NotImplementedError

    def default_dist(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_transition_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_emission_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def _generate_reward_matrix(self) -> np.ndarray:
        raise NotImplementedError
