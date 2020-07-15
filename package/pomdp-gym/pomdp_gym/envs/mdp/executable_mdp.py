from .base_mdp import *

class ExecutableMDP(BaseMDP):
    def step(self,
        action_index : int,
    ) -> (int, float):
        raise NotImplementedError

    def transition(self,
        state_index  : int,
        action_index : int,
    ) -> int:
        raise NotImplementedError

    def emission(self,
        action_index     : int,
        next_state_index : int,
    ) -> int:
        raise NotImplementedError

    def reward(self,
        state_index      : int,
        action_index     : int,
        next_state_index : int,
    ) -> float:
        raise NotImplementedError
