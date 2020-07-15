from .base_buffer import *

class ReplayBuffer(BaseBuffer):
    def sample(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size = self.batch_size, replace = False)

        return dict(
            state      = self.state_buf     [indices],
            action     = self.action_buf    [indices],
            reward     = self.reward_buf    [indices],
            next_state = self.next_state_buf[indices],
            done       = self.done_buf      [indices],
        )
