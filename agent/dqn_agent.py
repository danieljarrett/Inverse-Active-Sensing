from .cuda_agent import *

from register import BaseRegister
from buffer import ReplayBuffer
from network import MLPNetwork

class DQNAgent(CUDAAgent):
    def __init__(self,
        env             : gym.Env     ,
        verbose         : bool        ,
        register        : BaseRegister,
        batch_size      : int         ,
        buffer          : ReplayBuffer,
        qvalue_function : MLPNetwork  ,
        target_function : MLPNetwork  ,
        lag_interval    : int         ,
        optimizer       : torch.optim ,
        epsilon_decay   : float       ,
        max_epsilon     : float       ,
        min_epsilon     : float       ,
        gamma           : float       ,
    ):
        super(DQNAgent, self).__init__(env, verbose)

        self.register        = register
        self.batch_size      = batch_size
        self.buffer          = buffer
        self.qvalue_function = qvalue_function.to(self.device)
        self.target_function = target_function.to(self.device)
        self.lag_interval    = lag_interval
        self.optimizer       = optimizer
        self.epsilon_decay   = epsilon_decay
        self.max_epsilon     = max_epsilon
        self.min_epsilon     = min_epsilon
        self.gamma           = gamma

        self.epsilon    = max_epsilon
        self.transition = list()

    def select_action(self,
        state : np.ndarray,
    ) -> np.ndarray:
        if not self.test_mode and self.epsilon > np.random.random():
            action = self.env.action_space.sample()
        else:
            action = self.qvalue_function(torch.FloatTensor(state).to(self.device)).argmax()
            action = action.detach().cpu().numpy()

        if not self.test_mode:
            self.transition = [state, action]

        return action

    def perform_action(self,
        action : np.ndarray,
    ) -> Tuple[np.ndarray, np.float64, bool]:
        reward, next_state, done = super(DQNAgent, self).perform_action(action)

        if not self.test_mode:
            self.transition += [reward, next_state, done]
            self.buffer.store(*self.transition)

        return reward, next_state, done

    def train(self,
        num_transitions : int = 10000,
    ):
        super(DQNAgent, self).train()

        state = self.env.reset()
        retvrn = 0

        target_index = 0

        for transition_index in range(num_transitions):
            action = self.select_action(state)
            reward, next_state, done = self.perform_action(action)

            state = next_state
            retvrn += reward

            if done:
                state = self.env.reset()
                self.register.read(retvrn = retvrn)
                retvrn = 0

            if len(self.buffer) < self.batch_size:
                continue

            loss = self._update_qvalue_function()
            self.register.read(loss = loss)

            self.epsilon = max(self.min_epsilon, self.epsilon -
                (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
            self.register.read(epsilon = self.epsilon)

            target_index += 1

            if self._target_to_update(target_index):
                self._update_target_function()

            if self.register.to_plot(transition_index):
                self.register.plot(transition_index)

        self.env.close()

    def demo(self) -> List[np.ndarray]:
        super(DQNAgent, self).demo()

        state = self.env.reset()
        retvrn = 0

        frames = []

        done = False

        while not done:
            frames.append(self.env.render(mode = 'rgb_array'))

            action = self.select_action(state)
            reward, next_state, done = self.perform_action(action)

            state = next_state
            retvrn += reward

        print('return: ', retvrn)

        self.env.close()

        return frames

    def _update_qvalue_function(self) -> torch.Tensor:
        samples = self.buffer.sample()

        loss = self._compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _update_target_function(self):
        self.target_function.load_state_dict(self.qvalue_function.state_dict())

    def _compute_loss(self,
        samples : Dict[str, np.ndarray],
    ) -> torch.Tensor:
        state      = torch.FloatTensor(samples['state'     ]               ).to(self.device)
        action     = torch.LongTensor (samples['action'    ].reshape(-1, 1)).to(self.device)
        reward     = torch.FloatTensor(samples['reward'    ].reshape(-1, 1)).to(self.device)
        next_state = torch.FloatTensor(samples['next_state']               ).to(self.device)
        done       = torch.FloatTensor(samples['done'      ].reshape(-1, 1)).to(self.device)

        qvalue = self.qvalue_function(state).gather(1, action)
        next_qvalue = self.target_function(next_state).max(dim = 1, keepdim = True)[0].detach()
        mask = 1 - done
        retvrn = (reward + self.gamma * next_qvalue * mask).to(self.device)

        loss = F.smooth_l1_loss(qvalue, retvrn)

        return loss

    def _target_to_update(self,
        target_index : int,
    ) -> bool:
        return (target_index + 1) % self.lag_interval == 0
