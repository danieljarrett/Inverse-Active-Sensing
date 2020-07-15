from .__head__ import *

class BaseAgent:
    def __init__(self,
        env     : gym.Env,
        verbose : bool   ,
    ):
        self.env     = env
        self.verbose = verbose

        self.test_mode = False

    def select_action(self,
        state : np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def perform_action(self,
        action : np.ndarray,
    ) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        return reward, next_state, done

    def demo(self):
        self.test_mode = True

    def test(self,
        num_episodes : int,
    ) -> float:
        self.test_mode = True

        returns = []

        for episode_index in range(num_episodes):
            state = self.env.reset()
            retvrn = 0

            done = False

            print_np(self.env.mdp.belief, '0.3f')
            while not done:
                action = self.select_action(state)
                reward, next_state, done = self.perform_action(action)

                if self.verbose:
                    print('Action:', action, end = '; ')
                    print('Reward:', '%0.3f' % reward, end = '; ')
                    print_np(self.env.mdp.belief, '0.3f')

                state = next_state
                retvrn += reward

            returns += [retvrn]

            if self.verbose:
                print('--------------------', episode_index, '--------------------')

        return np.mean(returns)

    def test_with_inits(self,
        init_states : List[object],
    ) -> float:
        self.test_mode = True

        returns = []

        for episode_index, init_state in enumerate(init_states):
            state = self.env.reset(init_state)
            retvrn = 0

            done = False

            # print_np(self.env.mdp.belief, '0.3f')
            while not done:
                action = self.select_action(state)
                reward, next_state, done = self.perform_action(action)

                if self.verbose:
                    print('Action:', action, end = '; ')
                    print('Reward:', '%0.3f' % reward, end = '; ')
                    print_np(self.env.mdp.belief, '0.3f')

                state = next_state
                retvrn += reward

            returns += [retvrn]

            if self.verbose:
                print('--------------------', episode_index, '--------------------')

        return np.mean(returns)

    def save(self,
        num_episodes : int,
    ) -> float:
        self.test_mode = True

        trajectories = []
        returns = []

        for episode_index in range(num_episodes):
            state = self.env.reset()
            trajectory = []
            retvrn = 0

            done = False

            print_np(self.env.mdp.belief, '0.3f')
            while not done:
                action = self.select_action(state)
                trajectory += [(state, action)]
                reward, next_state, done = self.perform_action(action)

                if self.verbose:
                    print('Action:', action, end = '; ')
                    print('Reward:', '%0.3f' % reward, end = '; ')
                    print_np(self.env.mdp.belief, '0.3f')

                state = next_state
                retvrn += reward

            trajectories += [trajectory]
            returns += [retvrn]

            if self.verbose:
                print('--------------------', episode_index, '--------------------')

        flat = [pair for traj in trajectories for pair in traj]

        np.save('volume/decision.traj', flat)

        return np.mean(returns)

    def save_with_inits(self,
        init_states : List[object],
    ) -> float:
        self.test_mode = True

        trajectories = []
        returns = []

        for episode_index, init_state in enumerate(init_states):
            state = self.env.reset(init_state)
            trajectory = []
            retvrn = 0

            done = False

            print_np(self.env.mdp.belief, '0.3f')
            while not done:
                action = self.select_action(state)
                trajectory += [(state, action)]
                reward, next_state, done = self.perform_action(action)

                if self.verbose:
                    print('Action:', action, end = '; ')
                    print('Reward:', '%0.3f' % reward, end = '; ')
                    print_np(self.env.mdp.belief, '0.3f')

                state = next_state
                retvrn += reward

            trajectories += [trajectory]
            returns += [retvrn]

            if self.verbose:
                print('--------------------', episode_index, '--------------------')

        flat = [pair for traj in trajectories for pair in traj]

        np.save('volume/decision.traj', flat)

        return np.mean(returns)
