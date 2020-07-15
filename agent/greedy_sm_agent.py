from .greedy_agent import *

class GreedySMAgent(GreedyAgent):
    def __init__(self,
        env     : gym.Env     ,
        verbose : bool        ,
        gamma   : float       ,
        handler : POMDPHandler,
        method  : str         ,
        itemp   : float       ,
    ):
        super(GreedySMAgent, self).__init__(env, verbose, gamma, handler, method)

        self.itemp = itemp

    def select_action(self,
        state : np.ndarray,
    ) -> int:
        num_actions = self.env.mdp.pomdp.num_actions

        log_ener = np.array([self.itemp * self.action_value(state, action)
            for action in range(num_actions)])
        log_part = sp.special.logsumexp(log_ener)

        return np.random.choice(num_actions, p = np.exp(log_ener - log_part))
