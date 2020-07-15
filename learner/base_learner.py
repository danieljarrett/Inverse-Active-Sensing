from .__head__ import *

class BaseLearner:
    def __init__(self,
        agent : BaseAgent                   ,
        trajs : List[Tuple[np.ndarray, int]],
    ):
        self.agent = agent
        self.trajs = trajs

        self.path = STORE
        self.name = PNAME

    def forward(self):
        self.agent.train()

    def inverse(self):
        raise NotImplementedError

    def configure_reward(self,
        eta : np.ndarray,
    ):
        cut = self.agent.env.mdp.pomdp.dim_abc

        self.agent.env.mdp.pomdp.configure_reward(eta[:cut])
        self.agent.env.mdp.configure_reward(eta[cut:])

    def restore_reward(self):
        self.agent.env.mdp.pomdp.restore_reward()
        self.agent.env.mdp.restore_reward()
