from .trainable_agent import *

class CUDAAgent(TrainableAgent):
    def __init__(self,
        env     : gym.Env,
        verbose : bool   ,
    ):
        super(CUDAAgent, self).__init__(env, verbose)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
