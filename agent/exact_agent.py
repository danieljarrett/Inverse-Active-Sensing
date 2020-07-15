from .mutable_dp_agent import *

class ExactAgent(MutableDPAgent):
    def __init__(self,
        env     : gym.Env         ,
        verbose : bool            ,
        gamma   : float           ,
        handler : POMDPHandler    ,
        method  : str = 'incprune',
    ):
        super(ExactAgent, self).__init__(env, verbose, gamma, handler)

        if method not in [
            'enum'    ,
            'twopass' ,
            'linsup'  ,
            'witness' ,
            'incprune',
        ]:
            raise ValueError

        self.method = method

    def train(self):
        self.handler.write(self.path + self.name + '.pomdp')

        solver(
            opt = {
                'pomdp'   : self.path + self.name + '.pomdp',
                'method'  : self.method                     ,
                'epsilon' : '1e-6'                          ,
            }                                               ,
            out = self.path + self.name + '.solver'         ,
        )

        self.alphas = self.handler.read(self.path)
