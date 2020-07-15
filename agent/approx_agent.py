from .mutable_dp_agent import *

class ApproxAgent(MutableDPAgent):
    def __init__(self,
        env     : gym.Env     ,
        verbose : bool        ,
        gamma   : float       ,
        handler : POMDPHandler,
        method  : str         ,
    ):
        super(ApproxAgent, self).__init__(env, verbose, gamma, handler)

        if method not in [
            'pbvi'  ,
            'sarsop',
        ]:
            raise ValueError

        self.method = method

    def train(self):
        self.handler.write(self.path + self.name + '.pomdp')

        if self.method == 'pbvi':
            solver(
                opt = {
                    'pomdp'   : self.path + self.name + '.pomdp',
                    'method'  : 'grid'                          ,
                    'epsilon' : '1e-6'                          ,
                }                                               ,
                out = self.path + self.name + '.solver'         ,
            )

        if self.method == 'sarsop':
            sarsop(
                opt = {
                    ''  : self.path + self.name + '.pomdp',
                    'o' : self.path + self.name + '.value',
                    'p' : '1e-6'                          ,
                }                                         ,
                out = self.path + self.name + '.sarsop'   ,
            )

        self.alphas = self.handler.read(self.path)
