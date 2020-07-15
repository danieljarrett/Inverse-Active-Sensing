from .__head__ import *

from register import ConsoleRegister
from buffer   import ReplayBuffer
from network  import MLPNetwork
from agent    import BaseAgent, DQNAgent

def make_agent() -> BaseAgent:
    env                 = gym.make('CartPole-v0')
    verbose             = False
    gamma               = 0.95
    state_dim           = env.observation_space.shape[0]
    action_dim          = env.action_space.n
    plot_interval       = 200
    register            = ConsoleRegister(
        variables       = [
                            'retvrn'   ,
                            'loss'     ,
                            'epsilon'  ,
                          ]            ,
        save_interval   = None         ,
        plot_interval   = plot_interval,
        fullname        = None         ,
    )
    batch_size          = 32
    buffer_size         = 1000
    buffer              = ReplayBuffer(state_dim, buffer_size, batch_size)
    qvalue_function     = MLPNetwork(state_dim, action_dim)
    target_function     = MLPNetwork.copy(qvalue_function); target_function.eval()
    lag_interval        = 100
    optimizer           = optim.Adam(qvalue_function.parameters())
    epsilon_decay       = 1 / 5000
    max_epsilon         = 1.0
    min_epsilon         = 0.1

    return DQNAgent(
        env             = env            ,
        verbose         = verbose        ,
        register        = register       ,
        batch_size      = batch_size     ,
        buffer          = buffer         ,
        qvalue_function = qvalue_function,
        target_function = target_function,
        lag_interval    = lag_interval   ,
        optimizer       = optimizer      ,
        epsilon_decay   = epsilon_decay  ,
        max_epsilon     = max_epsilon    ,
        min_epsilon     = min_epsilon    ,
        gamma           = gamma          ,
    )

if __name__ == '__main__':
    agent = make_agent()
    agent.register.on()
    agent.train()
    agent.demo()
    agent.register.off()
