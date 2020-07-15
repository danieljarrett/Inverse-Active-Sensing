import gym
import numpy as np
import scipy as sp

from handler  import POMDPHandler
from agent    import ApproxAgent, SoftmaxAgent, GreedySMAgent
from learner  import BayesianMCMCLearner, Example2Learner
from density  import UniformDensity
from sampler  import GridWalkSampler
from register import ConsoleRegister
from config   import decision_kwargs

PICK_SCENE = 5
PICK_INITS = 2

# (1) ACTIVE SENSING

if False:
    env       = gym.make('pomdp_gym:decision-v0', **decision_kwargs(PICK_SCENE))
    verbose   = True
    state_dim = env.observation_space.shape[0]
    gamma     = 0.99
    handler   = POMDPHandler(env.mdp.pomdp, gamma)
    method    = 'sarsop'
    itemp     = 10.0

    teacher = SoftmaxAgent(
        env     = env    ,
        verbose = verbose,
        gamma   = gamma  ,
        handler = handler,
        method  = method ,
        itemp   = itemp  ,
    ); teacher.train()

    if PICK_INITS == 1:
        teacher.save(1000)

    if PICK_INITS == 2:
        num_episodes, num_inits = 300, 50
        init, init_dists = -0.01, []

        for idx in range(num_inits):
            init += 1. / num_inits

            for episode in range(int(num_episodes / num_inits)):
                init_dists.append(np.array([init, 1. - init, 0]))

        teacher.save_with_inits(init_dists)

# (2) INVERSE ACTIVE SENSING

    lower = -1.00
    upper =  0.00
    delta =  0.05
    burns =  3000
    iters =  10000

    if PICK_SCENE in [4, 6]:
        dim   =  9
        learn = [0] * 4 + [1] * 2 + [0] * 3

    if PICK_SCENE == 5:
        dim   =  8
        learn = [1] * 2 + [0] * 6

    if PICK_SCENE == 7:
        dim   =  9
        learn = [1] * 2 + [0] * 7

    learner = Example2Learner(
        agent       = ApproxAgent(
            env     = env    ,
            verbose = verbose,
            gamma   = gamma  ,
            handler = handler,
            method  = method ,
        ),
        trajs       = np.load('volume/decision.traj.npy', allow_pickle = True)[()],
        prior       = UniformDensity(
            dim     = dim    ,
            lower   = lower  ,
            upper   = upper  ,
        ),
        itemp       = itemp  ,
        burns       = burns  ,
        iters       = iters  ,
        sampler     = GridWalkSampler(
            dim     = dim    ,
            learn   = learn  ,
            delta   = delta  ,
            lower   = lower  ,
            upper   = upper  ,
        ),
        register    = ConsoleRegister(
            variables = [
                'samples'    ,
                'eta_1'    ,
                'eta_2'    ,
                'eta_1_bar',
                'eta_2_bar',
                'ratio'      ,
            ],
            save_interval = 100,
            plot_interval = None,
            fullname = 'volume/decision.trace',
        ),
    )

    learner.compute()
    learner.log_posteriors = np.load('volume/decision.logps.npy', allow_pickle = True)[()]

# (3) FIGURE A: IDENTIFIABILITY

if False:
    (i, j) = np.unravel_index(learner.log_posteriors.argmax(), learner.log_posteriors.shape)

    learner.inverse()
    trace = np.load('volume/decision.trace.npy', allow_pickle = True)[()]

    ETA_MAP    = (i * delta, j * delta)
    ETA_MEAN   = (np.average(   trace['eta_1'])      , np.average(   trace['eta_2'])      )
    ETA_MEDIAN = (np.median(    trace['eta_1'])      , np.median(    trace['eta_2'])      )
    ETA_MODE   = (sp.stats.mode(trace['eta_1'])[0][0], sp.stats.mode(trace['eta_2'])[0][0])

    ary = np.array([ETA_MAP, ETA_MEAN, ETA_MEDIAN, ETA_MODE])

    np.savetxt('volume/figures/ary.txt', ary)

    heatmap_id = np.zeros_like(learner.log_posteriors)
    for eta_1, eta_2 in list(zip(trace['eta_1'], trace['eta_2'])):
        i = int(-eta_1 / 0.05)
        j = int(-eta_2 / 0.05)

        heatmap_id[i, j] += 1

    heatmap_id = np.transpose(heatmap_id)[::-1]
    np.save('volume/figures/heatmap_id.npy', heatmap_id)

# (4) FIGURE B: PREDICTABILITY

if False:
    env       = gym.make('pomdp_gym:decision-v0', **decision_kwargs(PICK_SCENE))
    verbose   = True
    state_dim = env.observation_space.shape[0]
    gamma     = 0.99
    handler   = POMDPHandler(env.mdp.pomdp, gamma)
    method    = 'sarsop'
    itemp     = 10.0

    teacher = SoftmaxAgent(
        env     = env    ,
        verbose = verbose,
        gamma   = gamma  ,
        handler = handler,
        method  = method ,
        itemp   = itemp  ,
    ); teacher.train()

    env       = gym.make('pomdp_gym:decision-v0', **decision_kwargs(PICK_SCENE))
    verbose   = False
    state_dim = env.observation_space.shape[0]
    gamma     = 0.99
    handler   = POMDPHandler(env.mdp.pomdp, gamma)
    method    = 'sarsop'
    itemp     = 10.0

    apprentice = SoftmaxAgent(
        env     = env    ,
        verbose = verbose,
        gamma   = gamma  ,
        handler = handler,
        method  = method ,
        itemp   = itemp  ,
    )

    if PICK_SCENE in [4, 6]:
        eta_a = [-0.50, -0.50]
        eta_b = [-0.50, -0.50]
        eta_d = [ 0.00] * 3

    if PICK_SCENE == 5:
        eta_b = [-1.00, -1.00]
        eta_c = [-0.25]
        eta_d = [ 0.00] * 3

    if PICK_SCENE == 7:
        eta_b = [-1.00, -1.00]
        eta_c = [-0.50, -0.50]
        eta_d = [ 0.00] * 3

    heatmap_pr = np.zeros_like(learner.log_posteriors)
    for i in range(21):
        for j in range(21):
            print('Computing %d of %d ...' % (i * 21 + j, 21 * 21))

            eta_s = [-i * 0.05, -j * 0.05]

            if PICK_SCENE in [4, 6]:
                eta = eta_a + eta_b + eta_s + eta_d

            if PICK_SCENE == 5:
                eta = eta_s + eta_b + eta_c + eta_d

            apprentice.env.mdp.pomdp.configure_reward(eta)
            apprentice.train()
            apprentice.env.mdp.pomdp.restore_reward()

            score = 0

            for state, action in np.load('volume/decision.traj.npy', allow_pickle = True)[()]:
                score += apprentice.prob_actions(state).dot(teacher.prob_actions(state))

            heatmap_pr[i, j] = score

    heatmap_pr = np.transpose(heatmap_pr)[::-1]
    np.save('volume/figures/heatmap_pr.npy', heatmap_pr)

# (5) FIGURE C: OPTIMALITY

if False:
    env       = gym.make('pomdp_gym:decision-v0', **decision_kwargs(PICK_SCENE))
    verbose   = False
    state_dim = env.observation_space.shape[0]
    gamma     = 0.99
    handler   = POMDPHandler(env.mdp.pomdp, gamma)
    method    = 'sarsop'

    apprentice = ApproxAgent(
        env     = env    ,
        verbose = verbose,
        gamma   = gamma  ,
        handler = handler,
        method  = method ,
    )

    if PICK_SCENE in [4, 6]:
        eta_a = [-0.50, -0.50]
        eta_b = [-0.50, -0.50]
        eta_d = [ 0.00] * 3

    if PICK_SCENE == 5:
        eta_b = [-1.00, -1.00]
        eta_c = [-0.25]
        eta_d = [ 0.00] * 3

    if PICK_SCENE == 7:
        eta_b = [-1.00, -1.00]
        eta_c = [-0.50, -0.50]
        eta_d = [ 0.00] * 3

    heatmap_op = np.zeros_like(learner.log_posteriors)
    for i in range(21):
        for j in range(21):
            print('Computing %d of %d ...' % (i * 21 + j, 21 * 21))

            eta_s = [-i * 0.05, -j * 0.05]

            if PICK_SCENE in [4, 6]:
                eta = eta_a + eta_b + eta_s + eta_d

            if PICK_SCENE == 5:
                eta = eta_s + eta_b + eta_c + eta_d

            apprentice.env.mdp.pomdp.configure_reward(eta)
            apprentice.train()
            apprentice.env.mdp.pomdp.restore_reward()

            heatmap_op[i, j] = apprentice.test_with_inits(init_dists)

    heatmap_op = np.transpose(heatmap_op)[::-1]
    np.save('volume/figures/heatmap_op.npy', heatmap_op)

# (6) BETA GRID

if False:
    for row, itemp in enumerate([
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
        1.00,
        2.00,
        5.00,
        10.0,
        20.0,
        50.0,
        100.,
    ]):
        env       = gym.make('pomdp_gym:decision-v0', **decision_kwargs(PICK_SCENE))
        verbose   = True
        state_dim = env.observation_space.shape[0]
        gamma     = 0.99
        handler   = POMDPHandler(env.mdp.pomdp, gamma)
        method    = 'sarsop'
        itemp     = itemp

        teacher = SoftmaxAgent(
            env     = env    ,
            verbose = verbose,
            gamma   = gamma  ,
            handler = handler,
            method  = method ,
            itemp   = itemp  ,
        ); teacher.train()

        if PICK_INITS == 1:
            teacher.save(1000)

        if PICK_INITS == 2:
            num_episodes, num_inits = 300, 50
            init, init_dists = -0.01, []

            for idx in range(num_inits):
                init += 1. / num_inits

                for episode in range(int(num_episodes / num_inits)):
                    init_dists.append(np.array([init, 1. - init, 0]))

            teacher.save_with_inits(init_dists)

        lower = -1.00
        upper =  0.00
        delta =  0.05
        burns =  3000
        iters =  10000

        if PICK_SCENE in [4, 6]:
            dim   =  9
            learn = [0] * 4 + [1] * 2 + [0] * 3

        if PICK_SCENE == 5:
            dim   =  8
            learn = [1] * 2 + [0] * 6

        if PICK_SCENE == 7:
            dim   =  9
            learn = [1] * 2 + [0] * 7

        learner = Example2Learner(
            agent       = ApproxAgent(
                env     = env    ,
                verbose = verbose,
                gamma   = gamma  ,
                handler = handler,
                method  = method ,
            ),
            trajs       = np.load('volume/decision.traj.npy', allow_pickle = True)[()],
            prior       = UniformDensity(
                dim     = dim    ,
                lower   = lower  ,
                upper   = upper  ,
            ),
            itemp       = itemp  ,
            burns       = burns  ,
            iters       = iters  ,
            sampler     = GridWalkSampler(
                dim     = dim    ,
                learn   = learn  ,
                delta   = delta  ,
                lower   = lower  ,
                upper   = upper  ,
            ),
            register    = ConsoleRegister(
                variables = [
                    'samples'    ,
                    'eta_1'    ,
                    'eta_2'    ,
                    'eta_1_bar',
                    'eta_2_bar',
                    'ratio'      ,
                ],
                save_interval = 100,
                plot_interval = None,
                fullname = 'volume/decision.trace',
            ),
        )

        learner.betas(row)

    mat = []

    for row in range(13):
        mat.append(np.load('volume/betas/decision.betas' + str(row) + '.npy',
            allow_pickle = True)[()])

    np.save('volume/betas/betas.npy', np.array(mat))
