from .__head__ import *

from ..pomdp import DecisionPOMDP
from ..mdp import DecisionMDP

# Deadline Risk (for Hi/Lo risk Tests)
# Hi risk Test: Hypothesis 1? (Yes/No)
p_hi = 0.12
p_lo = 0.06

# Test Accuracy (for Hi/Lo risk Tests)
# Lo risk Test: Hypothesis 2/3? (2/3)
q_hi = 0.90
q_lo = 0.80

unif = 0.50

p_val = np.array([
    # State 1
    [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
    # State 2
    [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
    # State 3
    [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
])
q_val = np.array([
    [ # State 1
        [    q_hi, 1 - q_hi],
        [1 - q_hi,     q_hi],
        [1 - q_hi,     q_hi],
        [    unif,     unif],
        [1 - q_lo,     q_lo],
        [    q_lo, 1 - q_lo],
    ],
    [ # State 2
        [1 - q_hi,     q_hi],
        [    q_hi, 1 - q_hi],
        [1 - q_hi,     q_hi],
        [    q_lo, 1 - q_lo],
        [    unif,     unif],
        [1 - q_lo,     q_lo],
    ],
    [ # State 3
        [1 - q_hi,     q_hi],
        [1 - q_hi,     q_hi],
        [    q_hi, 1 - q_hi],
        [1 - q_lo,     q_lo],
        [    q_lo, 1 - q_lo],
        [    unif,     unif],
    ],
])
eta_a = [-0.40] * 3
eta_b = [-0.60] * 3
eta_c = [-0.10] * 6
costs = [ 1.00] * 6
eta_d = [ 0.00] * 4

TEST_POMDP = True
TEST_MDP   = True

pomdp = DecisionPOMDP(
    p_val,
    q_val,
    eta_a,
    eta_b,
    eta_c,
    costs,
)

mdp = DecisionMDP(
    pomdp,
    eta_d,
)

action_1 = 0
action_2 = 1
action_3 = 2
action_4 = 3
action_5 = 4
action_6 = 5

if TEST_POMDP:
    pomdp.state_index = 0

    print('\nCheck transitions (acquisitions: hi) ...\n')
    print(pomdp.transition_matrix[:, 0, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 1, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 2, :], end = '\n\n')

    print('\nCheck transitions (acquisitions: lo) ...\n')
    print(pomdp.transition_matrix[:, 3, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 4, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 5, :], end = '\n\n')

    print('\nCheck transitions (decisions) ...\n')
    print(pomdp.transition_matrix[:, 6, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 7, :], end = '\n\n')
    print(pomdp.transition_matrix[:, 8, :], end = '\n\n')

    print('\nCheck emissions (acquisitions: hi) ...\n')
    print(pomdp.emission_matrix[0], end = '\n\n')
    print(pomdp.emission_matrix[1], end = '\n\n')
    print(pomdp.emission_matrix[2], end = '\n\n')

    print('\nCheck emissions (acquisitions: lo) ...\n')
    print(pomdp.emission_matrix[3], end = '\n\n')
    print(pomdp.emission_matrix[4], end = '\n\n')
    print(pomdp.emission_matrix[5], end = '\n\n')

    print('\nCheck emissions (decisions) ...\n')
    print(pomdp.emission_matrix[6], end = '\n\n')
    print(pomdp.emission_matrix[7], end = '\n\n')
    print(pomdp.emission_matrix[8], end = '\n\n')

    print('\nCheck reward matrix (from hypothesis) ...\n')
    print(pomdp.reward_matrix[0], end = '\n\n')

    print('\nCheck reward matrix (from done) ...\n')
    print(pomdp.reward_matrix[-1], end = '\n\n')

    print('\nSimulate Example ...\n')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_1), end = '; ')
    print('Observation, Reward:', pomdp.step(action_1), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_2), end = '; ')
    print('Observation, Reward:', pomdp.step(action_2), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_3), end = '; ')
    print('Observation, Reward:', pomdp.step(action_3), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_4), end = '; ')
    print('Observation, Reward:', pomdp.step(action_4), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_5), end = '; ')
    print('Observation, Reward:', pomdp.step(action_5), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_6), end = '; ')
    print('Observation, Reward:', pomdp.step(action_6), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

if TEST_MDP:
    pomdp.state_index = 0

    print('\nSimulate Example ...\n')
    print(mdp.belief)

    print('Action:', str(action_1), end = '; ')
    _, reward = mdp.step(action_1)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_2), end = '; ')
    _, reward = mdp.step(action_2)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_3), end = '; ')
    _, reward = mdp.step(action_3)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_4), end = '; ')
    _, reward = mdp.step(action_4)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_5), end = '; ')
    _, reward = mdp.step(action_5)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_6), end = '; ')
    _, reward = mdp.step(action_6)
    print(mdp.belief, end = ' ')
    print(reward)
