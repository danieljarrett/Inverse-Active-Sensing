from .__head__ import *

from ..pomdp import DiagnosisPOMDP
from ..mdp import DiagnosisMDP

tree_factor  = 2
tree_height  = 4
num_diseases = tree_factor ** (tree_height - 1)
num_tests    = int((tree_factor ** (tree_height - 1) - 1) / (tree_factor - 1))
prob_comp    = [ 0.05] * num_diseases + [1.0]
prob_fail    = [ 0.10] * num_tests
phi_test     = 1
phi_comp     = 1
phi_mark     = 1
theta_test   = [- 1.0] * num_tests
theta_comp   = [- 2.0] * num_diseases
theta_mark   = [-10.0] * num_diseases
omega        = [  0.0] * num_diseases + [0.0]

TEST_POMDP = True
TEST_MDP   = True

pomdp = DiagnosisPOMDP(
    tree_factor,
    tree_height,
    prob_comp  ,
    prob_fail  ,
    phi_test   ,
    phi_comp   ,
    phi_mark   ,
    theta_test ,
    theta_comp ,
    theta_mark ,
)

mdp = DiagnosisMDP(
    pomdp,
    omega,
)

action_1_index = pomdp.action2index([0])
action_2_index = pomdp.action2index([0, 0])
action_3_index = pomdp.action2index([0, 0, 0])
action_4_index = pomdp.action2index([0, 0, 0, 0])

if TEST_POMDP:
    pomdp.state_index = 0

    print('\nVerify action indexing ...\n')
    for index in range(pomdp.num_actions):
        print(index, end = ' ')
        print(pomdp.index2action(index), end = ' ')
        print(pomdp.action2index(pomdp.index2action(index)))

    print('\nCheck transitions (test actions) ...\n')
    print(pomdp.transition_matrix[:, pomdp.root_action, :])

    print('\nCheck transitions (stop actions) ...\n')
    print(pomdp.transition_matrix[:, pomdp.num_tests  , :])

    print('\nCheck emissions (test action) ...\n')
    for action in [[0], [0, 1], [0, 1, 1]]:
        print(pomdp.emission_matrix[pomdp.action2index(action)])

    print('\nCheck emissions (stop action) ...\n')
    print(pomdp.emission_matrix[pomdp.action2index([0, 1, 1, 0])])

    print('\nCheck reward matrix (from disease) ...\n')
    print(pomdp.reward_matrix[0])

    print('\nCheck reward matrix (from done) ...\n')
    print(pomdp.reward_matrix[-1])

    print('\nSimulate Example ...\n')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_1_index), end = '; ')
    print('Observation, Reward:', pomdp.step(action_1_index), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_2_index), end = '; ')
    print('Observation, Reward:', pomdp.step(action_2_index), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_3_index), end = '; ')
    print('Observation, Reward:', pomdp.step(action_3_index), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

    print('Action:', str(action_4_index), end = '; ')
    print('Observation, Reward:', pomdp.step(action_4_index), end = '; ')
    print('State:', pomdp.state_index, end = '; ')
    print('Done:', pomdp.done)

if TEST_MDP:
    pomdp.state_index = 0

    print('\nSimulate Example ...\n')
    print(mdp.belief)

    print('Action:', str(action_1_index), end = '; ')
    _, reward = mdp.step(action_1_index)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_2_index), end = '; ')
    _, reward = mdp.step(action_2_index)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_3_index), end = '; ')
    _, reward = mdp.step(action_3_index)
    print(mdp.belief, end = ' ')
    print(reward)

    print('Action:', str(action_4_index), end = '; ')
    _, reward = mdp.step(action_4_index)
    print(mdp.belief, end = ' ')
    print(reward)
