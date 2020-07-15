from .envs.pomdp import BasePOMDP

from gym.envs.registration import register

try: # NOTE: Toggle for package use
    register(
        id = 'diagnosis-v0',
        entry_point='pomdp_gym.envs:DiagnosisEnv',
    )
except:
    pass

try: # NOTE: Toggle for package use
    register(
        id = 'decision-v0',
        entry_point='pomdp_gym.envs:DecisionEnv',
    )
except:
    pass
