from .base_learner import *

from density import BaseDensity

class BayesianLearner(BaseLearner):
    def __init__(self,
        agent : BaseAgent                   ,
        trajs : List[Tuple[np.ndarray, int]],
        prior : BaseDensity                 ,
        itemp : float                       ,
    ):
        super(BayesianLearner, self).__init__(
            agent,
            trajs,
        )

        self.prior = prior
        self.itemp = itemp

    def log_prior(self,
        eta : np.ndarray,
    ) -> float:
        return self.prior.log_pdf(eta)

    def log_likelihood(self,
        eta : np.ndarray,
    ) -> float:
        self.configure_reward(eta)
        self.forward()

        log_likelihood = self._compute_log_likelihood()

        self.restore_reward()

        return log_likelihood

    def log_posterior(self,
        eta : np.ndarray,
    ) -> float:
        return self.log_likelihood(eta) + self.log_prior(eta)

    def _compute_log_likelihood(self) -> float:
        log_ener = 0.0
        log_part = 0.0

        for (state, action) in self.trajs:
            log_ener += self.itemp * self.agent.action_value(state, action)
            log_part += sp.special.logsumexp([self.itemp * self.agent.action_value(
                state, a) for a in range(self.agent.env.mdp.pomdp.num_actions)])

        return log_ener - log_part
