from .bayesian_learner import *

from sampler import MCMCSampler
from register import BaseRegister

class BayesianMCMCLearner(BayesianLearner):
    def __init__(self,
        agent    : BaseAgent                   ,
        trajs    : List[Tuple[np.ndarray, int]],
        prior    : BaseDensity                 ,
        itemp    : float                       ,
        burns    : int                         ,
        iters    : int                         ,
        sampler  : MCMCSampler                 ,
        register : BaseRegister                ,
    ):
        super(BayesianMCMCLearner, self).__init__(
            agent,
            trajs,
            prior,
            itemp,
        )

        self.burns    = burns
        self.iters    = iters
        self.sampler  = sampler
        self.register = register

    def inverse(self):
        eta_a = [-1.70, -1.30]
        eta_b = [-1.90, -1.50]
        eta_c = [-0.50, -0.50]
        eta_d = [ 0.00] * 3
        eta = eta_a + eta_b + eta_c + eta_d

        eta_bar = np.array(eta)

        for iter in range(1, self.iters + 1):
            print('----------------------------------------------------------------------' + str(iter))
            eta_prime = self.sampler.step(eta)

            ratio = self.accept_ratio(eta, eta_prime)

            if np.random.uniform() < min([1.0, ratio]):
                eta = eta_prime

            if iter > self.burns:
                samples = iter - self.burns
                eta_bar = self.iterate_mean(eta_bar, eta, samples)

                self.register.read(
                    samples     = samples   ,
                    eta_c_1     = eta[4]    ,
                    eta_c_2     = eta[5]    ,
                    eta_c_1_bar = eta_bar[4],
                    eta_c_2_bar = eta_bar[5],
                    ratio       = ratio     ,
                )

                if self.register.to_plot(iter):
                    self.register.plot(iter)

                if self.register.to_save(iter):
                    self.register.save()

    def accept_ratio(self,
        eta       : np.ndarray,
        eta_prime : np.ndarray,
    ) -> float:
        log_posterior = self.log_posterior(eta)
        log_posterior_prime = self.log_posterior(eta_prime)

        return np.exp(log_posterior_prime - log_posterior)

    def iterate_mean(self,
        eta_bar : np.ndarray,
        eta     : np.ndarray,
        samples : int       ,
    ) -> np.ndarray:
        prev_sum = eta_bar * (samples - 1)
        next_sum = prev_sum + eta

        return next_sum / samples
