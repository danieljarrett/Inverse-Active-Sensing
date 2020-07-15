from .bayesian_learner import *

from sampler import MCMCSampler
from register import BaseRegister

PICK_SCENE = 5

class Example2Learner(BayesianLearner):
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
        super(Example2Learner, self).__init__(
            agent,
            trajs,
            prior,
            itemp,
        )

        self.burns    = burns
        self.iters    = iters
        self.sampler  = sampler
        self.register = register

    def betas(self, row):
        if PICK_SCENE == 5:
            eta_a = [-0.25, -0.75]
            eta_b = [-1.00, -1.00]
            eta_c = [-0.25]
            eta_d = [ 0.00] * 3

        self.beta_possible = np.array([
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
        ])
        self.log_posteriors = np.zeros([13])

        for i in range(13):
            print('Computing %d of %d ...' % (i, 13))

            self.itemp = self.beta_possible[i]

            if PICK_SCENE == 5:
                eta = eta_a + eta_b + eta_c + eta_d

            self.log_posteriors[i] = self.log_posterior(eta)

        np.save('volume/betas/decision.betas' + str(row), self.log_posteriors)

    def compute(self):
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

        self.eta_possible = -np.array(list(range(0, 105, 5))) / 100.
        self.log_posteriors = np.zeros([21, 21])

        for i in range(21):
            for j in range(21):
                print('Computing %d of %d ...' % (i * 21 + j, 21 * 21))

                eta_s = [self.eta_possible[i], self.eta_possible[j]]

                if PICK_SCENE in [4, 6]:
                    eta = eta_a + eta_b + eta_s + eta_d

                if PICK_SCENE in [5, 7]:
                    eta = eta_s + eta_b + eta_c + eta_d

                self.log_posteriors[i, j] = self.log_posterior(eta)

        np.save('volume/decision.logps', self.log_posteriors)

    def inverse(self):
        if PICK_SCENE in [4, 6]:
            eta_a = [-0.50, -0.50]
            eta_b = [-0.50, -0.50]
            eta_c = [-0.50, -0.50]
            eta_d = [ 0.00] * 3

        if PICK_SCENE == 5:
            eta_a = [-0.50, -0.50]
            eta_b = [-1.00, -1.00]
            eta_c = [-0.25]
            eta_d = [ 0.00] * 3

        if PICK_SCENE == 7:
            eta_a = [-0.50, -0.50]
            eta_b = [-1.00, -1.00]
            eta_c = [-0.50, -0.50]
            eta_d = [ 0.00] * 3

        eta = eta_a + eta_b + eta_c + eta_d

        eta_bar = np.array(eta)

        for iter in range(1, self.iters + 1):
            print('Gridwalking %d of %d ...' % (iter, self.iters))
            eta_prime = self.sampler.step(eta)

            ratio = self.accept_ratio(eta, eta_prime)

            if np.random.uniform() < min([1.0, ratio]):
                eta = eta_prime

            if iter > self.burns:
                samples = iter - self.burns
                eta_bar = self.iterate_mean(eta_bar, eta, samples)

                if PICK_SCENE in [4, 6]:
                    self.register.read(
                        samples     = samples   ,
                        eta_1       = eta[4]    ,
                        eta_2       = eta[5]    ,
                        eta_1_bar   = eta_bar[4],
                        eta_2_bar   = eta_bar[5],
                        ratio       = ratio     ,
                    )

                if PICK_SCENE in [5, 7]:
                    self.register.read(
                        samples     = samples   ,
                        eta_1       = eta[0]    ,
                        eta_2       = eta[1]    ,
                        eta_1_bar   = eta_bar[0],
                        eta_2_bar   = eta_bar[1],
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
        if PICK_SCENE in [4, 6]:
            eta_1 = eta[4]
            eta_2 = eta[5]
            eta_1_prime = eta_prime[4]
            eta_2_prime = eta_prime[5]

        if PICK_SCENE in [5, 7]:
            eta_1       = eta[0]
            eta_2       = eta[1]
            eta_1_prime = eta_prime[0]
            eta_2_prime = eta_prime[1]

        i       = int(-eta_1 / 0.05)
        j       = int(-eta_2 / 0.05)
        i_prime = int(-eta_1_prime / 0.05)
        j_prime = int(-eta_2_prime / 0.05)

        log_posterior = self.log_posteriors[i, j]
        log_posterior_prime = self.log_posteriors[i_prime, j_prime]

        return np.exp(log_posterior_prime - log_posterior)

    def iterate_mean(self,
        eta_bar : np.ndarray,
        eta     : np.ndarray,
        samples : int       ,
    ) -> np.ndarray:
        prev_sum = eta_bar * (samples - 1)
        next_sum = prev_sum + eta

        return next_sum / samples
