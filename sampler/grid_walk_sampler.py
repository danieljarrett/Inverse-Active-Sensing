from .mcmc_sampler import *

class GridWalkSampler(MCMCSampler):
    def __init__(self,
        dim   : int       ,
        learn : List[bool],
        delta : float     ,
        lower : float     ,
        upper : float     ,
    ):
        super(GridWalkSampler, self).__init__(dim, learn)

        self.delta = delta
        self.lower = lower
        self.upper = upper

    def step(self,
        location : np.array,
    ) -> np.ndarray:
        location_prime = location.copy()

        coord = np.random.choice(self.dim, p = self.dist)
        delta = np.random.choice([-self.delta, self.delta])

        if self.lower <= location_prime[coord] + delta <= self.upper:
            location_prime[coord] += delta

        return location_prime
