from .__head__ import *

class BaseDensity:
    def __init__(self,
        dim : int,
    ):
        self.dim = dim

    def pdf(self,
        vector : np.ndarray,
    ) -> float:
        return np.prod([self.dist.pdf(value) for value in vector])

    def log_pdf(self,
        vector : np.ndarray,
    ) -> float:
        return np.sum([self.dist.logpdf(value) for value in vector])

    def sample(self) -> np.ndarray:
        sample = self.dist.rvs(size = self.dim)

        return np.around(sample, decimals = 1)
