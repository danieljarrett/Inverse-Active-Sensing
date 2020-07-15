from .__head__ import *

class MCMCSampler:
    def __init__(self,
        dim   : int       ,
        learn : List[bool],
    ):
        self.dim  = dim
        self.dist = np.array(learn) / sum(np.array(learn))

    def step(self,
        location : np.array,
    ) -> np.ndarray:
        raise NotImplementedError
