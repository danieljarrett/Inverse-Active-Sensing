from .base_density import *

class GaussianDensity(BaseDensity):
    def __init__(self,
        dim   : int  ,
        mu    : float,
        sigma : float,
    ):
        super(GaussianDensity, self).__init__(dim)

        self.dist = st.norm(
            loc   = mu   ,
            scale = sigma,
        )
