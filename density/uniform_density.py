from .base_density import *

class UniformDensity(BaseDensity):
    def __init__(self,
        dim   : int  ,
        lower : float,
        upper : float,
    ):
        super(UniformDensity, self).__init__(dim)

        self.lower = lower
        self.upper = upper

        self.dist = st.uniform(
            loc   = lower,
            scale = upper - lower,
        )
