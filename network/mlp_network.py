from .base_network import *

class MLPNetwork(BaseNetwork):
    def __init__(self,
        in_dim  : int,
        out_dim : int,
    ):
        super(MLPNetwork, self).__init__()

        self.in_dim  = in_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim),
        )

    @classmethod
    def copy(self,
        network : 'MLPNetwork',
    ) -> 'MLPNetwork':
        copy = self(network.in_dim, network.out_dim)
        copy.load_state_dict(network.state_dict())

        return copy

    def forward(self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)
