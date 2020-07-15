from .base_register import *

class LiveRegister(BaseRegister):
    def __init__(self,
        variables     : List[str],
        save_interval : int      ,
        plot_interval : int      ,
        fullname      : str      ,
    ):
        super(LiveRegister, self).__init__(
            variables     = variables    ,
            save_interval = save_interval,
            fullname      = fullname     ,
        )

        self.plot_interval = plot_interval

    def to_plot(self,
        index : int,
    ) -> bool:
        if not self.plot_interval:
            return True
        else:
            return (index + 1) % self.plot_interval == 0

    def plot(self,
        index : int,
    ):
        raise NotImplementedError

    def on(self):
        self._set(True)

    def off(self):
        self._set(False)

    def _set(self,
        interactive : bool,
    ):
        if interactive:
            plt.ion()
        else:
            plt.ioff()
            plt.show()
