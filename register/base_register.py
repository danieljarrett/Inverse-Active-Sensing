from .__head__ import *

class BaseRegister:
    def __init__(self,
        variables     : List[str],
        save_interval : int      ,
        fullname      : str      ,
    ):
        self.variables     = variables
        self.save_interval = save_interval
        self.fullname      = fullname

        self.data = {variable: [] for variable in self.variables}

    def read(self, **kwargs):
        for variable in kwargs:
            if variable in self.variables:
                self.data[variable].append(kwargs[variable])
            else:
                raise KeyError

    def to_save(self,
        index : int,
    ) -> bool:
        if not self.save_interval:
            return True
        else:
            return (index + 1) % self.save_interval == 0

    def save(self):
        np.save(self.fullname, self.data)
