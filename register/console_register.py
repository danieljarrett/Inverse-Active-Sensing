from .live_register import *

class ConsoleRegister(LiveRegister):
    def plot(self,
        index : int,
    ):
        row = {}
        lag = {}

        for variable in self.variables:
            values = self.data[variable]

            row[variable] = values[-1]

            if len(values) >= 100:
                lag[variable] = np.mean(values[-100:])
            else:
                lag[variable] = 0

        buf = []

        for key, value in row.items():
            buf.append(key + ': %.3f   ' % value)

        print(''.join(buf))
