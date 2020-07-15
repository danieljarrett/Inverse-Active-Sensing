from .live_register import *

class GraphicRegister(LiveRegister):
    def plot(self,
        index : int,
    ):
        plt.figure(num = 1, figsize = (20, 5))
        plt.clf()

        for plot, variable in enumerate(self.variables):
            values = self.data[variable]

            plt.subplot(int('13' + str(plot + 1)))
            plt.title(('index %s; ' % (index + 1)) + variable)
            plt.plot(values)

            if len(values) >= 100:
                values = torch.tensor(values, dtype = torch.float)
                means = values.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait = True)
            display.display(plt.gcf())
