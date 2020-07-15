from .trainable_agent import *

class MutableAgent(TrainableAgent):
    def store_value_function(self):
        raise NotImplementedError

    def restore_value_function(self):
        raise NotImplementedError
