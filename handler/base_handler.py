from .__head__ import *

class BaseHandler:
    def write(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError
