from .base_handler import *

class POMDPHandler(BaseHandler):
    def __init__(self,
        pomdp : BasePOMDP,
        gamma : float    ,
    ):
        self.pomdp = pomdp
        self.gamma = gamma

    def write(self,
        fullname : str,
    ):
        discount     = self.gamma
        values       = 'reward'
        states       = self.pomdp.num_states
        actions      = self.pomdp.num_actions
        observations = self.pomdp.num_observations

        with open(fullname, 'w') as out:
            out.write('discount     : ' + str(discount    ) + '\n')
            out.write('values       : ' + str(values      ) + '\n')
            out.write('states       : ' + str(states      ) + '\n')
            out.write('actions      : ' + str(actions     ) + '\n')
            out.write('observations : ' + str(observations) + '\n')

            for action in range(self.pomdp.num_actions):
                out.write('\n')
                out.write('T : ' + str(action) + '\n')

                for state in range(self.pomdp.num_states):
                    out.write(ary2str(self.pomdp.transition_matrix[state, action]))
                    out.write('\n')

            for action in range(self.pomdp.num_actions):
                out.write('\n')
                out.write('O : ' + str(action) + '\n')

                for state in range(self.pomdp.num_states):
                    out.write(ary2str(self.pomdp.emission_matrix[action, state]))
                    out.write('\n')

            out.write('\n')

            for action in range(self.pomdp.num_actions):
                for start in range(self.pomdp.num_states):
                    for end in range(self.pomdp.num_states):
                        out.write('R : ' + str(action) + ' : ' + \
                            str(start) + ' : ' + str(end) + ' : * ')

                        out.write(str(self.pomdp.reward_matrix[start, action, end]))
                        out.write('\n')

            out.close()

    def read(self,
        path : str,
    ) -> List[Tuple[np.ndarray, int]]:
        with open(get_latest(path, ['alpha', 'value']), 'r') as inp:
            dump = [e.strip() for e in inp.readlines()]

        tups = [tuple(dump[i:i + 3]) for i in range(0, len(dump), 3)]

        return [(str2ary(vec), int(act)) for (act, vec, _) in tups]
