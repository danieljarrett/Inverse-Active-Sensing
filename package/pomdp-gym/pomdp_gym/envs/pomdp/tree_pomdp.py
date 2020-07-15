from .base_pomdp import *

class TreePOMDP(BasePOMDP):
    def __init__(self,
        tree_factor : int,
        tree_height : int,
    ):
        self.tree_factor = tree_factor
        self.tree_height = tree_height

        self.root_action = 0

        super(TreePOMDP, self).__init__()

    def action2index(self,
        action : List[int],
    ) -> int:
        index = 0

        max_level = len(action) - 1

        if max_level:
            for level in range(1, max_level):
                index += self.tree_factor ** level
                index += action[level] * self.tree_factor ** (max_level - level)

            index += action[max_level] + 1

        return index

    def index2action(self,
        index : int,
    ) -> List[int]:
        action = []

        q = index

        while q:
            q, r = divmod(q - 1, self.tree_factor)

            action.insert(0, r)

        action.insert(0, self.root_action)

        return action
