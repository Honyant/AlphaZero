
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.to_play = 0
        self.visits = 0
        self.value = 0
        self.mean_value = 0
        self.probablity = 0

    def add_child(self, child_state, action):
        child = Node(child_state, self, action)
        self.children.append(child)

    def update(self, reward):
        self.value += reward
        self.visits += 1
        self.mean_value = self.value / self.visits
    
    def fully_expanded(self):
        return len(self.children) == len(self.state.get_moves())

    def __repr__(self):
        return f"<Node {self.state} {self.action}>"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.children:
            s += c.tree_to_string(indent + 1)
        return s

    def indent_string(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.children:
            s += str(c) + "\n"
        return s

#testing:
# root = Node(1, action=0)
# root.add_child(2, 'a')
# root.add_child(3, 'b')
# children = root.children
# children[0].add_child(4, 'g')
# children[0].add_child(5, 'h')
# children[1].add_child(6, 'i')

# print(root.children_to_string())
# print(root.tree_to_string(0))