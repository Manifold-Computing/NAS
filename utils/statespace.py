import pprint
import numpy as np

from collections import OrderedDict

class StateSpace:
    """
    This class manages the states space.
    It contains utility functions for storing states/actions.
    These states and actions are needed by the controller during training.
    This class provides structure for defining the state space.
    """

    def __init__(self):
        self.states = OrderedDict()
        self.state_count = 0
    

    def add_state(self, name, values):
        """
        This function is used to add a state to the state-manager.
        It also adds some metadata which provides utility to the controller.

        params:
            - name (string) : name for the state.
            - values (list) : valid values for the state.
        returns:
            - state_id (int) : a global id used to index the state.
        """

        # dictionaries for mapping idx-val/val-idx.
        idx_to_val, val_to_idx = {}, {}

        # populate dictionaries.
        for i, val in enumerate(values):
            idx_to_val[i] = val
        
        for i, val in enumerate(values):
            val_to_idx[val] = i
        
        # store metadata.
        metadata = {
            "id" : self.state_count,
            "name" : name,
            "values" : values,
            "size": len(values),
            "idx_to_val": idx_to_val,
            "val_to_idx" : val_to_idx
        }
        self.states[self.state_count] = metadata
        # updata state count.
        self.state_count += 1

        return self.state_count - 1
    
    def embedding_encode(self, state_id, value):
        """
        This function creates an embedding for a specific state value
        params:
            - state_id (int): global state id.
            - value (int): state value.
        returns:
            - embedding (np.ndarray) : embedding encoded representation 
                                       of the state value.
        """
        # get state metadata for given
        # state_id.
        state = self.states[state_id]
        size = state["size"]
        value_to_idx = state["value_to_idx"]
        value_idx = value_to_idx[value]

        # create embedding
        embedding = np.zeros((1, size), dtype=np.float32)
        embedding[np.arange(1), value_idx] = value_idx + 1

        return embedding
    
    def get_state_value(self, state_id, idx):
        """
        This function fetches the state value for a given id

        params:
            - state_id (int): global state id.
            - idx (int): index of the state value (usually from argmax)
        Returns:
            - value (int): The state value at given index.
        """
        # retreive state metadata using id.
        state = self.states[state_id]
        index_map = state['idx_to_val']

        # sanity check
        if (type(idx) == list or type(idx) == np.ndarray) and len(idx) == 1:
            idx = idx[0]
        # get value
        value = index_map[idx]

        return value

    def get_random_state_space(self, num_layers):
        """
        This function generates a random initial state space. 
        This can be fed as an initial value to the Controller.

        params:
            - num_layers (int): number of layers to duplicate the search space.
        returns:
            - states (list) : A list of one hot encoded states
        """
        # list to hold states
        states = []

        for state_id in range(self.size * num_layers):
            # retreive state metadata using id.
            state = self.states(state_id)
            size = state['size']

            # sample random idx.
            sample_idx = np.random.choice(size, size=1)[0]
            # retreive value using map.
            sample_value = state['idx_to_val'][sample_idx]
            # create embedding
            state = self.embedding_encode(state_id, sample_value)
            # add to list
            states.append(state)

        return states

    def parse_state_space_list(self, state_list):
        """
        This function parses a list of one hot encoded 
        states and returns a list of state values.

        params:
            - state_list (list): list of one hot encoded states.
        returns:
            - state_values (list): list of state values.
        """
        # create list to store values.
        state_values = []

        for state_id, state_one_hot in enumerate(state_list):
            # get state index using state_id.
            state_idx = np.argmax(state_one_hot, axis=-1)[0]
            # get value for state_idx.
            state_value = self.get_state_value(state_id, state_idx)
            # append state values.
            state_values.append(state_value)

        return state_values

    def print_state_space(self):
        """This function pretty prints the state space."""

        print("*" * 40, "STATE SPACE", "*" * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for _, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        """ This function prints the action space """

        print("Actions :")

        for state_id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((state_id + 1) // self.size) + 1), "*" * 20)

            state = self.states[state_id]
            name = state['name']
            values = [(n, p) for n, p in zip(state["values"], *action)]
            print("%s : " % name, values)
        print()

    def __getitem__(self, state_id):
        return self.states[state_id % self.size]

    @property
    def size(self):
        return self.state_count
