import numpy as np
import torch


class Controller:
    """
    This class is used to manage the controller.
    Provides utility functions for running NAS.
    """

    def __init__(self, num_layers, state_space, 
                 reg=0.001, gamma=0.99, eps=0.8, 
                 cells=32, embedding_dim=20, 
                 clip_norm=0.0, restore_controller=False):

        # number of layers.
        self.num_layers = num_layers
        # state space.
        self.state_space = state_space
        # size of state_space
        self.state_size = self.state_space.size
        # cells for rnn.
        self.cells = cells
        # embedding dimension.
        self.emb_dim = embedding_dim
        # regularization.
        self.reg = reg
        # discount factor.
        self.gamma = gamma
        # epsilon for exploration.
        self.eps = eps
        # restoring controller training.
        self.restore_controller = restore_controller
        # clipping norm.
        self.clip_norm = clip_norm

        # lists to store rewards and states
        self.reward_list = []
        self.state_list = []
        # list to store cell outputs
        self.cell_out = []
        self.policy_classifiers = []
        self.policy_actions = []
        self.policy_labels = []

        # create policy
        self.build_policy()
    
    def get_action(self, state):
        '''
        This function returns an action list, either through random 
        sampling or from the Controller RNN
        
        params:
            - state (StateSpace state): a list of one hot encoded states. 
                                        The first value is used as initial
                                        state for the controller RNN
        returns:
            - actions (list): A one hot encoded action list
        '''

        if np.random.random() < self.eps:

            # Generate a random action to explore.
            actions = self.state_space.get_random_state_space(self.state_size, self.num_layers)

            return actions

        else:
            # Use to controller to output an action.
            initial_state = self.state_space[0]
            size = initial_state['size']

            if state[0].shape != (1, size):
                state = state[0].reshape((1, size)).astype('int32')
            else:
                state = state[0]

            # return predictions
            pred_actions = self.apply_policy(state)
                
            return pred_actions
    
    def apply_policy(self, state):
        """
        This function returns a list of actions given a state.
        It applies the policy to the state to predict actions.

        params:
            - state:
        returns:
            - action:
        """
        pass
