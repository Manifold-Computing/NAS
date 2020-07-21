import torch
from config import config
from data.cifar import CifarDataset
from utils.statespace import StateSpace
from model.controller import Controller
from model.base-model import BaseCNN

def reinforce():


if __name__ == "__main__":

    # create State Space
    state_space = StateSpace()

    # add states
    # add states
    state_space.add_state(name='kernel', values=[1, 3])
    state_space.add_state(name='filters', values=[16, 32, 64])

    # print the state space being searched
    state_space.print_state_space()

    # create dataset

    # create Controller

    # create a manager?
    # don't need a manager here just a workaround.

    # call reinforce loop
    reinforce()



