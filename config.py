from types import SimpleNamespace
import torch

def config():

  # check if gpu is available
  is_cuda = torch.cuda.is_available()

  
  return SimpleNamespace(
    # number of layers of the state space
    num_layers = 4,  
    # maximum number of models generated
    max_models = 250,  
    # maximum number of epochs to train
    epochs = 10,  
    # batchsize of the child models
    child_batch_size = 128,  
    # high exploration for the first 1000 steps
    exploration = 0.8,  
    # regularization strength
    regularization = 1e-3, 
    # number of cells in RNN controller 
    controller_cells = 32,  
    # dimension of the embeddings for each state
    embedding_dims = 20,  
    # beta value for the moving average of the accuracy
    accuracy_beta = 0.8, 
    # clip rewards in the [-0.05, 0.05] range 
    clip_rewards = 0.0,  
    # restore controller to continue training
    restore_controller = True,
    # set num workers for pytorch loaders
    num_workers=4,
    # set device to cpu/gpu
    device=torch.device("cuda" if is_cuda else "cpu"),
  )
