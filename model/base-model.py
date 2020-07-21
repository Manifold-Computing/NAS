import torch

class BaseCNN:
    """
    Creates a simple model.
    """

    def __init__(self, actions):

        # unpack all the actions
        self.kernel_1 = actions[0]
        self.filters_1 = actions[1]
        self.kernel_2 = actions[2]
        self.filters_2 = actions[3]
        self.kernel_3 = actions[4]
        self.filters_3 = actions[5]
        self.kernel_4 = actions[6]
        self.filters_4 = actions[7]

        self.filters_0 = 0
        self.out = 10

    
    def create_model(self):
        return torch.nn.Sequential(
            # Conv Layer 1
            torch.nn.Conv2d(in_channels=self.filters_0, out_channels=self.filters_1, kernel_size=self.kernel_1, stride=2)
            # Activation 1
            torch.nn.ReLU()
            # Conv Layer 2
            torch.nn.Conv2d(in_channels=self.filters_1, out_channels=self.filters_2, kernel_size=self.kernel_2, stride=1)
            # Activation 2
            torch.nn.ReLU()
            # Conv Layer 3
            torch.nn.Conv2d(in_channels=self.filters_2, out_channels=self.filters_3, kernel_size=self.kernel_3, stride=2)
            # Activation 3
            torch.nn.ReLU()
            # Conv Layer 4
            torch.nn.Conv2d(in_channels=self.filters_3, out_channels=self.filters_4, kernel_size=self.kernel_4, stride=1)
            # Activation 4
            torch.nn.ReLU()
            # Average Pool
            torch.nn.AvgPool2d()
            # Output layer
            torch.nn.Linear(self.out)
        )