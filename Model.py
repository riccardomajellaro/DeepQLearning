import torch
from torch import nn
import torchvision.transforms as transforms

class DNN(nn.Module):
    # TODO: Add convolutional layers
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Model architecture
        super(DNN, self).__init__()
        
        # Create input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, neurons_per_layer),
            nn.ReLU()
        )
        
        # Create hidden layers
        hidden_layers = []
        for hidden_layer in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        del(hidden_layers)

        # Create output layer
        self.output_layer = nn.Sequential(
            nn.Linear(neurons_per_layer, output_dim)
        )
        

    def forward(self, x):
        """ Forward pass
        """
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

