import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Model architecture
        super(DNN, self).__init__()

        layers = []

        # Add input layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(nn.ReLU())
        # Add hidden layers
        for hidden_layer in range(n_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.ReLU())
        # Add output layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        # Create model
        seq_model = nn.Sequential(*layers)

        def forward(self, x):
            logits = self.seq_model(x)
            return logits

