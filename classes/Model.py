from torch import nn

from Utilities import softmax


class NN(nn.Module):
    """ Generic NN model
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
        # Model architecture
        super(NN, self).__init__()

        input_dim = input_dim
        output_dim = output_dim
        n_hidden_layers = n_hidden_layers
        neurons_per_layer = neurons_per_layer
        
        # Create hidden layers
        hidden_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                neurons_per_layer = input_dim
            hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            hidden_layers.append(nn.SELU())
            hidden_layers.append(nn.Dropout(0.2))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Create output layer
        if len(hidden_layers) == 0:
            neurons_per_layer = input_dim
        self.output_layer = nn.Sequential(
            nn.Linear(neurons_per_layer, output_dim)
        )

    def forward(self, x):
        """ Forward pass through network
        """
        x = x
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class MLP(NN):
    def __init__(self, input_dim, output_dim):
        # Model architecture
        super(NN, self).__init__()

        input_dim = input_dim
        output_dim = output_dim
        
        # Create hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # Create output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, output_dim),
        )

class ConvNet(NN):
    def __init__(self, input_h, input_w, input_c, output_dim):
        # Model architecture
        super(NN, self).__init__()

        input_h = input_h
        input_w = input_w
        input_c = input_c
        output_dim = output_dim

        # Create hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(input_c, 8, kernel_size=5, stride=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Create output layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, output_dim),
        )