from torch import nn


class NN(nn.Module):
    """ Generic NN model
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
        # Model architecture
        super(NN, self).__init__()
        
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
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class MLP(NN):
    def __init__(self, input_dim, output_dim):
        # Model architecture
        super(NN, self).__init__()
        
        # Create hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Create output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, output_dim),
        )


class ConvNet(NN):
    def __init__(self, input_c, output_dim):
        # Model architecture
        super(NN, self).__init__()

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


class SSLConvNet(NN):
    """ Self-supervised learning autoencoder convolutional network
        Architecture:
        - the encoder produces through convolutional layers
        a low-dimensional latent vector given two grayscale frames
        stacked toghether in 2 channels;
        - the decoder decodes the latent vector into the original input
        frames using convolutional transpose layers.
        An additional output head, composed of a single fully-connected
        layer from the latent vector dim to the action space dim,
        is defined for fine-tuning the model with the deep Q learning
        algorithm. The decoder is not used in this phase.

        forward_ssl(): used when performing self-supervised learning
        forward(): used when fine-tuning the model with deep RL
    """
    def __init__(self, input_c, output_dim):
        # Model architecture
        super(NN, self).__init__()

        # Create encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.2),
            nn.ReLU(),  # latent vector (not flattened)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Create output layer
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward_ssl(self, x):
        """ Self-supervised learning forward pass
            thorugh encoder and decoder
        """
        latent_vector = self.encoder(x)
        decoded_vector = self.decoder(latent_vector)
        return decoded_vector

    def forward(self, x):
        """ Fine-tuning forward pass
            thorugh encoder and output head
        """
        latent_vector = self.encoder(x)
        output = self.output_head(latent_vector)
        return output