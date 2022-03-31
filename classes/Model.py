import torch
from torch import nn
from Utilities import argmax


class NN(nn.Module):
    """ Generic NN model.
        This is not used anymore.
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers, neurons_per_layer):
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
        super(NN, self).__init__()
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, output_dim),
        )


class ConvNet(NN):
    def __init__(self, input_c, output_dim, dueling=False):
        super(NN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.v_output = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        if dueling:
            self.forward = self.forward_dueling
    
    def forward_dueling(self, x):
        features = self.hidden_layers(x)
        q_values = self.output_layer(features)
        v_value = self.v_output(features)
        return v_value + (q_values - torch.sum(q_values) / q_values.shape[-1])


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
    def __init__(self, input_c, output_dim, dueling=False, side_clf=True):
        super(NN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 16, kernel_size=5, stride=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            # nn.BatchNorm2d(16),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            # nn.Dropout(0.2),
            nn.ReLU(),  # latent vector (not flattened)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=3),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.side_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 16),
            # nn.ReLU(),
            nn.Linear(16, output_dim),
        )

        self.v_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1),
        )

        if side_clf:
            self.decoder = self.side_output

        if dueling:
            self.forward = self.forward_dueling

    def forward_ssl(self, x):
        """ Self-supervised learning forward pass
            thorugh encoder and decoder
        """
        latent_vector = self.encoder(x)
        # print(latent_vector.shape)
        decoded_vector = self.decoder(latent_vector)
        # print(decoded_vector.shape)
        # exit()
        return decoded_vector

    def forward(self, x):
        """ Fine-tuning forward pass
            thorugh encoder and output head
        """
        latent_vector = self.encoder(x)
        output = self.output_head(latent_vector)
        return output

    def forward_dueling(self, x):
        latent_vector = self.encoder(x)
        q_values = self.output_head(latent_vector)
        v_value = self.v_output(latent_vector)
        return v_value + (q_values - torch.sum(q_values) / q_values.shape[-1])


class ICM(NN):
    def __init__(self, input_c, output_dim):    
        super(NN, self).__init__()

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
            nn.ReLU(),
            nn.Flatten()  # latent vector
        )
        
        self.action_output = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Softmax(dim=1)
        )

        self.feature_output = nn.Sequential(
            nn.Linear(1024 + 1, 512),
            nn.Linear(512, 1024)
        )
    
    def forward_inverse(self, x1, x2):
        """ Forward pass through network
        """
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        #concatenate x1 and x2 into a single vector
        x = self.action_output(torch.cat((x1,x2), dim=1))
        return x1, x2, x  # returns feature vector (s_t) and predicted action a

    def forward_feature(self, x, a):
        """x must be the concatenation between feature vector (s_t)
           and action a
        """
        return self.feature_output(torch.cat((x, a.unsqueeze(0)), dim=1))

    def loss_feature(self, x, y):
        return torch.pow(torch.norm(y - x, p=2), 2) / 2
