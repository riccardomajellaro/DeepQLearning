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
        self.output_head = nn.Sequential(
            nn.Linear(neurons_per_layer, output_dim)
        )

    def forward(self, x):
        """ Forward pass through network
        """
        x = self.hidden_layers(x)
        x = self.output_head(x)
        return x


class MLP(NN):
    """ Simple multi-layer perceptron
    """
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.output_head = nn.Sequential(
            nn.Linear(64, output_dim)
        )


class ConvNet(NN):
    """ Simple convolutional neural network
    """
    def __init__(self, input_c, output_dim, dueling=False):
        super(NN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.output_head = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # output head used in dueling dqn mode
        self.v_output = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        if dueling:
            self.forward = self.forward_dueling
    
    def forward_dueling(self, x):
        features = self.hidden_layers(x)
        q_values = self.output_head(features)
        v_value = self.v_output(features)
        return v_value + (q_values - torch.sum(q_values) / q_values.shape[-1])


class SSLConvNet(NN):
    """ Self-supervised learning autoencoder convolutional network
        Architecture:
        - the encoder produces through convolutional layers
        a low-dimensional latent vector given four grayscale frames
        stacked toghether in 4 channels;
        - the decoder decodes the latent vector into the original input
        frames using convolutional transpose layers.
        An additional output head, composed of fully-connected
        layers from the latent vector dim to the action space dim,
        is defined for fine-tuning the model with the deep Q learning
        algorithm. The decoder is not used in this phase.

        forward_ssl(): used when performing self-supervised learning
        forward(): used when fine-tuning the model with deep RL
    """
    def __init__(self, input_c, output_dim, dueling=False):
        super(NN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()  # latent vector (not flattened)
        )
        
        # decoder used during self-supervised learning
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=3),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=3),
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        # output head used on the original task
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # output head used in dueling dqn mode
        self.v_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.hidden_layers = self.encoder

        if dueling:
            self.forward = self.forward_dueling

    def forward_ssl(self, x):
        """ Self-supervised learning forward pass
            thorugh encoder and decoder
        """
        latent_vector = self.encoder(x)
        decoded_vector = self.decoder(latent_vector)
        return decoded_vector

    def forward_dueling(self, x):
        latent_vector = self.encoder(x)
        q_values = self.output_head(latent_vector)
        v_value = self.v_output(latent_vector)
        return v_value + (q_values - torch.sum(q_values) / q_values.shape[-1])


class TLConvNet(NN):
    """ Transfer learning using a convolutional network
        Architecture:
        - convolutional layers used both in pretraining and finetuning, having
        as input four grayscale frames stacked toghether in 4 channels;
        - classification head using 1 final node used for predicting the side
        where the pole is falling.
        An additional output head, composed of fully-connected
        layers from the final feature maps dim to the action space dim,
        is defined for fine-tuning the model with the deep Q learning
        algorithm. The side_output head is not used in this phase.

        forward_tl(): used when performing transfer learning
        forward(): used when fine-tuning the model with deep RL
    """
    def __init__(self, input_c, output_dim, dueling=False):
        super(NN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # output head for predicting left or right side
        # during the pretraining phase
        self.side_output = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # output head for finetuning on the original task
        self.output_head = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # output head used in dueling dqn mode
        self.v_output = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        if dueling:
            self.forward = self.forward_dueling

    def forward_tl(self, x):
        """ Forward pass during pretraining phase of transfer
            learning thorugh conv layers and side_output head
        """
        x = self.hidden_layers(x)
        x = self.side_output(x)
        return x

    def forward_dueling(self, x):
        latent_vector = self.hidden_layers(x)
        q_values = self.output_head(latent_vector)
        v_value = self.v_output(latent_vector)
        return v_value + (q_values - torch.sum(q_values) / q_values.shape[-1])


class ICM(NN):
    """ Intrinsic Curiosity Module (ICM) used in the
        curiosity-based exploration
    """
    def __init__(self, input_c, output_dim):    
        super(NN, self).__init__()

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()  # latent vector
        )

        self.action_output = nn.Sequential(
            nn.Linear(5184*2, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Softmax(dim=1)
        )

        self.feature_output = nn.Sequential(
            nn.Linear(5184 + 1, 512),
            nn.Linear(512, 5184)
        )
    
    def forward_inverse(self, x1, x2):
        """ Encode x1 and x2 as feature maps, concatenate
            them and pass them through the action head
        """
        x1 = self.hidden_layers(x1)
        x2 = self.hidden_layers(x2)
        #concatenate x1 and x2 into a single vector
        x = self.action_output(torch.cat((x1,x2), dim=1))
        return x1, x2, x  # returns feature vector (s_t) and predicted action a

    def forward_feature(self, x, a):
        """ x is concatenated with the action a
            and passed through the feature head
        """
        return self.feature_output(torch.cat((x, a.unsqueeze(0)), dim=1))

    def loss_feature(self, x, y):
        return torch.pow(torch.norm(y - x, p=2), 2) / 2
