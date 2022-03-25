import numpy as np
import torch

def sigmoid(x):
    return torch.sigmoid(x)

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp'
        x should be of type torch.FloatTensor
     '''
    x = x / torch.FloatTensor([temp]) # scale by temperature
    z = x - torch.max(x) # substract max to prevent overflow of softmax 
    return torch.exp(z) / torch.sum(torch.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(torch.where(x == torch.max(x))[0])
    except:
        return torch.argmax(x)

def preprocess_img(img_array):
    # TODO: Make a function that processes the images for the cnn model version
    pass