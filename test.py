from Utilities import *
import torch

x = torch.FloatTensor([1.,2.,3.])

y = np.array([1.,2.,3.])


print(f'division: {x/2}')
print(f"test softmax: {softmax(x, 1)}")
print(f'argmax on numpy: {argmax(y)}')
print(f'argmax on tensors: {argmax(x)}')