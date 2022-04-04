import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(y, window=35, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def add_plot(title, label, y):
    ax.set_xlabel('Episode')
    ax.set_ylabel('Step')
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.plot(smooth(y), label=label)
    else:
        ax.plot(smooth(y))
    ax.legend()

name = "CNN"
fig, ax = plt.subplots()
# ["MLP_base", "MLP_buffer_target", "MLP_buffer_target_novelty"]
for conf in ["CNN_base", "SSL_base", "TL_base"]:
    steps = np.load("./exp_results/"+conf+".npy")
    add_plot("Vision CartPole with CNN - Steps per episode", conf.replace("_", " "), steps)
fig.savefig(name+".png", dpi=300)