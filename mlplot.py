from matplotlib import pyplot as plt
import numpy as np

def plot_gan_samples(samples, rows, cols):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=rows, ncols=cols, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2,2)), cmap='Greys_r')  
    return fig, axes

def plot_log():
    x = np.linspace(0.001, 0.999, 100)
    E_r = lambda x: -np.log(x)
    plt.plot(x, E_r(x), label=r'$- \, log \, (x)$')
    E_f = lambda x: -np.log(1 - x)
    plt.plot(x, E_f(x), label=r'$- \, log \, (1 - x)$')
    plt.legend(fontsize='large')
    plt.xlabel('Prediction')
    plt.ylabel('Error')
    return True