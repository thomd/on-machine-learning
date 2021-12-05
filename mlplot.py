from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def plot_gan_samples(samples, show=True):
    fig, axes = plt.subplots(figsize=(10, 2), nrows=1, ncols=len(samples), sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1 - img.reshape((2,2)), cmap='Greys_r')  
    if not show:
        plt.close(fig)
    return fig, axes

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

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

def plot_errors(generator_errors, discriminator_errors):
    plt.figure(figsize=(16, 5))
    plt.subplot(1,2,1)  
    plt.plot(generator_errors)
    plt.title("Generator Error")
    plt.xlabel("Epoche")
    plt.ylabel("Error")
    plt.subplot(1,2,2)
    plt.plot(discriminator_errors)
    plt.title("Discriminator Error")
    plt.xlabel("Epoche")
    plt.ylabel("Error")
    plt.show()