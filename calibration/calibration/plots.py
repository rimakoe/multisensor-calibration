import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from calibration.datatypes import *


def plot_heatmap(title: str, transform: Transform):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    fig, ax = plt.subplots()
    im = ax.imshow(transform.covariance, cmap="seismic", norm=colors.CenteredNorm())
    for i in range(transform.covariance.shape[0]):
        for j in range(transform.covariance.shape[1]):
            text = ax.text(j, i, np.round(transform.covariance[i, j] * 1e6, 2), ha="center", va="center", color="k")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), labels=labels, rotation=0, ha="right", rotation_mode="anchor")
    ax.set_title(title)
    plt.tight_layout()
    plt.show(block=True)
