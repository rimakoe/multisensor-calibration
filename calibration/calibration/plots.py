import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
import matplotlib.colors as colors
from calibration.datatypes import *


def plot_heatmap_compact(title: str, frame: Frame):
    assert frame.covariance.shape == (6, 6), "Unable to plot. Covariance has unexpected shape."

    def _plot_3x3(ax: Axes, title: str, matrix: np.array, labels_x: list[str], labels_y: list[str]):
        ax.imshow(matrix[:3, :3], cmap="seismic", norm=colors.CenteredNorm())
        for i in range(3):
            for j in range(3):
                ax.text(j, i, np.round(matrix[i, j] * 1e6, 3), ha="center", va="center", color="k")
        ax.set_xticks(range(len(labels_x)), labels=labels_x, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels_y)), labels=labels_y, rotation=0, ha="right", rotation_mode="anchor")
        ax.set_title(title)

    label_rotation = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    label_translation = ["$t_x$", "$t_y$", "$t_z$"]

    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    _plot_3x3(ax1, "rotation", frame.covariance[:3, :3], labels_x=label_rotation, labels_y=label_rotation)
    ax2 = plt.subplot(132)
    _plot_3x3(ax2, "rotation / translation", frame.covariance[3:, :3], labels_x=label_rotation, labels_y=label_translation)
    ax3 = plt.subplot(133)
    _plot_3x3(ax3, "translation", frame.covariance[3:, 3:], labels_x=label_translation, labels_y=label_translation)
    plt.tight_layout()
    plt.show(block=True)


def plot_heatmap(title: str, frame: Frame):
    labels = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$", "$t_x$", "$t_y$", "$t_z$"]
    fig, ax = plt.subplots()
    im = ax.imshow(frame.covariance, cmap="seismic", norm=colors.CenteredNorm())
    for i in range(frame.covariance.shape[0]):
        for j in range(frame.covariance.shape[1]):
            text = ax.text(j, i, np.round(frame.covariance[i, j] * 1e6, 2), ha="center", va="center", color="k")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), labels=labels, rotation=0, ha="right", rotation_mode="anchor")
    # ax.set_title(title)
    plt.tight_layout()
    plt.show(block=True)
