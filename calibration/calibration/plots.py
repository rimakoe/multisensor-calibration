import numpy as np
from scipy.stats import chi2, norm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as Axes
import matplotlib.colors as colors
from calibration.datatypes import *


def plot_heatmap_compact(map: np.ndarray, limits: list[float] = [-1.0, 1.0], title: str = ""):
    assert map.shape == (6, 6), "Unable to plot. Covariance has unexpected shape."
    assert len(limits) == 2, "Scale expects only [min, max]"

    def _plot_3x3(ax: Axes, title: str, matrix: np.array, labels_x: list[str], labels_y: list[str], limits: list[float]):
        color_norm = colors.CenteredNorm()
        color_norm.vmin = limits[0]
        color_norm.vmax = limits[1]
        ax.imshow(matrix[:3, :3], cmap="bwr", norm=color_norm)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, np.round(matrix[i, j], 3), ha="center", va="center", color="k")
        ax.set_xticks(range(len(labels_x)), labels=labels_x, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels_y)), labels=labels_y, rotation=0, ha="right", rotation_mode="anchor")
        ax.set_title(title)

    label_rotation = ["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    label_translation = ["$\\nu_x$", "$\\nu_y$", "$\\nu_z$"]

    plt.figure(figsize=(9, 3))
    ax1 = plt.subplot(131)
    _plot_3x3(ax1, "rotation", map[:3, :3], labels_x=label_rotation, labels_y=label_rotation, limits=limits)
    ax2 = plt.subplot(132)
    _plot_3x3(ax2, "rotation / translation", map[3:, :3], labels_x=label_rotation, labels_y=label_translation, limits=limits)
    ax3 = plt.subplot(133)
    _plot_3x3(ax3, "translation", map[3:, 3:], labels_x=label_translation, labels_y=label_translation, limits=limits)
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


def plot_residual(df: pd.DataFrame, camera: Camera):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = norm.pdf(X, 0, 1) * norm.pdf(Y, 0, 1)
    sigma_levels = [3, 2, 1, 0]
    probs = [1 - np.exp(-(s**2) / 2) for s in sigma_levels]

    # Convert probability mass → contour heights
    # Find the Z value that encloses given probability
    def find_contour_level(pdf, prob):
        sorted_vals = np.sort(pdf.ravel())[::-1]
        cumsum = np.cumsum(sorted_vals)
        cumsum /= cumsum[-1]
        return sorted_vals[np.searchsorted(cumsum, prob)]

    contour_levels = [find_contour_level(Z, p) for p in probs]
    g = sns.jointplot(
        data=df,
        x="u",
        y="v",
        fill=True,
        kind="kde",
    )
    g.set_axis_labels(xlabel="$\\Delta u / \\sigma$", ylabel="$\\Delta v / \\sigma$")
    g.ax_joint.contour(X, Y, Z, levels=contour_levels, colors="black", linewidths=1, linestyles="--", alpha=0.5)
    perfect_distribution = np.linspace(-5, 5, 100)
    g.ax_marg_x.plot(perfect_distribution, norm.pdf(perfect_distribution, 0, 1), color="black", ls="--", lw=1, alpha=0.5)
    g.ax_marg_y.plot(norm.pdf(perfect_distribution, 0, 1), perfect_distribution, color="black", ls="--", lw=1, alpha=0.5)
    plt.show(block=True)


def plot_convergence(df: pd.DataFrame, target: str, config: str = "mean", reference: float = 0.0, relative_lim: float = 0.1):
    evolve = []
    references = []
    df["delta"] = df[target]
    for i in range(10, len(df[target]), 10):
        if config == "mean":
            evolve.append(np.mean(df["delta"].iloc[:i]))
        if config == "median":
            evolve.append(np.median(df["delta"].iloc[:i]))
        references.append(reference)
    evolve = np.array(evolve)
    plt.figure(figsize=(6, 6))
    plt.grid(True, which="major", linestyle="-")
    plt.minorticks_on()
    plt.grid(True, which="minor", linestyle=":")
    plt.ylabel(target)
    plt.ylim(reference * np.array([1.0 - relative_lim, 1.0 + relative_lim]))
    plt.xlabel("iterations")
    plt.plot(list(range(10, len(df[target]), 10)), evolve)
    plt.plot(list(range(10, len(df[target]), 10)), references, ":")
    plt.show(block=True)
