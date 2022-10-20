import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def weight_fn(x):
    return np.ones_like(x) / len(x)

def ratio_plot(y_true, y_preds, bins=None):
    fig = plt.figure(figsize=(8, 6))
    if bins is None:
        bins = np.linspace(-0.001, 0.001, 50)
    plt.hist(
        y_true-y_preds,
        bins=bins,
        histtype="step",
        fill=True,
        ec="k",
        alpha=0.5,
        weights=weight_fn(y_true)
    )

    plt.ylabel("Percentage of points")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    plt.xlabel(r"$\Delta = y_{\mathrm{true}} - y_{\mathrm{pred}}$")
    plt.tight_layout()
    return fig

def hexplot(y_true, born_test, y_preds, n_bins=50, x_lims=[-0.008, 0.008]):
    gs_kw = dict(width_ratios=[2, 1])
    fig, ax = plt.subplot_mosaic([['left', 'right']],
                                gridspec_kw=gs_kw, figsize=(8, 6), constrained_layout=True)
    ymin, ymax = 1.02*np.min(np.log10(born_test)), 0.9*np.max(np.log10(born_test))

    hexplot = ax["left"].hexbin(
        x=y_true-y_preds,
        y=born_test,
        gridsize=n_bins,
        bins="log",
        yscale="log",
        mincnt=1,
        extent=[x_lims[0], x_lims[1], ymin, ymax]
    )
    ax["right"].hist(born_test, bins=np.logspace(ymin, ymax, n_bins), orientation="horizontal")
    ax["right"].set_yscale("log")
    ax["right"].set_xticks([])
    ax["right"].set_yticks([])

    ax["left"].set_xlabel(r"$\Delta$")
    ax["left"].set_ylabel(r"$|\mathcal{M}_{\mathrm{tree}}|^{2}$")

    cbar = plt.colorbar(hexplot, ax=ax["left"], orientation="horizontal", location="bottom")
    cbar.ax.tick_params(pad=5)
    cbar.ax.set_xlabel("Number of points in bin")
    return fig
