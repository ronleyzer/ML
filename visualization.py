import numpy as np
from matplotlib import pyplot as plt


def scatter_plot_with_text(df, text, font_size, title=None, ylabel=None, xlabel=None):
    xs = df[:, 0]
    ys = df[:, 1]
    plt.scatter(xs, ys, alpha=0.5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # Annotate the points
    for x, y, text in zip(xs, ys, text):
        plt.annotate(text, (x, y), fontsize=font_size, alpha=0.5)
    plt.show()