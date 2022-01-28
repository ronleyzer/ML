import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def bar_plot(series, text, rotation, title, path_out):
    figure(figsize=(10, 6))
    plt.bar(text, series)
    plt.title(f'{title}')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(f"{path_out}\\{title}.png")
    plt.close()


def bar_plot_horizontal(series, text, rotation, title, path_out):
    figure(figsize=(10, 6))
    plt.barh(text, series)
    plt.title(f'{title}')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(f"{path_out}\\{title}.png")
    plt.close()


def open_folder(folder):
    '''open html folder'''
    if not os.path.exists(folder):
        os.makedirs(folder)


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