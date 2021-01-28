import matplotlib.pyplot as plt
import itertools


def draw_image(ax, img_arr, title):
    ax.imshow(img_arr, cmap='gray')
    ax.set_title(title)


def draw_hist(ax, hist, title, x_lim=None, y_lim=None):
    ax.plot(hist)
    ax.set_title(title)
    if x_lim is not None:
        ax.set_xlim([None, x_lim])
    if y_lim is not None:
        ax.set_ylim(None, y_lim)


def draw_images(imgs, hists, titles, show=True, save_path=None):
    assert len(imgs) == len(hists)
    fig, axes = plt.subplots(len(hists), 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    hist_y_lim = max(itertools.chain.from_iterable(hists))
    for i, (img, hist, title) in enumerate(zip(imgs, hists, titles)):
        draw_image(ax=axes[i, 0], img_arr=img, title=f"{title} image")
        draw_hist(ax=axes[i, 1], hist=hist, y_lim=hist_y_lim, title=f"{title} histogram")

    if type(save_path) == str:
        plt.savefig(save_path, dpi=300)
    if show:
        fig.show()
