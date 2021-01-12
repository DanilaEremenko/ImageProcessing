import matplotlib.pyplot as plt


def draw_image(img_arr, title):
    plt.imshow(img_arr, cmap='gray')
    plt.title(title)
    plt.show()


def draw_hist(hist, title):
    plt.plot(hist)
    plt.title(title)
    plt.show()

