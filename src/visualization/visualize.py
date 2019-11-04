import matplotlib.pyplot as plt
import cv2


def show_image(image, title=None):
    """
    Show an image.

    Parameters
    ----------
    image
        The image to be shown
    title : str, optional
        Title of the image
    """
    b, g, r = cv2.split(image)
    image = cv2.merge((r, g, b))
    plt.imshow(image)
    plt.title(title)
    plt.show()
