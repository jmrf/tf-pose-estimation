import sys
from pprint import pformat

import cv2
import matplotlib.pyplot as plt
import requests


def post_image(img_file):
    """ post image and return the response """
    img = open(img_file, "rb").read()

    content_type = "image/jpeg"
    headers = {"content-type": content_type}
    response = requests.post(
        "http://localhost:5000/pose-estimation/angle", data=img, headers=headers
    )
    return response.json()


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def plot_image(image):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a.set_title("Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    try:
        img_file = sys.argv[1]
    except IndexError:
        img_file = "./images/apink1_crop_s1.jpg"

    print(pformat(post_image(img_file)))

    plot_image(read_imgfile(img_file))
