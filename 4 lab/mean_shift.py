import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from PIL import Image

constant = math.sqrt(2 * math.pi)


def gaussian_kernel(distance, k_width):
    val = math.exp(-0.5 * (distance / k_width) ** 2) / constant
    return val


def euclid_distance(x, xi):
    return math.sqrt((x[0] - xi[0]) ** 2 + (x[1] - xi[1]) ** 2 + (x[2] - xi[2]) ** 2)


def neighbourhood_points(X, w, h, i_centroid, j_centroid, distance):
    neighbours = []

    i_beg = max(i_centroid - 30, 0)
    i_end = min(i_centroid + 30, w)
    j_beg = max(j_centroid - 30, 0)
    j_end = min(j_centroid + 30, h)

    for i in range(i_beg, i_end):
        for j in range(j_beg, j_end):
            if euclid_distance(X[i, j], X[i_centroid, j_centroid]) <= distance:
                neighbours.append(X[i, j])

    return neighbours


def mean_shift(img, n_iterations=1, k_width=10, n_dist=50):
    pixels = img.load()

    for it in range(n_iterations):
        for i in range(img.size[0]):
            print('it ', it, ' row ', i)
            for j in range(img.size[1]):
                neighbours = neighbourhood_points(pixels, img.size[0], img.size[1], i, j, n_dist)

                numerator = [0, 0, 0]
                denominator = 0
                for neighbour in neighbours:
                    distance = euclid_distance(neighbour, pixels[i, j])
                    weight = gaussian_kernel(distance, k_width)
                    numerator[0] += weight * neighbour[0]
                    numerator[1] += weight * neighbour[1]
                    numerator[2] += weight * neighbour[2]
                    denominator += weight

                numerator[0] = int(numerator[0] / denominator)
                numerator[1] = int(numerator[1] / denominator)
                numerator[2] = int(numerator[2] / denominator)

                pixels[i, j] = tuple(numerator)


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = Image.open(filename)
    mean_shift(img)

    img.show()
