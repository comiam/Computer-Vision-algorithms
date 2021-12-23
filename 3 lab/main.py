import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def draw_area(pos, rgb_img):
    stack = [pos]

    while len(stack) > 0:
        dot = stack.pop()
        dot_index = dot[1]

        while dot_index >= 0 and rgb_img[dot[0]][dot_index][0] == 0:
            rgb_img[dot[0]][dot_index][0] = 255

            if dot[0] - 1 >= 0 and rgb_img[dot[0] - 1][dot_index][0] == 0:
                stack.append((dot[0] - 1, dot_index))
            if dot[0] + 1 < rgb_img.shape[0] and rgb_img[dot[0] + 1][dot_index][0] == 0:
                stack.append((dot[0] + 1, dot_index))

            dot_index -= 1

        dot_index = dot[1] + 1

        while dot_index < rgb_img.shape[1] and rgb_img[dot[0]][dot_index][0] == 0:
            rgb_img[dot[0]][dot_index][0] = 255

            if dot[0] - 1 >= 0 and rgb_img[dot[0] - 1][dot_index][0] == 0:
                stack.append((dot[0] - 1, dot_index))
            if dot[0] + 1 < rgb_img.shape[0] and rgb_img[dot[0] + 1][dot_index][0] == 0:
                stack.append((dot[0] + 1, dot_index))

            dot_index += 1

    return rgb_img


def find_cells(rgb_img):
    cells = 0

    for i in range(0, rgb_img.shape[0]):
        for j in range(0, rgb_img.shape[1]):
            if rgb_img[i][j][0] == 0:
                rgb_img = draw_area((i, j), rgb_img)
                cells += 1

    return (cells, rgb_img)


def erosion(img):
    window_size = 5
    mat = np.full((window_size, window_size), 0)
    mat[4][4] = 0
    mat[0][4] = 0
    mat[4][0] = 0
    mat[0][0] = 0

    res = img.copy()

    for i in range(0, img.shape[0] - window_size + 1):
        for j in range(0, img.shape[1] - window_size + 1):
            sub = img[i:i + window_size, j:j + window_size]
            indices = np.where(mat == 0)
            center = int(window_size / 2)

            res[i + center][j + center] = 0 if np.all(sub[indices] == mat[indices]) else 255

    return res


def dilation(img):
    window_size = 5
    mat = np.full((window_size, window_size), 255)
    mat[4][4] = 0
    mat[0][4] = 0
    mat[4][0] = 0
    mat[0][0] = 0

    res = img.copy()

    for i in range(0, img.shape[0] - window_size + 1):
        for j in range(0, img.shape[1] - window_size + 1):
            sub = img[i:i + window_size, j:j + window_size]
            indices = np.where(mat == 255)
            center = int(window_size / 2)

            res[i + center][j + center] = 0 if np.any(sub[indices] != mat[indices]) else 255

    return res


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    (T, res) = cv2.threshold(img, 200, 255, cv2.THRESH_OTSU)

    res = dilation(res)
    res = erosion(res)
    res = erosion(res)
    res = erosion(res)
    res = dilation(res)
    res = dilation(res)
    resRgb = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    (cells, img) = find_cells(resRgb)

    print(cells)

    cv2.imshow('res1', img.astype(np.uint8))
    cv2.waitKey(0)
