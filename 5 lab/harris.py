from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np


def convolve(image, kernel):
    h_kernel, w_kernel = kernel.shape
    h_image, w_image = image.shape
    h_offset = h_kernel // 2
    w_offset = w_kernel // 2

    padded_image = np.zeros((h_image + h_offset * 2, w_image + w_offset * 2))
    padd_h_image = padded_image.shape[0] - h_offset
    padd_w_image = padded_image.shape[1] - w_offset
    padded_image[h_offset:padd_h_image, w_offset:padd_w_image] = image
    output = np.zeros_like(padded_image)

    for i in range(h_offset, h_image):
        for j in range(w_offset, w_image):
            output[i, j] = np.sum(kernel * padded_image[i - h_offset: i + h_offset + 1, j - w_offset: j + w_offset + 1])

    return output[h_offset:padd_h_image, w_offset:padd_w_image]


# Sobel operator kernels.
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def grad_x(img):
    return convolve(img, kernel_x)


def grad_y(img):
    return convolve(img, kernel_y)


def harris(Ix, Iy, threshold=0.9, const=0.06, window_size=3):
    dots = []
    h, w = Ix.shape

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    offset = window_size // 2

    for i in range(offset, h - offset):
        start_i = i - offset
        end_i = i + offset + 1
        for j in range(offset, w - offset):
            start_j = j - offset
            end_j = j + offset + 1
            ss_xx = np.sum(Ixx[start_i:end_i, start_j:end_j])
            ss_yy = np.sum(Iyy[start_i:end_i, start_j:end_j])
            ss_xy = np.sum(Ixy[start_i:end_i, start_j:end_j])

            det = ss_xx * ss_yy - ss_xy ** 2
            trace = ss_xx + ss_yy
            response = det - const * trace ** 2

            if response >= threshold:
                dots.append((j, i))
    return dots


def draw_dots(img, dots):
    for dot in dots:
        cv2.circle(img, dot, 3, (0, 0, 255), cv2.FILLED)


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Ix = grad_x(gray_img)
    Iy = grad_y(gray_img)

    dots = harris(Ix, Iy, threshold=10)
    draw_dots(img, dots)

    cv2.imshow('harris', img)
    cv2.imwrite('harris.png', img)
    cv2.waitKey(0)
