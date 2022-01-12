from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


def create_coord_matrix(w, h):
    coord_temp = np.arange(w * h)
    coord_temp = coord_temp.reshape((h, w))
    coord_temp_rows = coord_temp // h
    coord_temp_cols = (coord_temp // w).T

    coord = np.zeros((w * h, 1, 2))
    coord[:, :, 0] = coord_temp_rows.reshape(w * h, 1)
    coord[:, :, 1] = coord_temp_cols.reshape(w * h, 1)

    return coord


def flatten_img(N, img, channel):
    pixels = img.flatten().reshape((N, 1, channel))
    return pixels.astype('uint8')


def create_weight_matrix(N, coord_matrix, img_pixels, rad=2, sigma_i=5, sigma_x=10):
    X = coord_matrix.repeat(N, axis=1)
    X_T = coord_matrix.reshape((1, N, 2)).repeat(N, axis=0)
    diff_X = X - X_T
    diff_X = diff_X[:, :, 0] ** 2 + diff_X[:, :, 1] ** 2

    F = img_pixels.repeat(N, axis=1)
    F_T = img_pixels.reshape((1, N, 3)).repeat(N, axis=0)
    diff_F = F - F_T
    diff_F = diff_F[:, :, 0] ** 2 + diff_F[:, :, 1] ** 2 + diff_F[:, :, 2] ** 2

    W_map = diff_X < rad ** 2

    W = np.exp(-((diff_F / (sigma_i ** 2)) + (diff_X / (sigma_x ** 2))))

    return W * W_map


def create_dist_matrix(W):
    d_i = np.sum(W, axis=1)
    return np.diag(d_i)


def solve_eig_eq(D, W):
    s_D = sparse.csr_matrix(D)
    s_W = sparse.csr_matrix(W)
    s_D_nhalf = np.sqrt(s_D).power(-1)
    L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
    lam, y = eigs(L)
    index = np.argsort(lam)

    return y[:, index[1]].real


def dilation(img):
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


def convert_eigen_vector_to_mask(w, h, eig_vec):
    eig_vec = eig_vec.reshape((w, h)).astype('float64')
    eigenvector = ((eig_vec / eig_vec.max()) * 255).astype('uint8')

    return dilation(eigenvector)


def n_cut(img):
    h, w, channel = img.shape

    coord_matrix = create_coord_matrix(w, h)
    img_pixels = flatten_img(h * w, img, channel)

    W = create_weight_matrix(h * w, coord_matrix, img_pixels, rad=8, sigma_i=6, sigma_x=13)
    D = create_dist_matrix(W)

    eig_vec = solve_eig_eq(D, W)

    mask = convert_eigen_vector_to_mask(w, h, eig_vec)
    img[mask <= 0] = 0

    return img


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    img = n_cut(img)

    cv2.imshow('img', img)
    cv2.waitKey()
