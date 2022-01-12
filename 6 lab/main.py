from math import sqrt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np


def ransac_iteration(edge_dots: list, threshold):
    inlier_dots = [[] for _ in range(0, 200)]

    max_tolerance = 0
    argmax_index = 0
    for i in range(1, 200):
        if not len(edge_dots):
            break
        randoms = np.random.choice(len(edge_dots), 2)
        dot0 = edge_dots[randoms[0]]
        dot1 = edge_dots[randoms[1]]

        a = dot0[1] - dot1[1]
        b = dot1[0] - dot0[0]
        c = dot0[0] * dot1[1] - dot1[0] * dot0[1]
        norm = sqrt(a ** 2 + b ** 2)

        if not norm:
            norm = 1

        tolerance_count = 0

        for dot in edge_dots:
            if (abs(a * dot[0] + b * dot[1] + c) / norm) <= threshold:
                tolerance_count += 1
                inlier_dots[i] += [dot]

        if max_tolerance < tolerance_count:
            max_tolerance = tolerance_count
            argmax_index = i

    if max_tolerance < 70:
        return []

    for dot in inlier_dots[argmax_index]:
        edge_dots.remove(dot)

    return [inlier_dots[argmax_index]]


def ransac(img, threshold, count):
    img_edge = cv2.Canny(img, 150, 250, apertureSize=3)
    edge_dots = cv2.findNonZero(img_edge)[:, 0].tolist()

    inliers = []

    for i in range(0, count):
        inliers += ransac_iteration(edge_dots, threshold)

    return inliers


def run_ransac(img):
    inliers = ransac(img, 3, 30)

    for l in inliers:
        if len(l) > 0:
            cv2.line(img, min(l), max(l), (0, 255, 0))

    cv2.imshow('ransac', img)


def hough_lines(img_edge, angle_step=1):
    edges_points = cv2.findNonZero(img_edge)[:, 0]

    thetas = np.deg2rad(np.arange(0, 180, step=angle_step))
    diag = int(np.sqrt(img_edge.shape[0] ** 2 + img_edge.shape[1] ** 2))
    rhos = np.linspace(-diag, diag, diag * 2)
    num_thetas = len(thetas)
    num_rhos = len(rhos)

    hough_dim = np.zeros((num_rhos, num_thetas))
    point_list = [[[] for _ in range(num_thetas)] for _ in range(num_rhos)]

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for (x, y) in edges_points:
        for theta_idx in range(num_thetas):
            rho = x * cos_t[theta_idx] + y * sin_t[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            hough_dim[rho_idx][theta_idx] += 1
            point_list[rho_idx][theta_idx].append((x, y))

    return hough_dim, point_list


def run_hough(img):
    img_edge = cv2.Canny(img, 150, 250, apertureSize=3)

    hough_dim, point_list = hough_lines(img_edge)
    indexes = np.argwhere(hough_dim > 90)

    for (rho_idx, theta_idx) in indexes:
        points = point_list[rho_idx][theta_idx]
        cv2.line(img, min(points), max(points), (0, 255, 0))

    cv2.imshow('hough', img)


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    run_ransac(img.copy())
    run_hough(img.copy())

    cv2.waitKey(0)
