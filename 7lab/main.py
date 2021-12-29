import os

import cv2
import numpy as np

INPUT = 'Dumptruck'
OUTPUT = 'saved'


def draw_arrow(image: np.ndarray, p: (int, int), q: (int, int), color: (np.uint8, np.uint8, np.uint8),
               arrow_magnitude: int = 9, thickness: int = 1, line_type: int = 8, shift: int = 0) -> None:
    cv2.line(image, p, q, color, thickness, line_type, shift)
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))
    cv2.line(image, p, q, color, thickness, line_type, shift)


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


def derX(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, 1, 0], [-1, 1, 0], [0, 0, 0]])
    dx = convolve(image, kernel)
    return dx


def derY(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, 0], [1, 1, 0], [0, 0, 0]])
    dy = convolve(image, kernel)
    return dy


def derT(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    dt = convolve(image1, kernel) + convolve(image2, -kernel)
    return dt


def harris(Ix, Iy, threshold, const, window_size=3):
    dots = []
    h, w = Ix.shape

    square_Ix = Ix ** 2
    square_Iy = Iy ** 2
    Ixy = Ix * Iy
    offset = window_size // 2

    for i in range(offset, h - offset):
        start_i = i - offset
        end_i = i + offset + 1
        for j in range(offset, w - offset):
            start_j = j - offset
            end_j = j + offset + 1
            ss_xx = np.sum(square_Ix[start_i:end_i, start_j:end_j])
            ss_yy = np.sum(square_Iy[start_i:end_i, start_j:end_j])
            ss_xy = np.sum(Ixy[start_i:end_i, start_j:end_j])
            det = ss_xx * ss_yy - ss_xy ** 2
            response = det - const * (ss_xx + ss_yy) ** 2
            if response >= threshold:
                dots.append((j, i))
    return dots


def lucas_kanade_step(first_image, second_image, harris_thr=0.1, harris_constant=0.06,
                                         velocity_threshold=0.7):
    first_dX = derX(first_image)
    first_dY = derY(first_image)
    dots = harris(first_dX, first_dY, threshold=harris_thr, const=harris_constant)
    dT = derT(first_image, second_image)
    offset = 1  # 3x3 window_size
    points_with_velocity = []
    for (j, i) in dots:
        # process harris for second image by dots from harris of first image
        start_i = i - offset
        end_i = i + offset + 1
        start_j = j - offset
        end_j = j + offset + 1
        Ix = first_dX[start_i:end_i, start_j:end_j].flatten()
        Iy = first_dX[start_i:end_i, start_j:end_j].flatten()
        It = dT[start_i:end_i, start_j:end_j].flatten()

        # calculate equation of lukas-kanade
        A = np.array([Ix, Iy])
        A_trans = np.array(A)  # transpose of A
        A = np.array(np.transpose(A))
        A_pinv = np.linalg.pinv(np.dot(A_trans, A))
        vel_x, vel_y = np.dot(np.dot(A_pinv, A_trans), It)

        if abs(vel_x) >= velocity_threshold and abs(vel_y) >= velocity_threshold:
            points_with_velocity.append(((j, i), (vel_x, vel_y)))
    return points_with_velocity


def normalize(img):
    return img / 255


if __name__ == '__main__':
    input_img = [os.path.join(INPUT, item) for item in os.listdir(INPUT)]

    for i in range(1, len(input_img)):
        print('Process image ' + str(i-1))
        current_img = cv2.imread(input_img[i - 1], cv2.IMREAD_GRAYSCALE)

        current_normalized = normalize(current_img.copy())
        next_normalized = normalize(cv2.imread(input_img[i], cv2.IMREAD_GRAYSCALE))

        vectors = lucas_kanade_step(current_normalized, next_normalized)

        current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)

        for ((j, k), (vel_x, vel_y)) in vectors:
            draw_arrow(current_img, (k, j), (int(k + 3 * vel_x), int(j + 3 * vel_y)), (0, 0, 255))

        cv2.imwrite(os.path.join(OUTPUT, str(i - 1) + '.jpg'), current_img)

