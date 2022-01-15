from functools import cmp_to_key
from math import sqrt, log, floor, cos, sin
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np


def sigma_diffs(sigma1, sigma2):
    return (sigma1 ** 2) - (sigma2 ** 2)


def blur_img(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def get_base_img(img, sigma, initial_blur):
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = sqrt(max(sigma_diffs(sigma, 2 * initial_blur), 0.01))

    return blur_img(img, sigma_diff)


def get_octaves_number(img):
    # One subtracted for floor effect
    return int(round(log(min(img.shape[0:2])) / log(2) - 1))


def get_gaussian_kernels(sigma, intervals):
    octave_size = intervals + 3
    k = 2 ** (1. / intervals)
    gaussian_kernels = np.zeros(octave_size)
    gaussian_kernels[0] = sigma

    for image_index in range(1, octave_size):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_diffs(sigma_total, sigma_previous))

    return gaussian_kernels


def generate_blurred_images(img, sigma, intervals, initial_blur):
    gaussian_images = []

    img = get_base_img(img, sigma, initial_blur)
    octaves_count = get_octaves_number(img)
    gaussian_kernels = get_gaussian_kernels(sigma, intervals)

    for octave_index in range(octaves_count):
        gaussian_images_in_octave = [img]
        # First is already blurred
        for gaussian_kernel in gaussian_kernels[1:]:
            img = blur_img(img, gaussian_kernel)
            gaussian_images_in_octave.append(img)

        gaussian_images.append(gaussian_images_in_octave)

        octave_base = gaussian_images_in_octave[-3]
        img = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                         interpolation=cv2.INTER_NEAREST)

    return np.array(gaussian_images, dtype=object)


def generate_dog_pyramid(gaussian_images):
    dog_pyramid = []

    for gaussian_images_in_octave in gaussian_images:
        dog_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_octave.append(np.subtract(second_image, first_image))

        dog_pyramid.append(dog_octave)

    return np.array(dog_pyramid, dtype=object)


def find_scale_space_extremas(gaussian_images, dog_images, intervals, sigma, border_width,
                              contrast_threshold=0.04):
    threshold = floor(0.5 * contrast_threshold / intervals * 255)
    keypoints = []

    for octave_index, octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(octave, octave[1:], octave[2:])):
            for i in range(border_width, first_image.shape[0] - border_width):
                for j in range(border_width, first_image.shape[1] - border_width):
                    first_layer = first_image[i - 1:i + 2, j - 1:j + 2]
                    second_layer = second_image[i - 1:i + 2, j - 1:j + 2]
                    third_layer = third_image[i - 1:i + 2, j - 1:j + 2]

                    if check_pixel_is_extremum(first_layer, second_layer, third_layer, threshold):
                        localization_result = localize_extremum(i, j, image_index + 1, octave_index,
                                                                intervals, octave, sigma,
                                                                contrast_threshold, border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_orientations = compute_points_with_orientations(keypoint, octave_index,
                                                                                      gaussian_images[octave_index][
                                                                                          localized_image_index])
                            for keypoint_with_orientation in keypoints_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints


def localize_extremum(i, j, layer_index, octave_index, num_intervals, dog_images_in_octave,
                      sigma, contrast_threshold, border_width, eigenvalue_ratio=10, localization_iterations=5):
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape

    attempt_index = 0
    pixel_cube = None
    gradient = None
    extremum_update = None
    hessian = None

    for attempt_index in range(localization_iterations):
        first_image, second_image, third_image = dog_images_in_octave[layer_index - 1:layer_index + 2]
        pixel_cube = np.stack([first_image[i - 1:i + 2, j - 1:j + 2],
                               second_image[i - 1:i + 2, j - 1:j + 2],
                               third_image[i - 1:i + 2, j - 1:j + 2]])
        pixel_cube = pixel_cube.astype('float32') / 255.0

        gradient = compute_grad_of_center_cube(pixel_cube)
        hessian = compute_hessian_of_center_cube(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break

        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        layer_index += int(round(extremum_update[2]))

        if i < border_width or i >= image_shape[0] - border_width or j < border_width or \
                j >= image_shape[1] - border_width or layer_index < 1 or layer_index > num_intervals:
            extremum_is_outside_image = True
            break

    if extremum_is_outside_image:
        return None
    if attempt_index >= localization_iterations - 1:
        return None

    extremum_response = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(extremum_response) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)

        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = cv2.KeyPoint()

            y = (j + extremum_update[0]) * (2 ** octave_index)
            x = (i + extremum_update[1]) * (2 ** octave_index)

            keypoint.pt = (y, x)
            keypoint.octave = octave_index + layer_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (
                    2 ** 16)
            # octave_index + 1 because the input image was doubled
            keypoint.size = sigma * (2 ** ((layer_index + extremum_update[2]) /
                                           np.float32(num_intervals))) * (2 ** (octave_index + 1))
            keypoint.response = abs(extremum_response)

            return keypoint, layer_index

    return None


def compute_grad_of_center_cube(pixel_array):
    # f'(x) is (f(x + h) - f(x - h)) / (2 * h) of order O(h^2)
    # h = 1,  f'(x) = (f(x + 1) - f(x - 1)) / 2

    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


def compute_hessian_of_center_cube(pixel_array):
    # Order O(h^2)
    # f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # h = 1, so f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # h = 1, so (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4

    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


def check_pixel_is_extremum(first_subimage, second_subimage, third_subimage, threshold):
    center_pixel_value = second_subimage[1, 1]

    if abs(center_pixel_value) > threshold:
        if center_pixel_value >= 0:
            return np.all(center_pixel_value >= first_subimage) and \
                   np.all(center_pixel_value >= third_subimage) and \
                   np.all(center_pixel_value >= second_subimage[0, :]) and \
                   np.all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                   np.all(center_pixel_value <= third_subimage) and \
                   np.all(center_pixel_value <= second_subimage[0, :]) and \
                   np.all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False


def compute_points_with_orientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,
                                     peak_ratio=0.8, scale_factor=1.5):
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i

        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j

                if 0 < region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]

                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # dafaq вычи долбанные
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]

        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) /
                                       (left_value - 2 * peak_value + right_value)) % num_bins

            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0

            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations


def compare_keypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def convert_keypoints_to_original_image_scales(keypoints):
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = (keypoint.pt[0] * 0.5, keypoint.pt[1] * 0.5)
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints


def remove_duplicates(keypoints):
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                last_unique_keypoint.size != next_keypoint.size or \
                last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints


def unpack_octave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale


def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3,
                         descriptor_max_value=0.2):
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(np.deg2rad(angle))
        sin_angle = sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2,
                                     num_bins))  # first two dimensions are increased by 2 to account for border effects

        # From OpenCV impl
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(
            round(hist_width * sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5

                if -1 < row_bin < window_width and -1 < col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))

                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        # trilinear interpolation
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list,
                                                                orientation_bin_list):

            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(
                int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor

            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)

        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')


def get_keypoints_by_dog(gaussian_images, dog_images, intervals, sigma, border_width):
    keypoints = find_scale_space_extremas(gaussian_images, dog_images, intervals, sigma, border_width)
    keypoints = remove_duplicates(keypoints)

    return convert_keypoints_to_original_image_scales(keypoints)


def sift(img, sigma=1.6, initial_blur=0.5, intervals=3, border_width=5):
    working_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY).astype('float32')

    gaussian_images = generate_blurred_images(working_img, sigma, intervals, initial_blur)
    dog_images = generate_dog_pyramid(gaussian_images)
    keypoints = get_keypoints_by_dog(gaussian_images, dog_images, intervals, sigma, border_width)
    descriptors = generate_descriptors(keypoints, gaussian_images)

    return keypoints, descriptors


def match_images(img1, img2, min_match_count=10):
    print('begin computing')
    kp1, des1 = sift(img1)
    print('first image is processed')
    kp2, des2 = sift(img2)
    print('second image is processed')

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int(abs(h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1] = img1
            newimg[:h2, w1:w1 + w2] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, tuple(np.random.random(size=3) * 255))

        cv2.imwrite('sift2.png', newimg)
        cv2.imshow('sift', newimg)
        cv2.waitKey(0)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), min_match_count))


if __name__ == '__main__':
    Tk().withdraw()
    filename1 = askopenfilename()
    filename2 = askopenfilename()

    img1 = cv2.imread(filename1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(filename2, cv2.IMREAD_COLOR)

    match_images(img1, img2)

    # keypoints, descriptors = sift(img)

    # for dot in keypoints:
    #     cv2.circle(img, (int(dot.pt[0]), int(dot.pt[1])), 2, (0, 0, 255), cv2.FILLED)
    # cv2.imshow('key_points', img)

    # cv2.imshow('harris', img)
    # cv2.imwrite('harris.png', img)
