import matplotlib.pyplot as plt
import numpy as np
import cv2
import Preprocessing


def AbsoluteDifferenceImage(nose_left, image2):
    assert(nose_left.shape == image2.shape), "The images must be the same size"
    abs_diff_image = np.empty([nose_left.shape[0], nose_left.shape[1]])
    abs_diff_val = 0
    for i in range(nose_left.shape[0]):
        for j in range(image2.shape[1]):
            diff = abs(int(nose_left[i, j]) - int(image2[i, j]))
            abs_diff_image[i, j] = diff
            abs_diff_val += diff


    return abs_diff_val

def FindXCoordinateOfPatch(patch, strip):
    min_diff = np.inf
    best_x = 0
    height, width = patch.shape
    # print(f'Size of patch is: {patch.shape}')

    for x in range(strip.shape[1] - width):
        path_strip = strip[:, x:x + width]
        # print(f'Interval: [{x}, {x + width -1}]; strip patch size: {path_strip.shape}')

        diff = AbsoluteDifferenceImage(patch, path_strip)

        if diff < min_diff:
            min_diff = diff
            best_x = x
            # print(f'Min diff: {min_diff}; {best_x}')

    return best_x

def ComputeDisparityMapBetweenTwoImages(image_left, image_right):
    assert(image_left.shape == image_right.shape), "THe iamges must have the same size"
    gray_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    print(gray_left.shape)

    block_size = 7
    row_size, col_size = gray_right.shape
    disparity_matrix = np.ndarray(shape=(row_size - block_size + 1, col_size - block_size + 1), dtype=np.float32)
    disparity_matrix[:, :] = 0
    offset = block_size // 2

    max_disparity = 200

    for i in range(offset, row_size - offset):

        if i % 100:
            print(f'Row pixels computed: {i}')

        for j in range(offset, col_size - offset):
            # print(f'Interval width:  [{i - offset}, {i + offset + 1}], height: [{j - offset}, {j + offset + 1}]')
            # print(f'Interval height')
            width_start = j - offset
            if width_start - max_disparity > 0:
                width_start = j - offset - max_disparity
            else:
                width_start = 0
            patch_left = gray_left[i - offset:i + offset + 1, j - offset:j + offset + 1]
            strip_right = gray_right[i - offset:i + offset + 1, width_start:j+offset + 1]

            # plt.imshow(strip_right)
            # plt.show()

            disparity = FindXCoordinateOfPatch(patch_left, strip_right)

            disparity_matrix[i - offset, j - offset] = disparity

    return disparity_matrix

nose_left = cv2.imread('./nose_left.png')
nose_right = cv2.imread('./nose_right.png')

strip = cv2.imread('./nose_span.png')
strip_gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)

# Show strip
plt.imshow(strip)
plt.show()

nose_left_gray = cv2.cvtColor(nose_left, cv2.COLOR_RGB2GRAY)
nose_right_gray = cv2.cvtColor(nose_right, cv2.COLOR_RGB2GRAY)

## Absolute difference of the two images
abs_diff_image = AbsoluteDifferenceImage(nose_left_gray, nose_right_gray)
print(f'Diff noses left right: {abs_diff_image}')

## Find the most resembling nose
noses = ['./nose1.png', './nose2.png', './nose3.png']
vals = []
for nose in noses:
    image = cv2.imread(nose)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    result = AbsoluteDifferenceImage(image_gray, nose_left_gray)
    vals.append(result)

## WHich of the noses images mattch best
print(f'Best match is: {noses[vals.index(min(vals))]}')

## Going through strip
x_position_nose = FindXCoordinateOfPatch(nose_left_gray, strip_gray)
print(x_position_nose)

image_1 = cv2.imread('./tsukuba_left.png')
image_2 = cv2.imread('./tsukuba_right.png')

disparity_matrix = ComputeDisparityMapBetweenTwoImages(image_1, image_2)
plt.imshow(disparity_matrix, cmap='gray')
plt.show()

min_disp = 7
num_disp = 3 * 16
block_size = 15
stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setMinDisparity(min_disp)
stereo.setDisp12MaxDiff(100)
stereo.setUniquenessRatio(1)
stereo.setSpeckleRange(3)
stereo.setSpeckleWindowSize(3)

gray_left = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
gray_right = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

# plt.imshow(disp, cmap='gray')
# plt.show()

# plt.figure(figsize=(12,12))
# plt.imshow(disp)