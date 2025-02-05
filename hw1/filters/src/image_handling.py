import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    raise NotImplementedError("your code here")


def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    raise NotImplementedError("your code here")


def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    raise NotImplementedError("your code here")


def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    raise NotImplementedError("your code here")


def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    raise NotImplementedError("your code here")


def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    raise NotImplementedError("your code here")


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    raise NotImplementedError("your code here")


def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    raise NotImplementedError("your code here")


def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    raise NotImplementedError("your code here")


def greens(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the green
    channel
    """
    raise NotImplementedError("your code here")


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    raise NotImplementedError("your code here")


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    raise NotImplementedError("your code here")


def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    raise NotImplementedError("your code here")


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    raise NotImplementedError("your code here")


def main():
    # Write your testing code here and provide standard input/output calls to run the functions
    # If use gives 1, run function 1, if 2, run function 2, etc.
    # If user gives 0, exit the program, etc

    # TODO: Implement the main function
    raise NotImplementedError("your code here")


main()