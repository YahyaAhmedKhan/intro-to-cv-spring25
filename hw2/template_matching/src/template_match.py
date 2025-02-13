import cv2 as cv
import numpy as np
import os
# from google.colab.patches import cv2_imshow

# NOTE: these values are just placeholders, you may need to adjust them
_DEFAULT_THRESHOLD = 0.7
_DEFAULT_NMS_WINDOW = 9


def non_maximal_suppression_2d(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Performs non-maximal suppression on the given values matrix. That is, anywhere that
    values[i,j] is a *local maximum* within a (window x window) box, it is kept, and anywhere
    that it is not a local maximum it is 'suppressed' (set to zero).

    The original values matrix is not modified, but a new matrix is returned.
    """
    if (window % 2) == 0:
        raise ValueError("Window size must be odd.")
    h, w = values.shape
    pad = (window - 1)//2
    padded_values = np.pad(values, pad, mode='constant')
    
    # UNOPTIMIZED CODE
    # This was my own code, Claude suggested the optimization below
    # It works perfectly as well
    
    # ret = np.zeros(values.shape) 
    # for i in range(pad, pad+h):
    #   for j in range(pad, pad+w):
    #     val = padded_values[i, j]
    #     max_val = -1
    #     for w_i in range(window):
    #       for w_j in range(window):
    #         max_val = max(max_val, padded_values[i-pad+w_i][j-pad+w_j])
    #     ret[i-pad][j-pad] = val if val == max_val else 0
    # return ret

    # OPTIMIZED CODE
    
    # Create a view of the padded array with rolling windows 
    windows = np.lib.stride_tricks.sliding_window_view(
        padded_values, 
        (window, window)
    )
    
    # Find the maximum value in each window
    local_max = np.max(windows, axis=(2, 3))
    
    # Keep only the values that are equal to their local maximum
    return np.where(values == local_max, values, 0)


def apply_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Applies a threshold to the given values matrix. That is, anywhere that values[i,j] is greater
    than the threshold, it is kept with the same value, and anywhere that it is below the threshold
    it is set to zero.

    The original values matrix is not modified, but a new matrix is returned.
    """
    ret = np.zeros(values.shape)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ret[i, j] = values[i,j] if values[i, j] > threshold else 0

    return ret


def find_objects_by_template_matching(
    image: np.ndarray, template: np.ndarray, threshold: float, nms_window: int
) -> list[tuple[int, int]]:
    """Finds copies of the given template in the given image by template-matching. Returns a list of
    (x, y) coordinates of the top-left corner of each match. The main steps of this function are:

    1. Use cv.matchTemplate to get a score map. This map is a 2D array where score[i,j] gives a
       measure of how well the template matches the image at position (i,j). Depending on the choice
       of 'method' in cv.matchTemplate, the score can be positive or negative, and the best match
       can be either the maximum or minimum value in the score map.
    2. Normalize the score map so that the best match is 1 and the worst match is 0
    3. Apply the threshold to the score map to throw away any values below the threshold (i.e. set
       pixels to zero if their score is below the threshold). Use a call to apply_threshold for
       this.
    3. Use non-maximal suppression to keep only local maxima in a (nms_window x nms_window) window
       (i.e. set pixels to zero if they are not the maximum value among their neighbors). Use a call
       to non_maximal_suppression_2d for this.
    4. Use np.where() to find all remaining nonzero pixels --> these are our matches.
    """
    match_mat = cv.matchTemplate(image, template, method=cv.TM_CCOEFF_NORMED) 
    min_val, max_val, _, _ = cv.minMaxLoc(match_mat)

    if (max_val - min_val) != 0:
        norm_match_mat = (match_mat - min_val) / (max_val - min_val)
        
    thresh_mat = apply_threshold(norm_match_mat, threshold)
    nms_mat = non_maximal_suppression_2d(thresh_mat, nms_window)
    
    matches = np.where(nms_mat != 0)
    matches = list(zip(matches[1], matches[0]))

    return matches


def visualize_matches(scene: np.ndarray, obj_hw: tuple[int, int], xy: list[tuple[int, int]]):
    """Visualizes the matches found by find_objects_by_template_matching."""
    count = len(xy)
    h, w = obj_hw
    for x, y in xy:
        cv.rectangle(scene, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Add text in the bottom left corner by using x=10 and y=the height of the scene - 20 pixels
    cv.putText(
        scene,
        f"Found {count} matches",
        (10, scene.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv.imshow("Matches", scene)
    # cv2_imshow(scene)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Made a function to run the template matching instead
# of passing inputs as arguments to the python file
def run_template_matching(image_path: str, template_path: str, threshold: float = _DEFAULT_THRESHOLD, nms_window: int = _DEFAULT_NMS_WINDOW):
    """Runs template matching on the given image and template."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Image not found: {template_path}")

    if nms_window % 2 == 0:
        raise ValueError("The window size must be odd.")

    if nms_window < 1:
        raise ValueError("The window size must be greater than or equal to 1.")

    scene = cv.imread(image_path)
    template = cv.imread(template_path)

    xy = find_objects_by_template_matching(scene, template, threshold, nms_window)
    visualize_matches(scene, template.shape[:2], xy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the scene", required=True)
    parser.add_argument("--template", help="Path to the template", required=True)
    parser.add_argument(
        "--threshold", help="Threshold for matches", type=float, default=_DEFAULT_THRESHOLD
    )
    parser.add_argument(
        "--nms-window",
        help="Window size for non-maximal suppression",
        type=int,
        default=_DEFAULT_NMS_WINDOW,
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Image not found: {args.template}")

    if args.nms_window % 2 == 0:
        raise ValueError("The window size must be odd.")

    if args.nms_window < 1:
        raise ValueError("The window size must be greater than or equal to 1.")

    scene = cv.imread(args.image)
    object = cv.imread(args.template)
    xy = find_objects_by_template_matching(scene, object, args.threshold, args.nms_window)

    visualize_matches(scene, object.shape[:2], xy)