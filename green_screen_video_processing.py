import cv2
import numpy as np

# Global variables
window_name="Green Screen"
tolerance = 40
softness = 5
mean_color = None
# Default background color (black)
bg_color = [0, 0, 0]
frame_counter = 0

# Callback functions for sliders
def update_tolerance(val):
    """
    Updates the global tolerance value.

    This function allows updating the global variable `tolerance` with the given
    value. It modifies the value without returning any result. This is typically
    used in scenarios where global tolerance is required to be adjusted dynamically.

    :param val: A new value to update the global tolerance.
    :type val: Any
    """
    global tolerance
    tolerance = val

def update_softness(val):
    """
    Updates the global variable `softness` with the given value.

    This function modifies a global variable named `softness` using the provided
    value. It is intended for cases where the value of `softness` must be updated
    globally throughout the application.

    :param val: The new value to set for the `softness` global variable.
    :type val: Any
    """
    global softness
    softness = val

def update_bg_red(val):
    """
    Updates the red component of the global background color.

    This function modifies the red component of the global `bg_color`
    list using the provided `val`. The value provided should fall
    within an acceptable range, depending on its intended usage.

    :param val: New red component value for the background color.
    :type val: int
    """
    global bg_color
    bg_color[2] = val

def update_bg_green(val):
    """
    Update the green component of the background color.

    This function modifies a global variable `bg_color` by updating its green color
    component with the provided value `val`.

    :param val: The new value for the green component of `bg_color`. Should be within
        the valid range for color representation.
    """
    global bg_color
    bg_color[1] = val

def update_bg_blue(val):
    """
    Updates the blue component of the background color.

    This function modifies the global background color by updating its blue
    component with the specified value. The updated value directly affects
    the 0th index of the `bg_color` list, which represents the blue component.

    :param val: The new value for the blue component of the background color.
    :type val: Any value compatible with the container's data type.
    """
    global bg_color
    bg_color[0] = val

def select_patch(event, x, y, flags, param):
    """
    Handles a mouse event to select a patch of the image at the specified coordinate
    and calculates its mean color. The function ensures that the patch is selected
    within the bounds of the image, and calculates the mean color only if the patch
    is valid (non-empty).

    :param event: The type of mouse event triggered (e.g., left button click).
    :type event: int
    :param x: The x-coordinate of the mouse position.
    :type x: int
    :param y: The y-coordinate of the mouse position.
    :type y: int
    :param flags: Any relevant flags passed during the mouse event.
    :type flags: int
    :param param: Additional parameters.
    :type param: Any
    :return: None
    """
    global mean_color
    if event == cv2.EVENT_LBUTTONDOWN:
        height, width, _ = frame.shape
        # Ensure patch coordinates are within bounds
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(width, x + 10), min(height, y + 10)
        patch = frame[y1:y2, x1:x2]
        if patch.size > 0:  # Avoid empty patches
            mean_color = np.mean(patch, axis=(0, 1))  # Get mean color of the patch

# Initialize video
video = cv2.VideoCapture('greenscreen-asteroid.mp4')  # Input video

# create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

out = cv2.VideoWriter('output_video.mp4', fourcc, fps, frame_size)

# Creating trackbar
cv2.namedWindow(window_name)
cv2.createTrackbar('Tolerance', window_name, tolerance, 100, update_tolerance)
cv2.createTrackbar('Softness', window_name, softness, 20, update_softness)
cv2.createTrackbar('BG Red', window_name, bg_color[2], 255, update_bg_red)
cv2.createTrackbar('BG Green', window_name, bg_color[1], 255, update_bg_green)
cv2.createTrackbar('BG Blue', window_name, bg_color[0], 255, update_bg_blue)
cv2.setMouseCallback(window_name, select_patch)

while True:
    ret, frame = video.read()

    # Restarting the video capture
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_counter += 1

    # Animate background color dynamically
    bg_color[0] = (frame_counter % 256)  # Blue changes over time
    bg_color[1] = (frame_counter * 2 % 256)  # Green changes faster
    bg_color[2] = (frame_counter * 3 % 256)  # Red changes fastest

    if mean_color is not None:
        # Convert frame to HSV for better color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # Define the green screen range
        # Handle HSV bounds safely
        lower_bound_green_color = np.array([max(0, mean_hsv[0] - tolerance), 50, 50])
        upper_bound_green_color = np.array([min(179, mean_hsv[0] + tolerance), 255, 255])

        # Create mask and apply softness
        mask = cv2.inRange(hsv_frame, lower_bound_green_color, upper_bound_green_color)
        softness = max(1, min(softness, 20))  # Clamp softness to safe range
        mask = cv2.GaussianBlur(mask, (softness * 2 + 1, softness * 2 + 1), 0)

        # Invert mask for foreground
        mask_inv = cv2.bitwise_not(mask)

        # Create a gradient-based background
        height, width, _ = frame.shape
        gradient = np.zeros_like(frame, dtype=np.uint8)

        for i in range(height):
            ratio = i / height
            gradient[i, :, 0] = int(bg_color[0] * ratio)  # Blue gradient
            gradient[i, :, 1] = int(bg_color[1] * (1 - ratio))  # Green gradient
            gradient[i, :, 2] = int(bg_color[2])  # Red gradient

        # Combine with foreground and new background
        bg_part = cv2.bitwise_and(gradient, gradient, mask=mask)
        fg_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

        result = cv2.add(bg_part, fg_part)

        out.write(result)
        cv2.imshow(window_name, result)
    else:
        # Show original frame until a patch is selected
        cv2.imshow(window_name, frame)

    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

video.release()
out.release()
cv2.destroyAllWindows()
