"""
Stub for homework 2
"""

import time
import random
import numpy as np
import mujoco
from mujoco import viewer


import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 2


world_xml_path = f"car_{TASK_ID}.xml"
model = mujoco.MjModel.from_xml_path(world_xml_path)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(
    n_steps: int, /, view=False, rendering_speed = 10, **controls: float
) -> NDArray[np.uint8]:
    """A wrapper around `mujoco.mj_step` to advance the simulation held in
    the `data` and return a photo from the dash camera installed in the car.

    Args:
        n_steps: The number of simulation steps to take.
        view: Whether to render the simulation.
        rendering_speed: The speed of rendering. Higher values speed up the rendering.
        controls: A mapping of control names to their values.
        Note that the control names depend on the XML file.

    Returns:
        A photo from the dash camera at the end of the simulation steps.

    Examples:
        # Advance the simulation by 100 steps.
        sim_step(100)

        # Move the car forward by 0.1 units and advance the simulation by 100 steps.
        sim_step(100, **{"forward": 0.1})

        # Rotate the dash cam by 0.5 radians and advance the simulation by 100 steps.
        sim_step(100, **{"dash cam rotate": 0.5})
    """

    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img



# TODO: add addditional functions/classes for task 1 if needed
def navigate_to_red_ball(img):
    while True:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv_img, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            img_center = img.shape[1] / 2
            if radius < 4:
                print("Ball not found. Rotating to search...")
                sim_step(1, view=True, **{"forward": 0, "turn": 1})
            elif x < img_center - img_center / 6:
                print("Ball found. Rotating to adjust the direction...")
                sim_step(1, view=True, **{"forward": 0, "turn": 1})
            elif x > img_center + img_center / 6:
                print("Ball found. Rotating to adjust the direction...")
                sim_step(1, view=True, **{"forward": 0, "turn": -1})
            elif radius > 30:
                print("Ball is close!")
                break
            else:
                controls = {"forward": 1, "turn": 0}
                sim_step(50, view=True, **controls)

            print(f"Ball position: ({x:.2f}, {y:.2f}), Radius: {radius:.2f}, Turn: {0}, Forward: {1}")
        else:
            print("Ball not found. Rotating to search...")
            sim_step(1, view=True, **{"forward": 0, "turn": 1})

        car = data.body("car").xpos
        ball = data.body("target-ball").xpos
        distance = (car[0] - ball[0])**2 + (car[1] - ball[1])**2
        print(f"distance to the ball: ${distance}")
        img = sim_step(1, view=False)
# /TODO


def task_1():
    steps = random.randint(0, 2000)
    controls = {"forward": 0, "turn": 0.1}
    img = sim_step(steps, view=False, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    navigate_to_red_ball(img)

    # /TODO



# TODO: add addditional functions/classes for task 2 if needed
from PIL import Image, ImageDraw

def plot_contours(img, contours):
    """
    Draw contours on the image and display it using PIL.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_image)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        points = [(point[0][0], point[0][1]) for point in approx]

        draw.line(points + [points[0]], fill="red", width=3)

    pil_image.show()

def green_mask(hsv_img):
    lower_green = np.array([50, 40, 40])
    upper_green = np.array([60, 255, 255])
    color_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    return color_mask

def blue_mask(hsv_img):
    lower_blue = np.array([120, 100, 40])
    upper_blue = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    return color_mask

def gray_mask(img):
    lower_gray = np.array([112, 0, 0])
    upper_gray = np.array([140, 255, 255])
    gray_mask = cv2.inRange(img, lower_gray, upper_gray)
    return gray_mask

def crop_upper_two_thirds(img):
    height, width = img.shape[:2]
    cropped_img = img[:height * 2 // 3, :]
    return cropped_img

def cut_right_half(img):
    height, width, _ = img.shape
    return img[:, :width // 2]

def detect_gray_vertical_stripe(img, upper_treshold=300, lower_treshold=80):
    """
    Detects if a thin vertical gray stripe is visible on the screen.
    """
    gray_mask_img = gray_mask(img)
    gray_mask_img = crop_upper_two_thirds(gray_mask_img)

    contours, _ = cv2.findContours(gray_mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Detected vertical gray stripe at x={x}, y={y}, width={w}, height={h}")
        if h > lower_treshold and w > 5 and h < upper_treshold:
            print(f"Detected SPECIAL vertical gray stripe at x={x}, y={y}, width={w}, height={h}")
            return True

    return False

def calculate_upper_edge_angle(contour):
    """
    Calculates the angle of the upper edge of the trapezoidal wall contour.
    """
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    top_points = sorted(approx[:, 0, :], key=lambda p: p[1])[:2]

    if len(top_points) < 2:
        raise ValueError("Not enough points to calculate the upper edge.")

    (x1, y1), (x2, y2) = top_points
    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = np.arctan2(delta_y, delta_x)
    angle_degrees = np.degrees(angle_radians)

    if angle_degrees > 90:
        angle_degrees -= 180
    elif angle_degrees < -90:
        angle_degrees += 180

    return angle_degrees

def calculate_lower_edge_angle(contour, img_height):
    """
    Calculates the angle of the upper edge of the trapezoidal wall contour.
    """
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    sorted_points = sorted(approx[:, 0, :], key=lambda p: p[1], reverse=True)
    bottom_points = sorted_points[:2]

    if len(bottom_points) < 2:
        raise ValueError("Not enough points to calculate the lower edge.")
    
    cnt = 0
    for _, y in bottom_points:
        if y >= img_height - 20:
            cnt += 1
            if cnt >= 2:
                bottom_points = sorted_points[1:3]
    (x1, y1), (x2, y2) = bottom_points
    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = np.arctan2(delta_y, delta_x)
    angle_degrees = np.degrees(angle_radians)

    if angle_degrees > 90:
        angle_degrees -= 180
    elif angle_degrees < -90:
        angle_degrees += 180

    return angle_degrees

def detect_horizontal_stripe(img, color_name, prev_angle1=None, prev_angle2=None):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color_mask = []
    if color_name == "green":
        color_mask = green_mask(hsv_img)
    elif color_name == "both":
        color_mask = cv2.bitwise_or(green_mask(hsv_img), blue_mask(hsv_img))
    else:
        color_mask = blue_mask(hsv_img)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width = img.shape[:2]

    if not contours:
        print("No contours found.")
        return False, None, None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    try:
        angle1 = calculate_upper_edge_angle(contour)
        angle2 = calculate_lower_edge_angle(contour, img_height)

        if prev_angle2 != None and h > 40 and (abs(angle1) < 1 and abs(angle2) < 1 or abs(angle1) < 3 and abs(angle2) < 10 and prev_angle2 * angle2 < 0 and prev_angle2 < 6):
            print(f"SPECIAL Horizontal {color_name} stripe detected at y={y} x={x}, width={w}, height={h}, angle1={angle1:.4f}, angle2={angle2:.4f}, prev_angle2={prev_angle2}")
            return True, angle1, angle2

        if h > 40 and abs(angle1) < 15 and abs(angle2) < 15:
            print(f"Horizontal {color_name} stripe detected at y={y} x={x}, width={w}, height={h}, angle1={angle1:.4f}, angle2={angle2:.4f}, prev_angle2={prev_angle2}")
            return False, angle1, angle2

    except ValueError as e:
        print(f"Skipping contour due to error: {e}")
        
    return False,prev_angle1,prev_angle2

def rotate_to_face_wall(img, color, direction=0.3):
    """Rotates the car until it faces a horizontal wall."""
    print(f"Starting rotation to detect the {color} wall...")

    prev_angle1, prev_angle2 = None, None

    while True:
        detected, angle1, angle2 = detect_horizontal_stripe(img, color, prev_angle1, prev_angle2)

        if detected:
            print(f"Rotation complete, car is now facing the {color} wall.")
            break

        prev_angle1, prev_angle2 = angle1, angle2

        img = sim_step(1, view=True, **{"forward": -0.3, "turn": direction})
    if color == "blue":
        img = sim_step(50, view=True, **{"forward": -0.3, "turn": -direction})

def move_backwards(dist):
    return sim_step(dist, view=True, **{"forward": -1, "turn": 0})

def move_until(img, color_name, stop_condition, movement, halving=False):
    while True:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if color_name == "green":
            color_mask = green_mask(hsv_img) 
        elif color_name == "blue":
            color_mask = blue_mask(hsv_img)
        else:
            color_mask = cv2.bitwise_or(green_mask(hsv_img), blue_mask(hsv_img)) 
        contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No {color_name} wall detected. Stopping movement.")
            break

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        print(f"Detected {color_name} wall at x={x}, y={y}, width={w}, height={h}")

        if stop_condition(h):
            print(f"{color_name} wall is sufficiently large. Stopping movement.")
            break

        img = sim_step(10, view=True, forward=movement, turn=0)
        if (halving):
            img = cut_right_half(img)

    print("Finished moving backwards.")
    return img

def move_backwards_abit(img, color_name, stop_condition, movement):
    #while True:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color_mask = green_mask(hsv_img) if color_name == "green" else blue_mask(hsv_img)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    print(f"Detected {color_name} wall at x={x}, y={y}, width={w}, height={h}")

    #if stop_condition(h):
    #    print(f"{color_name} wall is sufficiently large. Stopping movement.")
    #    break

    if movement > 0:
        img = sim_step(70, view=True, forward=movement, turn=0)
    else:
        img = sim_step(20, view=True, forward=movement, turn=0)

    print("Finished moving backwards.")
    return img

def move_backwards_until_clear(img, color_name, width_threshold=150, height_threshold=45, halving=False):
    print(f"Starting to move backwards until the {color_name} wall is sufficiently small...")
    return move_until(img,color_name,lambda h: h<height_threshold, movement = -1, halving=halving)

def move_upfront_until_clear(img, color_name, width_threshold=150, height_threshold=45):
    print(f"Starting to move upfront until the {color_name} wall is sufficiently large...")
    return move_until(img,color_name,lambda h: h>height_threshold, movement = 1)

def turn_ninety_degrees_right():
    img = sim_step(115, view=True, forward=0, turn=-1)

def rotate_to_find_grey_pillar(img):
    while not detect_gray_vertical_stripe(img):
        img = sim_step(1, view=True, **{"forward": -0.4, "turn": 0.5})
    print(f"Rotation complete, car is now facing the gray pillar")

def move_until_grey_pillar(img):
    while not detect_gray_vertical_stripe(img, upper_treshold=63, lower_treshold=20):
        img = sim_step(10, view=True, **{"forward": -1, "turn": 0})
    print(f"Rotation complete, car is now facing the gray pillar")

# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=False, **controls)
    # TODO: Change the lines below.
    # For car control, you can use only sim_step function
    
    # If we are already in the wall at the start, we need to make some space first
    img = move_backwards_abit(img, "blue",lambda h: h<230, movement = -1)
    img = move_backwards_abit(img, "green",lambda h: h<230, movement = -1)
    img = move_backwards_abit(img, "both", lambda h: h<120, movement = 1)
    rotate_to_find_grey_pillar(img)
    rotate_to_face_wall(img,"blue", direction=-0.3)
    img = move_backwards_until_clear(img, "blue", height_threshold=45, halving=False)
    # first turn
    rotate_to_face_wall(img,"green")
    img = move_backwards_until_clear(img, "green", height_threshold=38, halving=True)
    #second turn
    turn_ninety_degrees_right()
    #rotate_to_face_wall(img,"blue")
    img = sim_step(900, view=True, **{"forward": 1, "turn": 0})
    #ball
    navigate_to_red_ball(img)
    # /TODO



def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
    return np.linalg.norm(ball_pos - car_pos) < 0.2


def ball_grab() -> bool:
    """Checks if the ball is inside the gripper."""
    print(data.body("target-ball").xpos[2])
    return data.body("target-ball").xpos[2] > 0.1


def teleport_by(x: float, y: float) -> None:
    data.qpos[0] += x
    data.qpos[1] += y
    sim_step(10, **{"dash cam rotate": 0})


def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    fovy = 90
    f = h / (2 * np.tan(fovy * np.pi / 360))
    intrinsic_matrix = np.array([[-f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients


# TODO: add addditional functions/classes for task 3 if needed
# /TODO


def task_3():
    #start_x = random.uniform(-0.2, 0.2)
    #start_y = random.uniform(0, 0.2)
    #teleport_by(start_x, start_y)

    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function

    teleport_by(0.8, 1.7)
    img = sim_step(900, view=False, **{"lift":1})
    img = sim_step(180, view=True, **{"lift":1, "turn":1})
    #plot_contours(img,[])
    time.sleep(1)

    img = sim_step(random.randint(40, 60), view=True, **{"lift":1, "turn":1})

    while True:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv_img, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            img_center = img.shape[1] / 2
            x = x - radius / 2
            y = y - radius / 2
            eps= 4
            if radius < 4:
                print("Ball not found. Rotating to search...")
                sim_step(1, view=True, **{"forward": 0, "turn": 1})
            elif x < img_center - eps:
                print("Ball found. Rotating to adjust the direction...")
                sim_step(1, view=True, **{"forward": 0, "turn": 1})
            elif x > img_center + eps:
                print("Ball found. Rotating to adjust the direction...")
                sim_step(1, view=True, **{"forward": 0, "turn": -1})
            else:
                print("Success!")
                time.sleep(10)
                break

            print(f"Ball position: ({x:.2f}, {y:.2f}), Radius: {radius:.2f}, Turn: {0}, Forward: {1}")
        else:
            print("Ball not found. Rotating to search...")
            sim_step(1, view=True, **{"forward": 0, "turn": 1})

    # /TODO

    assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can move the car as well if you need to
    # /TODO


    
    assert ball_grab()


if __name__ == "__main__":
    print(f"Running TASK_ID {TASK_ID}")
    if TASK_ID == 1:
        task_1()
    elif TASK_ID == 2:
        task_2()
    elif TASK_ID == 3:
        task_3()
    else:
        raise ValueError("Unknown TASK_ID")
