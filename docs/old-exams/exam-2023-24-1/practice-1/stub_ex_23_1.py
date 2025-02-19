import mujoco
import time
import random
import numpy as np
import mujoco
from mujoco import viewer
from PIL import Image
import cv2

### TODO: Add your code here ###

model = mujoco.MjModel.from_xml_path("car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def display_image():
    img = get_image()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.show()

def get_image():
    renderer.update_scene(data, camera="camera1")
    img = renderer.render()
    return img


def check_ball(seed=1337) -> bool:
    random.seed(seed)
    steps = random.randint(0, 500)
    data.actuator("turn 1").ctrl = 1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("turn 1").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # TODO: Add your code here
    best = 0
    for i in range(500):
        viewer.sync()
        data.actuator("turn 1").ctrl = 1
        mujoco.mj_step(model, data)
        #if i % 100 == 0:
            #display_image()
        img = get_image()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0, 100, 0])
        upper_red = np.array([0, 255, 255])
        color_mask = cv2.inRange(hsv_img, lower_red, upper_red)
        best = max(best, np.sum(color_mask))
    print(best)
    if best > 630000:
        return False
    return True


def drive_to_ball_1(seed=1337):
    random.seed(seed)
    steps = random.randint(0, 2500)
    data.actuator("forward 2").ctrl = -1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("forward 2").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # TODO Add your code here


def drive_to_ball_2(seed=1337):
    random.seed(seed)
    steps = random.randint(0, 2500)

    data.actuator("turn 2").ctrl = 1
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.actuator("turn 2").ctrl = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)

    # TODO Add your code here
    data.actuator("turn 1").ctrl = -1
    for i in range(75):
        mujoco.mj_step(model, data)
        viewer.sync()
    data.actuator("turn 1").ctrl = 0
    for i in range(1000):
        mujoco.mj_step(model, data)
    viewer.sync()
    img = get_image()
    img = img[280:300, 220:300, :]
    cv2.imwrite("pattern.png", img)
    data.actuator("turn 2").ctrl = 1
    for i in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()
        # print(data.body("target-ball").xpos)

        cur_img = get_image()
        cur_img = cur_img[280:300, 220:300, :]
        print(i, np.sum(np.abs(cur_img - img)))
        if i > 200 and np.sum(np.abs(cur_img - img)) < 150000:
            data.actuator("turn 2").ctrl = 0
            break
    data.actuator("turn 2").ctrl = -0.1
    for i in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()

        cur_img = get_image()
        cur_img = cur_img[280:300, 220:300, :]
        print(i, np.sum(np.abs(cur_img - img)))
        if i > 200 and np.sum(np.abs(cur_img - img)) < 130000:
            data.actuator("turn 2").ctrl = 0
            break
    data.actuator("forward 2").ctrl = 1
    for i in range(900):
        mujoco.mj_step(model, data)
        viewer.sync()
    data.actuator("forward 2").ctrl = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

print(drive_to_ball_2())