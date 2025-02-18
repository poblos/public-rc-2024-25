import mujoco
import numpy as np
from mujoco import viewer
import time

model = mujoco.MjModel.from_xml_path("./robot.xml")
data = mujoco.MjData(model)

for i in range(1, 5):
    joint_name = f"box{i}-joint"
    joint_id = model.joint(joint_name).id
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr:qpos_adr+3] = np.random.uniform([-0.8, -0.8, 0.3], [0.8, 0.8, 1.0])
    data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

mujoco.mj_forward(model, data)

viewer = viewer.launch_passive(model, data)

desired_pos = np.array([0,0,0.5])
treshold = 0.6
step = 0
phase = 0
counter_x = 101
counter_y = 11
preparation = 1000
while True:
    step_start = time.time()
    step += 1

    # if step % 1000 == 0:
    #     data.actuator("ac_x").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_y").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_z").ctrl = np.random.uniform(-1.1, 1.1)
    #     data.actuator("ac_r").ctrl = np.random.uniform(-3, 3)
    #     print(data.joint("box1-joint").qpos)
    mujoco.mj_step(model, data)
    viewer.sync()
    all_in_place = True
    if step % 1000 == 0:
        for i in range(1,5):
            pos = data.joint(f"box{i}-joint").qpos[:3]
            if np.linalg.norm(pos - desired_pos) > treshold:
                print(f"box{i}-joint too far: {np.linalg.norm(pos - desired_pos)}")
                all_in_place = False
    
        if all_in_place:
            print("success")

    if step > 1000:
        if phase == 0:
            if preparation > 0:
                data.actuator("ac_z").ctrl = 1
                data.actuator("ac_r").ctrl = 1.57
                preparation -= 1
            elif preparation > - 1000:
                data.actuator("ac_x").ctrl = counter_x
                data.actuator("ac_y").ctrl = counter_y
                preparation -= 1
            elif preparation > - 2000:
                data.actuator("ac_z").ctrl = 0.01
                preparation -= 1
            else:
                if step % 100 == 0:
                    data.actuator("ac_x").ctrl = counter_x / 100
                    counter_x -=1
                if counter_x == 0:
                    counter_x = 101
                    data.actuator("ac_y").ctrl = counter_y / 10
                    counter_y -=1
                if counter_y == -10:
                    phase += 1
                    counter_x = 101
                    counter_y = 11
                    preparation = 1000
        if phase == 1:
            if preparation > 0:
                data.actuator("ac_z").ctrl = 1
                data.actuator("ac_r").ctrl = 1.57
                preparation -= 1
            elif preparation > - 1000:
                data.actuator("ac_x").ctrl = -counter_x
                data.actuator("ac_y").ctrl = counter_y
                preparation -= 1
            elif preparation > - 2000:
                data.actuator("ac_z").ctrl = 0.01
                preparation -= 1
            else:
                if step % 100 == 0:
                    data.actuator("ac_x").ctrl = -counter_x / 100
                    counter_x -=1
                if counter_x == 0:
                    counter_x = 101
                    data.actuator("ac_y").ctrl = counter_y / 10
                    counter_y -=1
                if counter_y == -10:
                    phase += 1
                    counter_x = 101
                    counter_y = 11
                    preparation = 1000
        if phase == 2:
            if preparation > 0:
                data.actuator("ac_z").ctrl = 1
                data.actuator("ac_r").ctrl = 1.57
                preparation -= 1
            elif preparation > - 1000:
                data.actuator("ac_y").ctrl = counter_x
                data.actuator("ac_x").ctrl = counter_y
                preparation -= 1
            elif preparation > - 2000:
                data.actuator("ac_z").ctrl = 0.01
                preparation -= 1
            else:
                data.actuator("ac_r").ctrl = 0
                if step % 100 == 0:
                    data.actuator("ac_y").ctrl = counter_x / 100
                    counter_x -=1
                if counter_x == 0:
                    counter_x = 101
                    data.actuator("ac_x").ctrl = counter_y / 10
                    counter_y -=1
                if counter_y == -10:
                    phase += 1
                    counter_x = 101
                    counter_y = 11
                    preparation = 1000
        if phase == 3:
            if preparation > 0:
                data.actuator("ac_z").ctrl = 1
                data.actuator("ac_r").ctrl = 1.57
                preparation -= 1
            elif preparation > - 1000:
                data.actuator("ac_y").ctrl = -counter_x
                data.actuator("ac_x").ctrl = counter_y
                preparation -= 1
            elif preparation > - 2000:
                data.actuator("ac_z").ctrl = 0.01
                preparation -= 1
            else:
                data.actuator("ac_r").ctrl = 0
                if step % 100 == 0:
                    data.actuator("ac_y").ctrl = -counter_x / 100
                    counter_x -=1
                if counter_x == 0:
                    counter_x = 101
                    data.actuator("ac_x").ctrl = counter_y / 10
                    counter_y -=1
                if counter_y == -10:
                    phase += 1
                    counter_x = 101
                    counter_y = 11
                    preparation = 1000

    time_until_next_step = model.opt.timestep / 1000 - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
