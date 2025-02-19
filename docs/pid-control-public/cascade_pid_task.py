import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
viewer.cam.distance = 4.
viewer.cam.lookat = np.array([0, 0, 1])
viewer.cam.elevation = -30.

from drone_simulator import DroneSimulator
from pid import PID

if __name__ == '__main__':
    desired_altitude = 2

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, viewer, desired_altitude = desired_altitude,
        altitude_sensor_freq = 0.01, wind_change_prob = 0.1, rendering_freq = 1
        )

    # TODO: Create necessary PID controllers using PID class
    pid_altitude = PID(
        gain_prop = 1e-8, gain_int = 0, gain_der = 4.2e-1,
        sensor_period = drone_simulator.altitude_sensor_period
        )

    acceleration_sensor_period = 1e-3
    pid_acceleration = PID(
        gain_prop = 0, gain_int = 1e+2, gain_der = 0,
        sensor_period = acceleration_sensor_period
    )

    desired_acceleration_residual = 0

    # Increase the number of iterations for a longer simulation
    for i in range(4000):
        # TODO: Use the PID controllers in a cascade designe to control the drone
        base_acceleration = 9.8099933990
        current_acceleration = data.sensor("body_linacc").data[2]

        if i % 100 == 0:
            desired_acceleration_residual = pid_altitude.output_signal(
                commanded_variable=desired_altitude,
                sensor_readings=drone_simulator.measured_altitudes
            )

        current_acceleration_residual = current_acceleration - base_acceleration

        #antigravity_thrust = 3.249567
        base_thrust = 2.3

        thrust_residual = pid_acceleration.output_signal(
            commanded_variable=desired_acceleration_residual,
            sensor_readings=[current_acceleration_residual]
        )

        desired_thrust = base_thrust + thrust_residual

        drone_simulator.sim_step(desired_thrust)