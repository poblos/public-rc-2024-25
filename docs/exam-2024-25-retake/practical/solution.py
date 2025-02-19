import cv2
import mujoco
import numpy as np
import os
import math

# In case you want to use the Rotation class from scipy
from scipy.spatial.transform import Rotation as R

def get_global_camera_pose(model, data, camera_name):
    """
    Get the global pose of the camera in the world frame.

    As one can read in MuJoCo documentation,
    https://mujoco.readthedocs.io/en/stable/modeling.html#cameras,
    the cameras in MuJoCo look towards the negative Z axis of the camera frame,
    while positive X and Y correspond to right and up in the image plane, respectively.
    This is different from the standard camera frame in computer vision,
    where the camera looks towards the positive Z axis.
    Hence you might have to add an additional transformation to the camera pose
    before you can use OpenCV to process the images.
    """
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    rotation_matrix = data.cam_xmat[camera_id].reshape(3, 3)
    position = data.cam_xpos[camera_id]
    return position, rotation_matrix


class Detector:
    def __init__(self, model, data) -> None:
        self.im_height = 480  # Image height (pixels)
        self.im_width = 640   # Image width (pixels)

        # TODO: Determine camera parameters and construct camera matrix
        self.fovy = 90
        self.focal_length_y = 0.5 * self.im_height / math.tan(self.fovy * math.pi / 360)
        self.focal_length_x = self.focal_length_y  # Assuming square pixels

        self.principal_point_x = self.im_height / 2
        self.principal_point_y = self.im_width / 2

        self.camera_matrix = np.array([
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=float)

        self.counter = 0
        # END TODO

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        self.model = model
        self.data = data

    def detect(self, img, save_dir = 'imgs', file_name = None) -> None:
        # Load the dictionary that was used to generate the markers.
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # TODO: Detect the cube and find its position
        detected = False
        cube_center_world_frame = None

        self.counter += 1
        parameters = cv2.aruco.DetectorParameters()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids != None and len(ids) > 0:
            detected = True
            point = (corners[0][0][0] + corners[0][0][1] + corners[0][0][2] + corners[0][0][3])/4
            #we need to calculate this as z is set to 0.1 normally]
            y = (point[1] - self.principal_point_y * 0.1) / self.focal_length_y
            x = (point[0] - self.principal_point_x * 0.1) / self.focal_length_x
            z = 0.1
            print(f"image coords {point}")
            # 22 - 90 degrees
            r = R.from_euler('z', 90 * self.counter / 22, degrees=True)

            point_camera_coords = [x,y,z]
            print(f"camera coords {point_camera_coords}")
            point_world_coord = r.apply(point_camera_coords)
            print(f"world coords {point_world_coord}")
            cube_center_world_frame = point_world_coord

        # END TODO

        self.test_detection(detected, cube_center_world_frame)

        if file_name is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(f"{save_dir}/{file_name}.png", img)

    def test_detection(self, detected, estimated_pos = None):
        if detected:
            print("Cube detected. Estimated position:", estimated_pos)
            body_id = self.model.body('body').id
            body_position = self.data.xpos[body_id]
            print(body_position)

            error = np.linalg.norm(np.array(body_position) - np.array(estimated_pos))
            assert error < 0.05
