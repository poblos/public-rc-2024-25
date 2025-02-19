import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os

CALIBRATION_FILES = glob.glob("calibration/*.png")
STITCHING_FILES = glob.glob("stiching/*.png")
UNDISTORTED_FILES = "./undistorted"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
original_order = [29, 24, 19, 28, 23, 18]
marker_length = 168
spacing = 70
num_rows = 2
num_cols = 3


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
        imgpoints[i] = imgpoints[i].reshape(-1, 1, 2)
        
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    return total_error / len(objpoints)

def calibrateCamera(reference_points):
    images = [cv2.imread(img) for img in CALIBRATION_FILES]
    parameters = cv2.aruco.DetectorParameters()
    objpoints = []
    imgpoints = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        ids = ids.flatten()

        indexed_corners = {k: corner for k, corner in zip(ids, corners)}
        sorted_corners = [indexed_corners[k] for k in original_order]
        sorted_corners = np.array(sorted_corners).reshape(24, -1)

        imgpoints.append(sorted_corners)
        objpoints.append(reference_points)

    objpoints = np.array(objpoints, dtype = np.float32)
    imgpoints = np.array(imgpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    #error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    #print(f"Reprojection Error: {error}")
    print(mtx)
    print(dist)

    return ret, mtx, dist, rvecs, tvecs

def calibrateCameraMethodB():
    marker_positions = []
    for j in range(num_rows):
        for i in range(num_cols):
            marker_positions.extend([
                [j * (marker_length + spacing), i * (marker_length + spacing), 0],
                [(j + 1) * marker_length + j * spacing, i * (marker_length + spacing), 0],
                [(j + 1) * marker_length + j * spacing, (i + 1) * marker_length + i * spacing, 0],
                [j * (marker_length + spacing), (i + 1) * marker_length + i * spacing, 0]
            ])

    reference_points = np.array(marker_positions)

    return calibrateCamera(reference_points)


def calibrateCameraMethodA():
    '''When the point is further away from the centre, radial distortion increases.
    If we don't include distances between points, we naturally are worse at estimating
    what radial distortion coefficients are and our equations become underconstrained.
    Therefore, calibration error gets bigger with this method.'''
    reference_points = np.stack((np.array([
    [-marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, marker_length / 2, 0],
    [marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
    ]),) * 6, axis=0).reshape(-1,3)
    
    return calibrateCamera(reference_points)

def apply_projective_transformation(image, H):
    h, w = image.shape[:2]

    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [w - 1, h - 1, 1],
        [0, h - 1, 1]
    ]).T

    transformed_corners = H @ corners
    transformed_corners /= transformed_corners[2, :]
    min_x, max_x = transformed_corners[0].min(), transformed_corners[0].max()
    min_y, max_y = transformed_corners[1].min(), transformed_corners[1].max()

    max_x = int(np.ceil(max_x))
    max_y = int(np.ceil(max_y))
    min_x = int(np.floor(min_x))
    min_y = int(np.floor(min_y))

    transformed_image = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)

    H_inv = np.linalg.inv(H)

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            dst_pt = np.array([x, y, 1]).reshape(3, 1)
            src_pt = H_inv @ dst_pt
            src_pt /= src_pt[2, 0]

            src_x, src_y = int(round(src_pt[0, 0])), int(round(src_pt[1, 0]))
            if 0 <= src_x < w and 0 <= src_y < h:
                transformed_image[y- min_y, x- min_x] = image[src_y, src_x]

    return (transformed_image, min_x, min_y)

def find_projective_transformation(src_points, dst_points):
    A = []
    for (x, y), (x_prime, y_prime) in zip(src_points, dst_points):
        A.append([-x, -y, -1,  0,  0,  0, x * x_prime, y * x_prime, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    
    A = np.array(A)
    
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)
    
    return H

def test_find_projective_transformation(num_tests = 10, num_points = 8):
    np.random.seed(42)

    for _ in range(num_tests):
        true_H = np.random.rand(3, 3)
        true_H /= true_H[-1, -1]
        
        src_points = np.random.rand(num_points, 2) * 100
    
        src_points_homogeneous = np.hstack([src_points, np.ones((num_points, 1))])

        dst_points_homogeneous = (true_H @ src_points_homogeneous.T).T
        dst_points = dst_points_homogeneous[:, :2] / dst_points_homogeneous[:, 2][:, None]
        
        estimated_H = find_projective_transformation(src_points, dst_points)
        estimated_H /= estimated_H[-1, -1]
        
        assert np.allclose(true_H, estimated_H, atol=1e-6), "Projective transformation wrong"
    
    print("Projective transformation OK")

def extract_minimum_path_bottom_up(costs):
    h, w = costs.shape
    path = []
    min_cost_col = np.argmin(costs[-1])
    path.append((h - 1, min_cost_col))
    
    for y in range(h - 2, -1, -1):
        x = path[-1][1]
        x_range = [max(0, x - 1), x, min(w - 1, x + 1)]
        x_next = x_range[np.argmin([costs[y, i] for i in x_range])]
        path.append((y, x_next))
    
    return path[::-1]


def stitch(img1, img2, H):
    img1, min_x, min_y = apply_projective_transformation(img1, H)

    # Padding left and up to match coordinate system
    img1_padded = np.pad(
        img1,
        pad_width=((max(0, min_y), 0), (max(0, min_x), 0), (0, 0)),
        mode='constant',
        constant_values=0
    )

    img2_padded = np.pad(
        img2,
        pad_width=((abs(min(0, min_y)), 0), (abs(min(0, min_x)), 0), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Padding right and down to match dimensions
    x_max_diff = img1_padded.shape[1] - img2_padded.shape[1]
    y_max_diff = img1_padded.shape[0] - img2_padded.shape[0]

    img1_padded = np.pad(
        img1_padded,
        pad_width=((0, abs(min(0, y_max_diff))), (0, abs(min(0, x_max_diff))), (0, 0)),
        mode='constant',
        constant_values=0
    )

    img2_padded = np.pad(
        img2_padded,
        pad_width=((0, max(0, y_max_diff)), (0, max(0, x_max_diff)), (0, 0)),
        mode='constant',
        constant_values=0
    )

    overlap_mask = (img1_padded.sum(axis=2) > 0) & (img2_padded.sum(axis=2) > 0)

    # Compute the difference image
    diff = abs(img1_padded - img2_padded)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) ** 2

    diff = diff.astype(np.float32)
    diff[~overlap_mask] = 0

    central_region = (diff.shape[1] // 3, 2 * diff.shape[1] // 3)
    offset_x = central_region[0]

    diff = diff [:, central_region[0]:central_region[1]]

    costs = np.zeros_like(diff)
    for y in range(1, costs.shape[0]):
        for x in range(0, costs.shape[1]):
            costs[y, x] = diff[y, x] + min(costs[y - 1, max(x - 1,0)], costs[y - 1, x], costs[y - 1, min(costs.shape[1] - 1,x + 1)])

    plt.imshow(costs)
    plt.axis('off')
    plt.show()
    seam = extract_minimum_path_bottom_up(costs)

    stitched_img = np.zeros_like(img1_padded)
    

    for y,x in seam:
        global_x = x + offset_x
        if (min_x < 0):
            stitched_img[y, :global_x] = img1_padded[y, :global_x]
            stitched_img[y, global_x:] = img2_padded[y, global_x:]
        else: 
            stitched_img[y, :global_x] = img2_padded[y, :global_x]
            stitched_img[y, global_x:] = img1_padded[y, global_x:]
    stitched_img[~overlap_mask] = img2_padded[~overlap_mask] + img1_padded[~overlap_mask]

    return stitched_img

def task1():
    calibrateCameraMethodA() 
    ret, mtx, dist, rvecs, tvecs = calibrateCameraMethodB()
    images = [cv2.imread(img) for img in STITCHING_FILES]

    for i, image in enumerate(images):
        h, w = image.shape[:2]
        new_mtx_b, roi_b = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)

        undistorted_b = cv2.undistort(image, mtx, dist, None, new_mtx_b)

        output_filename = os.path.join(UNDISTORTED_FILES, f"img{i+1}.png")
        print(output_filename)
        cv2.imwrite(output_filename, undistorted_b)
        print(f"Saved undistorted image: {output_filename}")


def task2(img = "stiching/img1.png", H = np.array([[0.5,0,0],[0,2,0],[0,0,1]],dtype=np.float64)):
    img = cv2.imread(img)

    img2, _, _ = apply_projective_transformation(img, H)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def task3():
    test_find_projective_transformation()

def task45():
    img1 = "stiching/img1.png"
    img2 = "stiching/img2.png"
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    #top_left, top_rigth, bottom_left, bottom_right of tuna box; top_left, top_right, for white box, top left corner of pi letter; blackdot
    img1_points = [[294,285],[337,286],[296,328],[338,327],[457,424],[518,374],[806,474],[119,88]] #[478,452],[539,401]
    img2_points = [[419,297],[458,297],[420,336],[459,335],[569,429],[628,381],[915,481],[258,112]] #[590,456],[647,407]
    H = find_projective_transformation(img1_points, img2_points)

    stitched_img = stitch(img1,img2, H)

    output_filename = "./output/task5.png"
    cv2.imwrite("./output/task5.png", stitched_img)
    print(f"Saved image: {output_filename}")

def task6(img1_name = "/img1.png", img2_name = "/img2.png"):
    img1 = cv2.imread(UNDISTORTED_FILES + img1_name)
    img2 = cv2.imread(UNDISTORTED_FILES + img2_name)

    superglue_output = "./output/" + img1_name[1:-4] + "_" + img2_name[1:-4] + "_matches.npz"
    data = np.load(superglue_output)

    keypoints0 = data['keypoints0']
    keypoints1 = data['keypoints1']
    matches = data['matches']

    img1_points = keypoints0[matches != -1]
    img2_points = keypoints1[matches[matches != -1]]

    H = find_projective_transformation(img1_points, img2_points)

    stitched_img = stitch(img1, img2, H)

    output_filename = "./output/task6.png"
    cv2.imwrite("./output/task6.png", stitched_img)
    print(f"Saved image: {output_filename}")


#undistort(reference_points_b)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Task number to run")
    args = parser.parse_args()

    if args.task == 1:
        task1()
    elif args.task == 2:
        task2()
    elif args.task == 3:
        task3()
    elif args.task == 4 or args.task == 5:
        task45()
    elif args.task == 6:
        task6()
    elif args.task == 7:
        task6("/output34.png", "/output765.png")

if __name__ == "__main__":
    main()