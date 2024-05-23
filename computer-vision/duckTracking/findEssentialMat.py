import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    # Generating a random point cloud within specified spatial limits
    num_points = 100  # Reduced number of points for clarity
    points_3d = np.random.rand(num_points, 3)  # Random points in 3D
    # Scale and shift points to fit within the specified range
    points_3d[:, 0] = points_3d[:, 0] * 1000 - 500  # X: -500 to 500 mm
    points_3d[:, 1] = points_3d[:, 1] * 1000 - 500  # Y: -500 to 500 mm
    points_3d[:, 2] = points_3d[:, 2] * 4500 + 500   # Z: 500 to 5000 mm

    # Defining camera intrinsic parameters
    fov_x, fov_y = 80, 45  # Field of view in degrees for X and Y axes
    resolution_x, resolution_y = 1920, 1080  # Camera resolution
    # Focal lengths derived from field of view and sensor resolution
    fx = resolution_x / (2 * np.tan(np.deg2rad(fov_x) / 2))
    fy = resolution_y / (2 * np.tan(np.deg2rad(fov_y) / 2))
    # Assuming the principal point is at the center of the image
    cx, cy = resolution_x / 2, resolution_y / 2
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]  # Homogeneous component
    ])

    # Camera 1 (Origin) - no rotation or translation
    R1_identity = np.eye(3)  # Identity rotation matrix
    t1_zero = np.zeros((3, 1))  # Zero translation

    # Camera 2 - positioned 50mm to the right and rotated -20 degrees around Y-axis
    t2_translation = np.array([[50], [0], [0]])  # Translation vector
    R2_rotation = R.from_euler('y', -20, degrees=True).as_matrix()  # Rotation matrix

    # Constructing projection matrices without considering intrinsic parameters for simplicity
    P1_projection = np.dot(np.eye(3), np.hstack((R1_identity, t1_zero)))
    P2_projection = np.dot(np.eye(3), np.hstack((R2_rotation, t2_translation)))

    # Projecting 3D points to each camera's coordinate system
    points_3d_homogeneous = np.hstack((points_3d, np.ones((num_points, 1))))  # Homogeneous coordinates
    points_in_cam1 = (P1_projection @ points_3d_homogeneous.T).T
    points_in_cam2 = (P2_projection @ points_3d_homogeneous.T).T

    # Converting homogeneous coordinates to 2D image points
    points_in_cam1_2d = points_in_cam1[:, :2] / points_in_cam1[:, 2, np.newaxis]
    points_in_cam2_2d = points_in_cam2[:, :2] / points_in_cam2[:, 2, np.newaxis]

    # Estimating the essential matrix from point correspondences
    E_matrix, inliers = cv2.findEssentialMat(
        points_in_cam1_2d, points_in_cam2_2d,
        focal=1.0, pp=(0., 0.), method=cv2.LMEDS, prob=0.999, threshold=1.0
    )

    # Decomposing the essential matrix to find relative rotation and translation
    _, recovered_R_matrix, recovered_t_vector, _ = cv2.recoverPose(E_matrix, points_in_cam1_2d, points_in_cam2_2d)

    # Converting rotation matrices to Euler angles for easier interpretation
    recovered_rotation_euler = R.from_matrix(recovered_R_matrix).as_euler('xyz', degrees=True)
    groundtruth_rotation_euler = R.from_matrix(R2_rotation).as_euler('xyz', degrees=True)
    
    print('Recovered Rotation (Euler angles):', recovered_rotation_euler)
    print('Ground Truth Rotation (Euler angles):', groundtruth_rotation_euler)
    print (' Recovered Translation:', recovered_t_vector)
    print (' Ground Truth Translation:', t2_translation)
    
    # Calculating Camera 2's position relative to Camera 1
    camera2_position = - np.linalg.inv(R2_rotation) @ t2_translation
    camera2_orientation_in_world = R2_rotation.T  # Orientation relative to the world
    
    # Projection matrix of Camera 2 with intrinsic parameters
    P2_with_intrinsic = np.dot(intrinsic_matrix, np.hstack((R2_rotation, t2_translation)))
    M_orientation = P2_with_intrinsic[:, :3]  # Orientation component of the projection matrix
    principal_ray_direction = M_orientation[2, :]  # Camera 2's principal ray
    
    # Testing which rotation gives the correct principal ray direction
    incorrect_principal_ray = R2_rotation @ np.array([0, 0, 1])
    correct_principal_ray = R2_rotation.T @ np.array([0, 0, 1])
    
    print('Camera 2 Principal Ray Direction:', principal_ray_direction)
    print('Incorrect Principal Ray (Using R2):', incorrect_principal_ray)
    print('Correct Principal Ray (Using R2.T):', correct_principal_ray)
