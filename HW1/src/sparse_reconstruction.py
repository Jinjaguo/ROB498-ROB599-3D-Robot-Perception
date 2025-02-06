import cv2
import os
import sys
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import helper_functions as _helper


def compute_fundamental_matrix(pts1, pts2, scale):
    """
    Compute the Fundamental matrix from corresponding 2D points in two images.

    Given two sets of corresponding 2D image points from Image 1 (pts1) and Image 2 (pts2),
    as well as a scaling factor (scale) representing the maximum dimension of the images,
    this function calculates the Fundamental matrix.

    Parameters:
    pts1 (numpy.ndarray): An Nx2 array containing 2D points from Image 1.
    pts2 (numpy.ndarray): An Nx2 array containing 2D points from Image 2, corresponding to pts1.
    scale (float): The maximum dimension of the images, used for scaling the Fundamental matrix.

    Returns:
    F (numpy.ndarray): A 3x3 Fundamental matrix
    """
    T = np.array([[1 / scale, 0, 0],
                  [0, 1 / scale, 0],
                  [0, 0, 1]])
    pts1_normalized = pts1 / scale
    pts2_normalized = pts2 / scale
    A = np.empty((pts1.shape[0], 9))
    # Create the A matrix for the linear system Ax = 0
    x1 = pts1_normalized[:, 0]
    y1 = pts1_normalized[:, 1]
    x2 = pts2_normalized[:, 0]
    y2 = pts2_normalized[:, 1]
    A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1,
                   y2, x1, y1, np.ones(pts1.shape[0]))).T

    # Compute the SVD of A to get the Fundamental matrix
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Make sure F is rank 2
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    # Scale the Fundamental matrix to match the original image size
    F = np.dot((np.dot(T.T, F)), T)

    return F


def compute_epipolar_correspondences(img1, img2, pts1, F):
    """
    Compute epipolar correspondences in Image 2 for a set of points in Image 1 using the Fundamental matrix.

    Given two images (img1 and img2), a set of 2D points (pts1) in Image 1, and the Fundamental matrix (F)
    that relates the two images, this function calculates the corresponding 2D points (pts2) in Image 2.
    The computed pts2 are the epipolar correspondences for the input pts1.

    Parameters:
    img1 (numpy.ndarray): The first image containing the points in pts1.
    img2 (numpy.ndarray): The second image for which epipolar correspondences will be computed.
    pts1 (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates img1 and img2.

    Returns:
    pts2_ep (numpy.ndarray): An Nx2 array of corresponding 2D points (pts2) in Image 2, serving as epipolar correspondences
                   to the points in Image 1 (pts1).
    """
    pts2_ep = []
    window_size = 11
    half_win = window_size // 2
    for i in range(pts1.shape[0]):
        # Compute the epipolar line in Image 2 for the i-th point in Image 1
        x, y = pts1[i]
        m = np.array([x, y, 1])

        l_ = F @ m

        x_int, y_int = int(round(x)), int(round(y))
        # Check if the epipolar line intersects the image boundary
        if (y_int - half_win < 0 or y_int + half_win >= img1.shape[0] or
                x_int - half_win < 0 or x_int + half_win >= img1.shape[1]):
            pts2_ep.append((0, 0))
            continue
        template = img1[y_int - half_win: y_int + half_win + 1, x_int - half_win: x_int + half_win + 1]

        min_cost = float('inf')
        best_point = None
        search_range = 20
        # Find the epipolar line in Image 2 that intersects the template
        for u in range(max(0, x_int - search_range), min(img2.shape[1], x_int + search_range)):
            if abs(l_[1]) < 1e-6:
                continue
            v = -(l_[0] * u + l_[2]) / l_[1]
            v_int = int(round(v))

            # Check if the epipolar line intersects the image boundary
            if (v_int - half_win < 0 or v_int + half_win >= img2.shape[0] or
                    u - half_win < 0 or u + half_win >= img2.shape[1]):
                continue

            candidate = img2[v_int - half_win: v_int + half_win + 1, u - half_win: u + half_win + 1]

            # cost = np.sum(np.abs(candidate.astype(np.float32) - template.astype(np.float32)))
            cost = np.sum((candidate - template) ** 2)
            if cost < min_cost:
                min_cost = cost
                best_point = (u, v_int)

        if best_point is None:
            best_point = (0, 0)
        pts2_ep.append(best_point)

    pts2_ep = np.array(pts2_ep)
    return pts2_ep


def compute_essential_matrix(K1, K2, F):
    """
    Compute the Essential matrix from the intrinsic matrices and the Fundamental matrix.

    Given the intrinsic matrices of two cameras (K1 and K2) and the 3x3 Fundamental matrix (F) that relates
    the two camera views, this function calculates the Essential matrix (E).

    Parameters:
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates Camera 1 and Camera 2.

    Returns:
    E (numpy.ndarray): The 3x3 Essential matrix (E) that encodes the essential geometric relationship between
                   the two cameras.

    """
    E = np.dot((np.dot(K2.T, F)), K1)

    return E


def triangulate_points(E, pts1_ep, pts2_ep, k_1, k_2):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity.
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    # Decompose the Essential matrix into R1, R2, and t
    E = E.astype(np.float32)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.reshape(3, 1)

    # Extrinsic matrix for camera1 is assumed to be Identity.
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    candidate_P2 = []
    candidate_P2.append(np.hstack([R1, t]))  # Candidate 1: [R1 |  t]
    candidate_P2.append(np.hstack([R2, t]))  # Candidate 2: [R2 |  t]
    candidate_P2.append(np.hstack([R1, -t]))  # Candidate 3: [R1 | -t]
    candidate_P2.append(np.hstack([R2, -t]))  # Candidate 4: [R2 | -t]

    def linear_triangulation(P1, P2, m1, m2):
        x1, y1 = m1[0], m1[1]
        x2, y2 = m2[0], m2[1]
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]
        return X[0:3]

    # define a helper function to check if a point is in front of a camera
    def is_in_front(P, X):
        X_h = np.hstack([X, 1])
        x_proj = P @ X_h
        if abs(x_proj[-1]) < 1e-6:
            return False
        x_proj = x_proj / x_proj[-1]
        return x_proj[2] > 0

    best_count = -1
    best_P2 = None
    best_points = []
    num_points = pts1_ep.shape[0]

    # for each candidate P2, triangulate points and count the number of points in front of both cameras
    for P2 in candidate_P2:
        points_candidate = []
        count = 0
        for i in range(num_points):
            m1 = np.array([pts1_ep[i, 0], pts1_ep[i, 1], 1])
            m2 = np.array([pts2_ep[i, 0], pts2_ep[i, 1], 1])
            X = linear_triangulation(P1, P2, m1, m2)
            points_candidate.append(X)
            if is_in_front(P1, X) and is_in_front(P2, X):
                count += 1
        if count > best_count:
            best_count = count
            best_P2 = P2
            best_points = points_candidate

    print("Best candidate P:")
    print(best_P2)
    point_cloud = np.array(best_points)  # Nx3

    pts1_h = pts1_ep[:, :2].T.astype(np.float32)
    pts2_h = pts2_ep[:, :2].T.astype(np.float32)  # (2, N)
    P1_cv = P1.astype(np.float32)
    P2_cv = best_P2.astype(np.float32)
    pts4D = cv2.triangulatePoints(P1_cv, P2_cv, pts1_h, pts2_h)  # 4 x N
    pts4D = pts4D / pts4D[3, :]
    point_cloud_cv = pts4D[:3, :].T  # N x 3

    print("Triangulated 3D points (custom linear triangulation):\n", point_cloud)
    print("Triangulated 3D points (cv2.triangulatePoints):\n", point_cloud_cv)

    P_1 = k_1 @ P1  # (3, 4)
    P_1_cv = k_1 @ P1_cv
    P_2 = k_2 @ best_P2  # (3, 4)
    P_2_cv = k_2 @ P2_cv

    def reprojection_error(point_cloud, P, pts):
        ones_pc = np.ones((point_cloud.shape[0], 1), dtype=point_cloud.dtype)

        points_3d_h = np.hstack((point_cloud, ones_pc))
        projected_pc = (P @ points_3d_h.T)  # (3, N)
        projected_pc = projected_pc.T  # (N, 3)
        projected_pc_2d = projected_pc[:, :2] / projected_pc[:, 2:]  # (N, 2)
        diff = projected_pc_2d - pts
        errors = np.linalg.norm(diff, axis=1)
        mean_error = np.mean(errors)
        return mean_error

    mean_error_pc_1 = reprojection_error(point_cloud, P_1, pts1_ep)
    mean_error_pc_2 = reprojection_error(point_cloud, P_2, pts2_ep)
    mean_error_pc_cv_1 = reprojection_error(point_cloud_cv, P_1_cv, pts1_ep)
    mean_error_pc_cv_2 = reprojection_error(point_cloud_cv, P_2_cv, pts2_ep)

    mean_error_pc_1 /= np.max(img1.shape)  # 归一化误差
    mean_error_pc_2 /= np.max(img2.shape)
    mean_error_pc_cv_1 /= np.max(img1.shape)
    mean_error_pc_cv_2 /= np.max(img2.shape)

    print("reprojection error for point cloud: ", np.mean([mean_error_pc_1, mean_error_pc_2]))
    print("reprojection error for cv point cloud: ", np.mean([mean_error_pc_cv_1, mean_error_pc_cv_2]))

    return point_cloud, point_cloud_cv


def visualize(point_cloud):
    """
    Function to visualize 3D point clouds
    Parameters:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud,where each row contains the 3D coordinates
                   of a triangulated point.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # Show the plot
    plt.show()


if __name__ == "__main__":
    data_for_fundamental_matrix = np.load("./data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = plt.imread('./data/im1.png')
    img2 = plt.imread('./data/im2.png')
    scale = np.max(img1.shape)

    data_for_temple = np.load("data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']

    # Compute the Fundamental matrix
    F = compute_fundamental_matrix(pts1_for_fundamental_matrix, pts2_for_fundamental_matrix, scale)
    print("Computed Fundamental Matrix:\n", F)

    # Use the GUI tool to visualize the epipolar lines and epipolar correspondences

    # uncomment the following lines to use the GUI tool
    # _helper.epipolar_lines_GUI_tool(img1, img2, F)
    # _helper.epipolar_correspondences_GUI_tool(img1, img2, F)

    E = compute_essential_matrix(K1, K2, F)
    print("Computed Essential Matrix:\n", E)

    N = pts1_epipolar.shape[0]
    pts2_epipolar = compute_epipolar_correspondences(img1, img2, pts1_epipolar, F)

    ################################################################################
    # Here we will visualize the epipolar lines and epipolar correspondences

    N1 = pts1_epipolar.shape[0]
    colors1 = plt.cm.jet(np.linspace(0, 1, N1))

    N2 = pts2_epipolar.shape[0]
    colors2 = plt.cm.jet(np.linspace(0, 1, N2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # show the original images and points
    axes[0].imshow(img1)
    axes[0].scatter(pts1_epipolar[:, 0], pts1_epipolar[:, 1],
                    c=colors1, s=40, marker='o')
    axes[0].set_title("Points in Image 1")
    axes[0].axis('off')

    # show the epipolar lines and epipolar correspondences
    axes[1].imshow(img2)
    axes[1].scatter(pts2_epipolar[:, 0], pts2_epipolar[:, 1],
                    c=colors2, s=40, marker='o')
    axes[1].set_title("Corresponding points in Image 2")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    ################################################################################

    # here we get the homogeneous coordinates of the points
    pts1_hom = np.hstack([pts1_epipolar, np.ones((N, 1))])  # (N, 3)
    pts2_hom = np.hstack([pts2_epipolar, np.ones((N, 1))])  # (N, 3)

    pts1_hom_T = pts1_hom.T  # (3, N)
    pts2_hom_T = pts2_hom.T  # (3, N)

    # normalize the points by using intrinsic matrices
    pts1_normalized = np.dot(np.linalg.inv(K1), pts1_hom_T).T  # (N, 3)
    pts2_normalized = np.dot(np.linalg.inv(K2), pts2_hom_T).T  # (N, 3)

    # Here we need N,2 as input for triangulatePoints()
    pts1_normalized = pts1_normalized[:, :2]  # (N, 2)
    pts2_normalized = pts2_normalized[:, :2]  # (N, 2)

    point_cloud, point_cloud_cv = triangulate_points(E, pts1_normalized, pts2_normalized, K1, K2)
    # comment the previous GUI line to see point cloud visualization.

    # visualize(point_cloud)
    # visualize(point_cloud_cv)
