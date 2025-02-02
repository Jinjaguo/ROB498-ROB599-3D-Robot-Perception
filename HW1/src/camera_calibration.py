import cv2
import os
import sys
import numpy as np


def calculate_projection(pts2d, pts3d):
    """
    Compute a 3x4 projection matrix M using a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the projection matrix M using the Direct Linear
    Transform (DLT) method. The projection matrix M relates the 3D world coordinates to
    their 2D image projections in homogeneous coordinates.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    M (numpy.ndarray): A 3x4 projection matrix M that relates 3D world coordinates to 2D
                   image points in homogeneous coordinates.
    """
    # Get the number of points
    N = pts2d.shape[0]
    A = []
    for i in range(N):
        # Get the 3D world coordinates and 2D image coordinates for the i-th point
        X, Y, Z = pts3d[i, :]
        x, y = pts2d[i, :]
        # Append the row to the A matrix
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    A = np.array(A)

    # Compute the SVD of the A matrix to obtain the M matrix
    U, S, Vt = np.linalg.svd(A)

    # The last column of Vt is the right singular vector corresponding to the smallest
    V = Vt[:, -1]
    M = np.reshape(V, (3, 4))

    # Normalize the M matrix to obtain a unitary matrix
    M = M / np.linalg.norm(M)
    return M


def calculate_reprojection_error(pts2d, pts3d):
    """
    Calculate the reprojection error for a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the reprojection error. The reprojection error is a
    measure of how accurately the 3D points project onto the 2D image plane when using a
    projection matrix.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    float: The reprojection error, which quantifies the accuracy of the 3D points'
           projection onto the 2D image plane.
    """
    # Compute the projection matrix M using the 2D-3D point correspondences
    M = calculate_projection(pts2d, pts3d)

    # Project the 3D points onto the image plane using the projection matrix
    pts3d_h = np.hstack([np.array(pts3d), np.ones((len(pts3d), 1))])

    # Project the 3D points onto the image plane using the projection matrix
    pts3d_projected = pts3d_h @ M.T

    # Normalize the projected 3D points to obtain the 2D image coordinates
    x_ = pts3d_projected[:, 0] / pts3d_projected[:, 2]
    y_ = pts3d_projected[:, 1] / pts3d_projected[:, 2]

    # Calculate the reprojection error for each 3D point
    diff = pts2d - np.column_stack([x_, y_])
    errors = np.linalg.norm(diff, axis=1)

    return errors.mean()


if __name__ == '__main__':
    data = np.load("./data/camera_calib_data.npz")
    pts2d = data['pts2d']
    pts3d = data['pts3d']

    P = calculate_projection(pts2d, pts3d)
    reprojection_error = calculate_reprojection_error(pts2d, pts3d)

    print(f"Projection matrix: {P}")
    print()
    print(f"Reprojection Error: {reprojection_error}")
