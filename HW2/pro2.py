import numpy as np
import open3d as o3d

pcd1 = np.loadtxt('./data/point_cloud_X.txt')
pcd2 = np.loadtxt('./data/point_cloud_Y.txt')

# Initializing the parameters
initial_R = np.eye(3)
initial_t = np.zeros((1,3))

def icp(X=pcd1, Y=pcd2, t0=initial_t, R0=initial_R , d_max=0.25, max_iters=30, tol=1e-6):
    t_hat = t0.copy()
    R_hat = R0.copy()

    for iter in range(max_iters):
        print(f"Iteration {iter+1}")
        C = [] # Correspondences
        for i, xi in enumerate(X):
            xi_trans = R_hat @ xi + t_hat
            dists = np.linalg.norm(Y - xi_trans, axis=1)
            # find the closest point in Y to xi_trans
            j = np.argmin(dists)
            if dists[j] < d_max:
                C.append((i, j))

        if not C:
            break

        X_corr = np.array([X[i] for i, j in C])
        Y_corr = np.array([Y[j] for i, j in C])

        R_new, t_new = compute_optimal_rigid_registration(X_corr, Y_corr)

        if np.linalg.norm(t_hat - t_new) < tol and np.linalg.norm(R_hat - R_new) < tol:
            break

        t_hat = t_new
        R_hat = R_new

    return t_hat, R_hat, C

def compute_optimal_rigid_registration(X_corr, Y_corr):
    X_mean = X_corr.mean(axis=0)
    Y_mean = Y_corr.mean(axis=0)
    X_centered = X_corr - X_mean
    Y_centered = Y_corr - Y_mean

    H = X_centered.T @ Y_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Y_mean - R @ X_mean
    return R, t

t_hat, R_hat, C = icp()
pcd1_aligned = (R_hat @ pcd1.T).T + t_hat

pcd_o3d_1 = o3d.geometry.PointCloud()
pcd_o3d_1.points = o3d.utility.Vector3dVector(pcd1)

pcd_o3d_aligned = o3d.geometry.PointCloud()
pcd_o3d_aligned.points = o3d.utility.Vector3dVector(pcd1_aligned)

pcd_o3d_2 = o3d.geometry.PointCloud()
pcd_o3d_2.points = o3d.utility.Vector3dVector(pcd2)

# o3d.visualization.draw_geometries([pcd_o3d_1, pcd_o3d_2])
# o3d.visualization.draw_geometries([pcd_o3d_1, pcd_o3d_aligned])
o3d.visualization.draw_geometries([pcd_o3d_2, pcd_o3d_aligned])

# calculate the error
rmse = np.sqrt(np.mean(np.sum((pcd1_aligned - pcd2)**2, axis=1)))
print(f"Error: {rmse}")