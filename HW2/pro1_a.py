import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

path = './data/generate_point_cloud.npz'

data = np.load(path)
print(data.files)

# load camera parameters
focal_length = data['focal_length']
f_x = focal_length[:, 0]
f_y = focal_length[:, 1]

camera_center = data['camera_center']
camera_center_x = camera_center[:, 0].item()
camera_center_y = camera_center[:, 1].item()
camera_center_z = camera_center[:, 2].item()

# The pricipal points
x_c, y_c = 0.4224, -0.0300

# load images
rgb = data['rgb']
width, height = rgb.shape[1], rgb.shape[0]
mask = data['mask']
depth = data['depth']
depth = depth.reshape(depth.shape[0], depth.shape[1], 1)

# convert focal length and camera center to pixel values
f_x = f_x * width
f_y = f_y * height
c_x = x_c * width
c_y = y_c * height

# concatenate rgb and depth
rgb_d = np.concatenate((rgb, depth), axis=-1)
# expand the mask to 3 channels
mask_expanded = np.expand_dims(mask, axis=-1)

# create point cloud
pcd = []

# take first layer of rgb image
for u in range(rgb.shape[1]):
    for v in range(rgb.shape[0]):
        if mask[v, u] == 1:
            # get depth value
            d = depth[v, u, 0]
            valid = d > 0
            if valid:
                # get 3D point
                recon_x = (((u - c_x) * d) / f_x).item()
                recon_y = (((v - c_y) * d) / f_y).item()
                recon_z = d.item()
                # print("Reconstructed 3D point: ({:.2f}, {:.2f}, {:.2f})".format(recon_x, recon_y, recon_z))
                pcd_world = np.array([recon_x, recon_y, recon_z]) + np.array([camera_center_x, camera_center_y, camera_center_z])
                pcd.append(pcd_world)

pcd = np.array(pcd)


# visualize point cloud
o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))])
