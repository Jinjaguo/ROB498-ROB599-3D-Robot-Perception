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

# camera_center = data['camera_center']
# camera_center_x = camera_center[:, 0].item()
# camera_center_y = camera_center[:, 1].item()
# camera_center_z = camera_center[:, 2].item()

# The pricipal points
x_c, y_c = 0.4224, -0.0300

# load images
rgb = data['rgb']
width, height = rgb.shape[1], rgb.shape[0]
mask = data['mask']
depth = data['depth']
depth = depth.reshape(depth.shape[0], depth.shape[1], 1)

point_cloud = []
colors = []
# Gather corresponding depths
for v in range(height):  # y坐标
        for u in range(width):  # x坐标
            if mask[v, u] > 0:
                z = depth[v, u]
                if z == 0:
                    continue  # skip invalid depths
                x = (u - x_c) * z / f_x
                y = (v - y_c) * z / f_y
                point_cloud.append([x, y, z])
                colors.append(rgb[v, u])
point_cloud = np.array(point_cloud)
point_cloud = np.squeeze(point_cloud, axis=-1)  # 从 (N, 3, 1) 变成 (N, 3)

colors = np.array(colors)

# 创建 open3d 点云对象
pcd = o3d.geometry.PointCloud()

# 设置点的位置
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 颜色归一化到 0-1
if colors.max() > 1.0:
    colors = colors / 255.0
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化
o3d.visualization.draw_geometries([pcd],
                                  window_name='Open3D Point Cloud',
                                  width=800, height=600,
                                  left=50, top=50,
                                  point_show_normal=False)


