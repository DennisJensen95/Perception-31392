import open3d as o3d

source = o3d.io.read_point_cloud("out.ply")
o3d.visualization.draw_geometries([source])
