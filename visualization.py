import os
import numpy as np
import open3d as o3d

def save_trajectory(points_xyz, out_txt_path):
    """
    Save matched LiDAR-GNSS trajectory points to a txt file (x y z per line).
    points_xyz: iterable of [x, y, z]
    """
    arr = np.asarray(points_xyz, dtype=np.float64)
    np.savetxt(out_txt_path, arr, fmt="%.6f")

def convert_xyz_to_txt(xyz_path, out_txt_path):
    """
    Convert merged_global.xyz (or any .xyz) to a plain txt with x y z per line.
    """
    pts = np.loadtxt(xyz_path, dtype=np.float64)
    np.savetxt(out_txt_path, pts, fmt="%.6f")

def plot_pointcloud_with_trajectory(xyz_path, traj_txt_path, line_color=(1.0, 0.0, 0.0)):
    """
    Visualize the point cloud with a superimposed trajectory line in Open3D.
    xyz_path: path to point cloud .xyz (x y z per line)
    traj_txt_path: path to trajectory .txt (x y z per line)
    """
    # Load point cloud
    pc_points = np.loadtxt(xyz_path, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points)

    # Load trajectory points
    traj_points = np.loadtxt(traj_txt_path, dtype=np.float64)
    traj_points = np.atleast_2d(traj_points)

    # Build line geometry
    traj_geom = o3d.geometry.LineSet()
    traj_geom.points = o3d.utility.Vector3dVector(traj_points)
    lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
    traj_geom.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.asarray(line_color), (len(lines), 1))
    traj_geom.colors = o3d.utility.Vector3dVector(colors)

    # Visualize together
    o3d.visualization.draw_geometries([pcd, traj_geom],
                                      window_name="Point Cloud + Trajectory",
                                      width=1280, height=720)

if __name__ == "__main__":
    # Paths produced by icp_pcd_gnss (icp copy.py): adjust if needed
    base = "/home/samantha/ELTE/ICP"
    xyz_in = os.path.join(base, "parkolo1/outputs/merged_global.xyz")
    traj_txt = os.path.join(base, "parkolo1/outputs/trajectory_icp.txt")
    plot_pointcloud_with_trajectory(xyz_in, traj_txt)